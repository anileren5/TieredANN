#include <omp.h>
#include <boost/program_options.hpp>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>

// TieredANN headers (for Greator disk index and DiskANN utils)
#include "tieredann/tiered_index.h"
// Include for SECTOR_LEN setting
#include "greator/pq_flash_index.h"

namespace po = boost::program_options;

template <typename T, typename TagT = uint32_t>
void calculate_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    for (int32_t i = 0; i < query_num; i++) {
        std::set<uint32_t> groundtruth_closest_neighbors;
        std::set<uint32_t> calculated_closest_neighbors;
        for (int32_t j = 0; j < K; j++) {
            calculated_closest_neighbors.insert(*(query_result_tags.data() + i * K + j));
            groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
        }
        uint32_t matching_neighbors = 0;
        for (uint32_t x : calculated_closest_neighbors) if (groundtruth_closest_neighbors.count(x)) matching_neighbors++;
        double recall = matching_neighbors / (double)K;
        total_recall += recall;        
    }
    double average_recall = total_recall / (query_num);
    std::cout << K << "Recall@" << K << ": " << average_recall * 100 << "%" << std::endl;
}

template <typename T, typename TagT = uint32_t>
void disk_search(
    std::unique_ptr<greator::PQFlashIndex<T>>& disk_index,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t L, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags, std::vector<T *>& res,
    uint32_t beamwidth, const std::string& data_path
) {
    std::vector<float> query_result_dists(K * query_num);
    greator::QueryStats* stats = new greator::QueryStats[query_num];
    std::vector<double> latencies_ms(query_num, 0.0);
    auto global_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        disk_index->cached_beam_search(
            query + i * query_aligned_dim,
            (uint64_t)K,
            (uint64_t)L,
            query_result_tags.data() + i * K,
            query_result_dists.data() + i * K,
            (uint64_t)beamwidth,
            stats + i
        );
        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    auto global_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(global_end - global_start).count();
    double total_time_sec = total_time_ms / 1000.0;
    double avg_latency_ms = 0.0;
    for (double latency : latencies_ms) avg_latency_ms += latency;
    avg_latency_ms /= query_num;
    double qps = static_cast<double>(query_num) / total_time_sec;
    double qps_per_thread = qps / static_cast<double>(search_threads);

    // Calculate tail latencies
    std::vector<double> sorted_latencies = latencies_ms;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    auto get_percentile = [&](double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * query_num)) - 1;
        if (idx >= query_num) idx = query_num - 1;
        return sorted_latencies[idx];
    };
    double p90 = get_percentile(0.90);
    double p95 = get_percentile(0.95);
    double p99 = get_percentile(0.99);

    std::cout << "queries=" << query_num
              << " threads=" << search_threads
              << " total_time_ms=" << total_time_ms
              << " avg_latency_ms=" << avg_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << " tail_latency_ms(p90,p95,p99)=" << p90 << ", " << p95 << ", " << p99
              << std::endl;

    delete[] stats;
}

template <typename T = float, typename TagT = uint32_t>
void experiment(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t L, uint32_t K,
    uint32_t B, uint32_t M,
    uint32_t build_threads,
    uint32_t search_threads,
    int disk_index_already_built,
    uint32_t beamwidth,
    int n_search_iter
) {
    // Build disk index
    if (disk_index_already_built == 0) {
        std::string disk_index_params = std::to_string(R) + " " + std::to_string(L) + " " + std::to_string(B) + " " + std::to_string(M) + " " + std::to_string(build_threads);
        greator::build_disk_index<T>(data_path.c_str(), disk_index_prefix.c_str(), disk_index_params.c_str(), greator::Metric::L2, false);
    }

    // Load disk index
    std::shared_ptr<greator::AlignedFileReader> reader = nullptr;
    reader.reset(new greator::LinuxAlignedFileReader());
    std::unique_ptr<greator::PQFlashIndex<T>> disk_index(new greator::PQFlashIndex<T>(greator::Metric::L2, reader, false, false));
    disk_index->load(disk_index_prefix.c_str(), build_threads);

    // Cache vectors near the centroid of the disk index.
    std::vector<uint32_t> node_list;
    disk_index->cache_bfs_levels(500, node_list);
    disk_index->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    // Load groundtruth ids for the results
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

    // Load queries
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);

    // I don't know why it is need but it is somehow necessary to call seach_with_tags.
    std::vector<T *> res = std::vector<T *>();

    // Allocate the space to store result of the queries
    std::vector<TagT> query_result_tags(query_num * K);

    for (int i = 0; i < n_search_iter; i++) {
        disk_search(disk_index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res, beamwidth, data_path);
        calculate_recall<T, TagT>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim); 
        query_result_tags.clear();
    }

    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, disk_index_prefix;
    uint32_t R, L, K, B, M, build_threads, search_threads, beamwidth;
    int disk_index_already_built;
    int n_search_iter;
    uint32_t sector_len = 4096; // Default value

    po::options_description desc("Allowed options");
    try {
        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix to index")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("L", po::value<uint32_t>(&L)->required(), "Value of L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("n_search_iter", po::value<int>(&n_search_iter)->default_value(100), "Number of search iterations")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    std::cout << "===== Program Parameters =====" << std::endl;
    std::cout << "data_type: " << data_type << std::endl;
    std::cout << "data_path: " << data_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "groundtruth_path: " << groundtruth_path << std::endl;
    std::cout << "disk_index_prefix: " << disk_index_prefix << std::endl;
    std::cout << "R: " << R << std::endl;
    std::cout << "L: " << L << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "B: " << B << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "build_threads: " << build_threads << std::endl;
    std::cout << "search_threads: " << search_threads << std::endl;
    std::cout << "beamwidth: " << beamwidth << std::endl;
    std::cout << "disk_index_already_built: " << disk_index_already_built << std::endl;
    std::cout << "n_search_iter: " << n_search_iter << std::endl;
    std::cout << "sector_len: " << sector_len << std::endl;
    std::cout << "==============================" << std::endl << std::endl;

    // Set the global SECTOR_LEN variable
    set_sector_len(sector_len);
    
    experiment(
        data_type, data_path, query_path, groundtruth_path, disk_index_prefix,
        R, L, K, B, M, build_threads, search_threads, disk_index_already_built, beamwidth, n_search_iter
    );
}