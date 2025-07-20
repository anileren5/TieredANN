// System headers
#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>

// TieredANN headers
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
        for (uint32_t x : calculated_closest_neighbors) if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
        double recall = matching_neighbors / (double)K;
        total_recall += recall;        
    }
    double average_recall = total_recall / (query_num);

    std::cout << K << "Recall@" << K << ": " << average_recall * 100 << "%" << std::endl;
}

template <typename T, typename TagT = uint32_t>
void calculate_hit_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, 
                         const std::vector<bool>& hit_results, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    size_t hit_count = 0;

    for (int32_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {  // Only calculate recall for cache hits
            std::set<uint32_t> groundtruth_closest_neighbors;
            std::set<uint32_t> calculated_closest_neighbors;
            for (int32_t j = 0; j < K; j++) {
                calculated_closest_neighbors.insert(*(query_result_tags.data() + i * K + j));
                groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
            }
            uint32_t matching_neighbors = 0;
            for (uint32_t x : calculated_closest_neighbors) if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
            double recall = matching_neighbors / (double)K;
            total_recall += recall;
            hit_count++;
        }
    }
    
    if (hit_count > 0) {
        double average_recall = total_recall / hit_count;
        std::cout << K << "Recall@" << K << " (cache hits only): " << average_recall * 100 << "% (" << hit_count << " hits)" << std::endl;
    } else {
        std::cout << K << "Recall@" << K << " (cache hits only): N/A (no hits)" << std::endl;
    }
}

template <typename T, typename TagT = uint32_t>
std::vector<bool> hybrid_search(
    tieredann::TieredIndex<T>& tiered_index,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t L, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags, std::vector<T *>& res,
    uint32_t beamwidth, const std::string& data_path
) {
    std::vector<float> query_result_dists(K * query_num);
    greator::QueryStats* stats = new greator::QueryStats[query_num];
    std::vector<double> latencies_ms(query_num, 0.0);
    std::vector<bool> hit_results(query_num, false);
    auto global_start = std::chrono::high_resolution_clock::now();

    std::atomic<size_t> hit_count{0};
    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        bool hit = tiered_index.search(
            query + i * query_aligned_dim,
            K,
            L,
            query_result_tags.data() + i * K,
            res,
            beamwidth,
            query_result_dists.data() + i * K,
            stats + i
        );
        hit_results[i] = hit;
        if (hit) hit_count.fetch_add(1, std::memory_order_relaxed);
        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Calculate average hit latency
    double total_hit_latency_ms = 0.0;
    size_t actual_hit_count = 0;
    for (size_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            total_hit_latency_ms += latencies_ms[i];
            actual_hit_count++;
        }
    }
    double avg_hit_latency_ms = (actual_hit_count > 0) ? total_hit_latency_ms / actual_hit_count : 0.0;
    
    // Print final hit ratio
    double final_ratio = static_cast<double>(hit_count.load(std::memory_order_relaxed)) / query_num;
    std::cout << "[TieredIndex] Hit ratio: " << final_ratio << " (" << hit_count << "/" << query_num << ")" << std::endl;

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

    std::cout << " threads=" << search_threads
              << " avg_latency_ms=" << avg_latency_ms
              << " avg_hit_latency_ms=" << avg_hit_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << " #memory_vectors=" << tiered_index.get_number_of_vectors_in_memory_index()
              << " tail_latency_ms(p90,p95,p99)=" << p90 << ", " << p95 << ", " << p99
              << std::endl;

    delete[] stats;
    return hit_results;
}

template <typename T = float, typename TagT = uint32_t>
void experiment(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t memory_L, uint32_t disk_L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha,
    uint32_t consolidate_threads,
    uint32_t build_threads,
    uint32_t search_threads,
    int disk_index_already_built,
    uint32_t beamwidth, 
    int use_reconstructed_vectors,
    double p,
    double deviation_factor,
    uint32_t n_theta_estimation_queries,
    int n_search_iter,
    bool eager_theta_update
) {
   // Create a tiered index
   tieredann::TieredIndex<T> tiered_index(
       data_path, disk_index_prefix,
       R, memory_L, disk_L, B, M, alpha, 
       consolidate_threads, build_threads, search_threads,
       disk_index_already_built, (bool)use_reconstructed_vectors,
       p, deviation_factor, n_theta_estimation_queries, eager_theta_update
    );

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
        std::vector<bool> hit_results = hybrid_search(tiered_index, query, query_num, query_aligned_dim, K, memory_L, search_threads, query_result_tags, res, beamwidth, data_path);
        calculate_recall<T, TagT>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim); 
        calculate_hit_recall<T, TagT>(K, groundtruth_ids, query_result_tags, hit_results, query_num, groundtruth_dim);
        query_result_tags.clear();
    }

    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, disk_index_prefix;
    uint32_t R, memory_L, disk_L, K, B, M;
    uint32_t build_threads, consolidate_threads, search_threads, beamwidth;
    float alpha;
    int single_file_index, disk_index_already_built, use_reconstructed_vectors;
    double hit_rate;
    double p, deviation_factor;
    uint32_t n_theta_estimation_queries;
    int n_search_iter;
    uint32_t sector_len = 4096; // Default value
    bool eager_theta_update = false; // Default value

    po::options_description desc;

    // Take command line parameters
    try
    {
        po::options_description desc("Allowed options");

        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix to index")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("memory_L", po::value<uint32_t>(&memory_L)->required(), "Value of memory L")
            ("disk_L", po::value<uint32_t>(&disk_L)->required(), "Value of disk L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("consolidate_threads", po::value<uint32_t>(&consolidate_threads)->required(), "Threads for consolidation")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("use_reconstructed_vectors", po::value<int>(&use_reconstructed_vectors)->default_value(true), "Use reconstructed vectors for insertion to memory index")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("p", po::value<double>(&p)->default_value(0.75), "Value of p")
            ("deviation_factor", po::value<double>(&deviation_factor)->default_value(0.05), "Value of deviation factor")
            ("n_theta_estimation_queries", po::value<uint32_t>(&n_theta_estimation_queries)->default_value(1000), "Number of theta estimation queries")
            ("n_search_iter", po::value<int>(&n_search_iter)->default_value(100), "Number of search iterations")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes")
            ("eager_theta_update", po::value<bool>(&eager_theta_update)->default_value(false), "Update theta immediately (true) or defer to insertion (false)");


        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Set the global SECTOR_LEN variable
    set_sector_len(sector_len);
    
    // Run the experiment
    std::cout << "===== Program Parameters =====" << std::endl;
    std::cout << "data_type: " << data_type << std::endl;
    std::cout << "data_path: " << data_path << std::endl;
    std::cout << "query_path: " << query_path << std::endl;
    std::cout << "groundtruth_path: " << groundtruth_path << std::endl;
    std::cout << "disk_index_prefix: " << disk_index_prefix << std::endl;
    std::cout << "R: " << R << std::endl;
    std::cout << "memory_L: " << memory_L << std::endl;
    std::cout << "disk_L: " << disk_L << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "B: " << B << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "build_threads: " << build_threads << std::endl;
    std::cout << "consolidate_threads: " << consolidate_threads << std::endl;
    std::cout << "search_threads: " << search_threads << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "use_reconstructed_vectors: " << use_reconstructed_vectors << std::endl;
    std::cout << "disk_index_already_built: " << disk_index_already_built << std::endl;
    std::cout << "beamwidth: " << beamwidth << std::endl;
    std::cout << "p: " << p << std::endl;
    std::cout << "deviation_factor: " << deviation_factor << std::endl;
    std::cout << "n_theta_estimation_queries: " << n_theta_estimation_queries << std::endl;
    std::cout << "n_search_iter: " << n_search_iter << std::endl;
    std::cout << "sector_len: " << sector_len << std::endl;
    std::cout << "==============================" << std::endl << std::endl;

    if (data_type == "float") {
        experiment<float>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, disk_L, K, B, M, alpha, consolidate_threads, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, n_theta_estimation_queries, n_search_iter, eager_theta_update);
    } else if (data_type == "int8") {
        experiment<int8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, disk_L, K, B, M, alpha, consolidate_threads, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, n_theta_estimation_queries, n_search_iter, eager_theta_update);
    } else if (data_type == "uint8") {
        experiment<uint8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, disk_L, K, B, M, alpha, consolidate_threads, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, n_theta_estimation_queries, n_search_iter, eager_theta_update);
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
    }

    return 0;
}