// System headers
#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <set>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// Include for SECTOR_LEN setting
#include "greator/pq_flash_index.h"
// QVCache headers
#include "qvcache/qvcache.h"

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
    spdlog::info("{{\"event\": \"recall\", \"K\": {}, \"recall\": {}, \"type\": \"all\"}}", K, average_recall);
}

template <typename T, typename TagT = uint32_t>
void memory_search(
    diskann::AbstractIndex& memory_index,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t L, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags, std::vector<T *>& res
) {
    std::vector<float> query_result_dists(K * query_num);
    std::vector<double> latencies_ms(query_num, 0.0);
    auto global_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        memory_index.search_with_tags(
            query + i * query_aligned_dim,
            K,
            L,
            query_result_tags.data() + i * K,
            query_result_dists.data() + i * K,
            res
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
    spdlog::info("{{\"event\": \"latency\", \"queries\": {}, \"threads\": {}, \"total_time_ms\": {}, \"avg_latency_ms\": {}, \"qps\": {}, \"qps_per_thread\": {}, \"tail_latency_ms\": {{\"p90\": {}, \"p95\": {}, \"p99\": {}}}}}", query_num, search_threads, total_time_ms, avg_latency_ms, qps, qps_per_thread, p90, p95, p99);
}

template <typename T = float, typename TagT = uint32_t>
void experiment(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    uint32_t R, uint32_t L, uint32_t K,
    float alpha,
    uint32_t consolidate_threads,
    uint32_t build_threads,
    uint32_t search_threads,
    int n_search_iter
) {
    // Read metadata
    size_t num_points, dim, aligned_dim;
    diskann::get_bin_metadata(data_path, num_points, dim);
    aligned_dim = ROUND_UP(dim, 8);

    // Build memory index
    diskann::IndexWriteParameters memory_index_write_params = diskann::IndexWriteParametersBuilder(L, R)
                                                                    .with_alpha(alpha)
                                                                    .with_num_threads(consolidate_threads)
                                                                    .build();

    diskann::IndexSearchParams memory_index_search_params = diskann::IndexSearchParams(L, search_threads);

    diskann::IndexConfig memory_index_config = diskann::IndexConfigBuilder()
                                                .with_metric(diskann::L2)
                                                .with_dimension(dim)
                                                .with_max_points(num_points)
                                                .is_dynamic_index(true)
                                                .with_index_write_params(memory_index_write_params)
                                                .with_index_search_params(memory_index_search_params)
                                                .with_data_type(diskann_type_to_name<T>())
                                                .with_tag_type(diskann_type_to_name<TagT>())
                                                .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                                .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                                .is_enable_tags(true)
                                                .is_filtered(false)
                                                .with_num_frozen_pts(0)
                                                .is_concurrent_consolidate(true)
                                                .build();

    diskann::IndexFactory memory_index_factory = diskann::IndexFactory(memory_index_config);
    std::unique_ptr<diskann::AbstractIndex> memory_index = memory_index_factory.create_instance();
    memory_index->set_start_points_at_random(static_cast<T>(0));

    // Load the vectors
    T *data = nullptr;
    diskann::alloc_aligned((void **)&data, num_points * aligned_dim * sizeof(T), 8 * sizeof(T));
    diskann::load_aligned_bin<T>(data_path, data, num_points, dim, aligned_dim);

    // Load groundtruth ids for the results
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

    // Load queries
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);

    // Build memory index
    #pragma omp parallel for num_threads((int32_t)build_threads) schedule(dynamic)
    for (int64_t j = 0; j < (int64_t)num_points; j++) {
        memory_index->insert_point(&data[j * aligned_dim], 1 + static_cast<TagT>(j));
    }
    std::cout << "Memory index built successfully!" << std::endl;

    // I don't know why it is need but it is somehow necessary to call seach_with_tags.
    std::vector<T *> res = std::vector<T *>();

    // Allocate the space to store result of the queries
    std::vector<TagT> query_result_tags(query_num * K);

    for (int i = 0; i < n_search_iter; i++) {
        memory_search(*memory_index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);
        calculate_recall<T, TagT>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim); 
        query_result_tags.clear();
    }

    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
    if (data) diskann::aligned_free(data);
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path;
    uint32_t R, L, K;
    uint32_t build_threads, consolidate_threads, search_threads;
    float alpha;
    int n_search_iter;
    uint32_t sector_len = 4096; // Default value

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
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("L", po::value<uint32_t>(&L)->required(), "Value of L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("consolidate_threads", po::value<uint32_t>(&consolidate_threads)->required(), "Threads for consolidation")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("n_search_iter", po::value<int>(&n_search_iter)->default_value(100), "Number of search iterations")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes");


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
    
    // Set up spdlog global logger
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v"); // Only print the message (JSON)
    logger->info("{{\n"
        "  \\\"event\\\": \\\"params\\\",\n"
        "  \\\"data_type\\\": \\\"{}\\\",\n"
        "  \\\"data_path\\\": \\\"{}\\\",\n"
        "  \\\"query_path\\\": \\\"{}\\\",\n"
        "  \\\"groundtruth_path\\\": \\\"{}\\\",\n"
        "  \\\"R\\\": {},\n"
        "  \\\"L\\\": {},\n"
        "  \\\"K\\\": {},\n"
        "  \\\"build_threads\\\": {},\n"
        "  \\\"consolidate_threads\\\": {},\n"
        "  \\\"search_threads\\\": {},\n"
        "  \\\"alpha\\\": {},\n"
        "  \\\"n_search_iter\\\": {},\n"
        "  \\\"sector_len\\\": {}\n"
        "}}",
        data_type, data_path, query_path, groundtruth_path, R, L, K, build_threads, consolidate_threads, search_threads, alpha, n_search_iter, sector_len);

    if (data_type == "float") {
        experiment<float>(data_path, query_path, groundtruth_path, R, L, K, alpha, consolidate_threads, build_threads, search_threads, n_search_iter);
    } else if (data_type == "int8") {
        experiment<int8_t>(data_path, query_path, groundtruth_path, R, L, K, alpha, consolidate_threads, build_threads, search_threads, n_search_iter);
    } else if (data_type == "uint8") {
        experiment<uint8_t>(data_path, query_path, groundtruth_path, R, L, K, alpha, consolidate_threads, build_threads, search_threads, n_search_iter);
    } else {
        spdlog::error("{{\"event\": \"error\", \"message\": \"Unsupported data type: {}\"}}", data_type);
    }

    return 0;
}