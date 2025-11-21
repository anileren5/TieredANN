// System headers
#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <cmath>

//  Include Brute-force backend implementation
#include "bruteforce_backend.h"

// Include TieredIndex
#include "tieredann/tiered_index.h"

// Utils for benchmarking
#include "tieredann/utils.h"

namespace po = boost::program_options;

template <typename T, typename TagT = uint32_t>
std::vector<bool> hybrid_search(
    tieredann::TieredIndex<T>& tiered_index,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t L, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags, std::vector<T *>& res,
    uint32_t beamwidth, const std::string& data_path
) {
    std::vector<float> query_result_dists(K * query_num);
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
            query_result_tags.data() + i * K,
            res,
            query_result_dists.data() + i * K,
            nullptr
        );
        hit_results[i] = hit;
        if (hit) hit_count.fetch_add(1, std::memory_order_relaxed);
        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    double total_hit_latency_ms = 0.0;
    size_t actual_hit_count = 0;
    for (size_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            total_hit_latency_ms += latencies_ms[i];
            actual_hit_count++;
        }
    }
    double avg_hit_latency_ms = (actual_hit_count > 0) ? total_hit_latency_ms / actual_hit_count : 0.0;
    double final_ratio = static_cast<double>(hit_count.load(std::memory_order_relaxed)) / query_num;
    spdlog::info("{{\"event\": \"hit_ratio\", \"hit_ratio\": {}, \"hits\": {}, \"total\": {}}}", final_ratio, hit_count, query_num);
    auto global_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(global_end - global_start).count();
    double total_time_sec = total_time_ms / 1000.0;
    double avg_latency_ms = 0.0;
    for (double latency : latencies_ms) avg_latency_ms += latency;
    avg_latency_ms /= query_num;
    double qps = static_cast<double>(query_num) / total_time_sec;
    double qps_per_thread = qps / static_cast<double>(search_threads);
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
    // Build mini index vector counts string
    std::string mini_index_counts = "";
    size_t num_mini_indexes = tiered_index.get_number_of_mini_indexes();
    for (size_t i = 0; i < num_mini_indexes; ++i) {
        if (i > 0) mini_index_counts += ", ";
        mini_index_counts += "\"index_" + std::to_string(i) + "_vectors\": " + std::to_string(tiered_index.get_index_vector_count(i));
    }
    
    spdlog::info("{{\"event\": \"latency\", "
              "\"threads\": {}, "
              "\"avg_latency_ms\": {}, "
              "\"avg_hit_latency_ms\": {}, "
              "\"qps\": {}, "
              "\"qps_per_thread\": {}, "
              "\"memory_active_vectors\": {}, "
              "\"memory_max_points\": {}, "
              "\"pca_active_regions\": {}, "
              "{}, "
              "\"tail_latency_ms\": {{\"p90\": {}, \"p95\": {}, \"p99\": {}}}}}",
              search_threads, avg_latency_ms, avg_hit_latency_ms, qps, qps_per_thread, tiered_index.get_number_of_vectors_in_memory_index(), tiered_index.get_number_of_max_points_in_memory_index(), tiered_index.get_number_of_active_pca_regions(), mini_index_counts, p90, p95, p99);
    return hit_results;
}

template <typename T = float, typename TagT = uint32_t>
void experiment_split(
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& pca_prefix,
    uint32_t R, uint32_t memory_L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha,
    uint32_t build_threads,
    uint32_t search_threads,
    uint32_t beamwidth, 
    int use_reconstructed_vectors,
    double p,
    double deviation_factor,
    int n_iteration_per_split,
    size_t memory_index_max_points,
    bool use_regional_theta,
    uint32_t pca_dim,
    uint32_t buckets_per_dim,
    int n_splits,
    int n_rounds,
    uint32_t n_async_insert_threads,
    bool lazy_theta_updates,
    size_t number_of_mini_indexes,
    bool search_mini_indexes_in_parallel,
    size_t max_search_threads,
    const std::string& search_strategy,
    std::unique_ptr<tieredann::BackendInterface<T, TagT>> backend
) {
   tieredann::TieredIndex<T> tiered_index(
       data_path, pca_prefix,
       R, memory_L, B, M, alpha, 
       build_threads, search_threads,
       (bool)use_reconstructed_vectors,
       p, deviation_factor,
       memory_index_max_points,
       beamwidth,
       use_regional_theta,
       pca_dim,
       buckets_per_dim,
       n_async_insert_threads,
       lazy_theta_updates,
       number_of_mini_indexes,
       search_mini_indexes_in_parallel,
       max_search_threads,
       std::move(backend)
    );

    // Set the search strategy
    if (search_strategy == "SEQUENTIAL_LRU_STOP_FIRST_HIT") {
        tiered_index.set_search_strategy(tieredann::TieredIndex<T>::SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT);
    } else if (search_strategy == "SEQUENTIAL_LRU_ADAPTIVE") {
        tiered_index.set_search_strategy(tieredann::TieredIndex<T>::SearchStrategy::SEQUENTIAL_LRU_ADAPTIVE);
        // Enable adaptive behavior and set parameters
        tiered_index.enable_adaptive_strategy(true);
        tiered_index.set_hit_ratio_window_size(100);  // Monitor last 100 queries
        tiered_index.set_hit_ratio_threshold(0.90);   // Switch to SEQUENTIAL_ALL when hit ratio < 90%
    } else if (search_strategy == "SEQUENTIAL_ALL") {
        tiered_index.set_search_strategy(tieredann::TieredIndex<T>::SearchStrategy::SEQUENTIAL_ALL);
    } else if (search_strategy == "PARALLEL") {
        tiered_index.set_search_strategy(tieredann::TieredIndex<T>::SearchStrategy::PARALLEL);
    } else {
        std::cerr << "Unknown search strategy: " << search_strategy << std::endl;
        std::cerr << "Available strategies: SEQUENTIAL_LRU_STOP_FIRST_HIT, SEQUENTIAL_LRU_ADAPTIVE, SEQUENTIAL_ALL, PARALLEL" << std::endl;
        exit(1);
    }

    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    load_ground_truth_data(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    load_aligned_binary_data(query_path, query, query_num, query_dim, query_aligned_dim);

    std::vector<T *> res = std::vector<T *>();
    // Split queries
    size_t split_size = (query_num + n_splits - 1) / n_splits;
    for (int round = 0; round < n_rounds; ++round) {
        for (int split = 0; split < n_splits; split += 2) {
            // Process splits in pattern: 1, 2, 2, 1, 3, 4, 4, 3, 5, 6, 6, 5, etc.
            
            // First split in the pair
            size_t start = split * split_size;
            size_t end = std::min(start + split_size, query_num);
            if (start < end) {
                size_t this_split_size = end - start;
                std::vector<TagT> query_result_tags(this_split_size * K);
                for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                    std::vector<bool> hit_results = hybrid_search(
                        tiered_index,
                        query + start * query_aligned_dim,
                        this_split_size,
                        query_aligned_dim,
                        K,
                        memory_L,
                        search_threads,
                        query_result_tags,
                        res,
                        beamwidth,
                        data_path
                    );
                    calculate_recall<T, TagT>(K, groundtruth_ids + start * groundtruth_dim, query_result_tags, this_split_size, groundtruth_dim);
                    calculate_hit_recall<T, TagT>(K, groundtruth_ids + start * groundtruth_dim, query_result_tags, hit_results, this_split_size, groundtruth_dim);
                    query_result_tags.clear();
                }
            }
            
            // Second split in the pair
            size_t start2 = (split + 1) * split_size;
            size_t end2 = std::min(start2 + split_size, query_num);
            if (start2 < end2) {
                size_t this_split_size2 = end2 - start2;
                std::vector<TagT> query_result_tags2(this_split_size2 * K);
                for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                    std::vector<bool> hit_results2 = hybrid_search(
                        tiered_index,
                        query + start2 * query_aligned_dim,
                        this_split_size2,
                        query_aligned_dim,
                        K,
                        memory_L,
                        search_threads,
                        query_result_tags2,
                        res,
                        beamwidth,
                        data_path
                    );
                    calculate_recall<T, TagT>(K, groundtruth_ids + start2 * groundtruth_dim, query_result_tags2, this_split_size2, groundtruth_dim);
                    calculate_hit_recall<T, TagT>(K, groundtruth_ids + start2 * groundtruth_dim, query_result_tags2, hit_results2, this_split_size2, groundtruth_dim);
                    query_result_tags2.clear();
                }
            }
            
            // Second split in the pair again (reverse order)
            if (start2 < end2) {
                size_t this_split_size2 = end2 - start2;
                std::vector<TagT> query_result_tags2(this_split_size2 * K);
                for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                    std::vector<bool> hit_results2 = hybrid_search(
                        tiered_index,
                        query + start2 * query_aligned_dim,
                        this_split_size2,
                        query_aligned_dim,
                        K,
                        memory_L,
                        search_threads,
                        query_result_tags2,
                        res,
                        beamwidth,
                        data_path
                    );
                    calculate_recall<T, TagT>(K, groundtruth_ids + start2 * groundtruth_dim, query_result_tags2, this_split_size2, groundtruth_dim);
                    calculate_hit_recall<T, TagT>(K, groundtruth_ids + start2 * groundtruth_dim, query_result_tags2, hit_results2, this_split_size2, groundtruth_dim);
                    query_result_tags2.clear();
                }
            }
            
            // First split in the pair again (reverse order)
            if (start < end) {
                size_t this_split_size = end - start;
                std::vector<TagT> query_result_tags(this_split_size * K);
                for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                    std::vector<bool> hit_results = hybrid_search(
                        tiered_index,
                        query + start * query_aligned_dim,
                        this_split_size,
                        query_aligned_dim,
                        K,
                        memory_L,
                        search_threads,
                        query_result_tags,
                        res,
                        beamwidth,
                        data_path
                    );
                    calculate_recall<T, TagT>(K, groundtruth_ids + start * groundtruth_dim, query_result_tags, this_split_size, groundtruth_dim);
                    calculate_hit_recall<T, TagT>(K, groundtruth_ids + start * groundtruth_dim, query_result_tags, hit_results, this_split_size, groundtruth_dim);
                    query_result_tags.clear();
                }
            }
        }
    }
    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, pca_prefix;
    uint32_t R, memory_L, K, B, M;
    uint32_t build_threads, search_threads, beamwidth;
    float alpha;
    int use_reconstructed_vectors;
    double p, deviation_factor;
    int n_iteration_per_split;
    bool use_regional_theta = true;
    uint32_t pca_dim, buckets_per_dim;
    size_t memory_index_max_points;
    int n_splits;
    int n_rounds;
    uint32_t n_async_insert_threads = 4;
    bool lazy_theta_updates = true;
    size_t number_of_mini_indexes = 2;
    bool search_mini_indexes_in_parallel = false;
    size_t max_search_threads = 32;
    std::string search_strategy = "SEQUENTIAL_LRU_STOP_FIRST_HIT";
    po::options_description desc;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("pca_prefix", po::value<std::string>(&pca_prefix)->required(), "Prefix for PCA files")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("memory_L", po::value<uint32_t>(&memory_L)->required(), "Value of memory L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("use_reconstructed_vectors", po::value<int>(&use_reconstructed_vectors)->default_value(0), "Use reconstructed vectors for insertion to memory index")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("p", po::value<double>(&p)->default_value(0.75), "Value of p")
            ("deviation_factor", po::value<double>(&deviation_factor)->default_value(0.05), "Value of deviation factor")
            ("n_iteration_per_split", po::value<int>(&n_iteration_per_split)->required(), "Number of search iterations per split")
            ("use_regional_theta", po::value<bool>(&use_regional_theta)->default_value(true), "Use regional theta (true) or global theta (false)")
            ("pca_dim", po::value<uint32_t>(&pca_dim)->required(), "Value of PCA dimension")
            ("buckets_per_dim", po::value<uint32_t>(&buckets_per_dim)->required(), "Value of buckets per dimension")
            ("memory_index_max_points", po::value<size_t>(&memory_index_max_points)->required(), "Max points for memory index")
            ("n_splits", po::value<int>(&n_splits)->required(), "Number of splits for queries")
            ("n_rounds", po::value<int>(&n_rounds)->default_value(1), "Number of rounds to repeat all splits")
            ("n_async_insert_threads", po::value<uint32_t>(&n_async_insert_threads)->default_value(4), "Number of async insert threads")
            ("lazy_theta_updates", po::value<bool>(&lazy_theta_updates)->default_value(true), "Enable lazy theta updates (true) or immediate updates (false)")
            ("number_of_mini_indexes", po::value<size_t>(&number_of_mini_indexes)->default_value(2), "Number of mini indexes for shadow cycling")
            ("search_mini_indexes_in_parallel", po::value<bool>(&search_mini_indexes_in_parallel)->default_value(false), "Search mini indexes in parallel (true) or sequential (false)")
            ("max_search_threads", po::value<size_t>(&max_search_threads)->default_value(32), "Maximum threads for parallel search")
            ("search_strategy", po::value<std::string>(&search_strategy)->default_value("SEQUENTIAL_LRU_STOP_FIRST_HIT"), "Search strategy: SEQUENTIAL_LRU_STOP_FIRST_HIT, SEQUENTIAL_LRU_ADAPTIVE, SEQUENTIAL_ALL, PARALLEL");
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
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");
    logger->info("{{\n"
        "  \"event\": \"params\",\n"
        "  \"data_type\": \"{}\",\n"
        "  \"data_path\": \"{}\",\n"
        "  \"query_path\": \"{}\",\n"
        "  \"groundtruth_path\": \"{}\",\n"
        "  \"pca_prefix\": \"{}\",\n"
        "  \"R\": {},\n"
        "  \"memory_L\": {},\n"
        "  \"K\": {},\n"
        "  \"B\": {},\n"
        "  \"M\": {},\n"
        "  \"build_threads\": {},\n"
        "  \"search_threads\": {},\n"
        "  \"alpha\": {},\n"
        "  \"use_reconstructed_vectors\": {},\n"
        "  \"beamwidth\": {},\n"
        "  \"p\": {},\n"
        "  \"deviation_factor\": {},\n"
        "  \"n_iteration_per_split\": {},\n"
        "  \"use_regional_theta\": {},\n"
        "  \"pca_dim\": {},\n"
        "  \"buckets_per_dim\": {},\n"
        "  \"memory_index_max_points\": {},\n"
        "  \"n_splits\": {},\n"
        "  \"n_rounds\": {},\n"
        "  \"n_async_insert_threads\": {},\n"
        "  \"lazy_theta_updates\": {},\n"
        "  \"number_of_mini_indexes\": {},\n"
        "  \"search_mini_indexes_in_parallel\": {},\n"
        "  \"max_search_threads\": {},\n"
        "  \"search_strategy\": \"{}\"\n"
        "}}",
        data_type, data_path, query_path, groundtruth_path, pca_prefix, R, memory_L, K, B, M, build_threads, search_threads, alpha, use_reconstructed_vectors, beamwidth, p, deviation_factor, n_iteration_per_split, use_regional_theta, pca_dim, buckets_per_dim, memory_index_max_points, n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_mini_indexes_in_parallel, max_search_threads, search_strategy);
    if (data_type == "float") {
        std::unique_ptr<tieredann::BackendInterface<float, uint32_t>> backend = std::make_unique<tieredann::BruteforceBackend<float>>(data_path);
        experiment_split<float>(
            data_path, query_path, groundtruth_path, pca_prefix, R, memory_L, K, B, M, alpha, 
            build_threads, search_threads, beamwidth, use_reconstructed_vectors, p, deviation_factor, 
            n_iteration_per_split, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, 
            n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, 
            search_mini_indexes_in_parallel, max_search_threads, search_strategy, std::move(backend));
    } else if (data_type == "int8") {
        std::unique_ptr<tieredann::BackendInterface<int8_t, uint32_t>> backend = std::make_unique<tieredann::BruteforceBackend<int8_t>>(data_path);
        experiment_split<int8_t>(
            data_path, query_path, groundtruth_path, pca_prefix, R, memory_L, K, B, M, alpha, 
            build_threads, search_threads, beamwidth, use_reconstructed_vectors, p, deviation_factor, 
            n_iteration_per_split, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, 
            n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, 
            search_mini_indexes_in_parallel, max_search_threads, search_strategy, std::move(backend));
    } else if (data_type == "uint8") {
        std::unique_ptr<tieredann::BackendInterface<uint8_t, uint32_t>> backend = std::make_unique<tieredann::BruteforceBackend<uint8_t>>(data_path);
        experiment_split<uint8_t>(
            data_path, query_path, groundtruth_path, pca_prefix, R, memory_L, K, B, M, alpha, 
            build_threads, search_threads, beamwidth, use_reconstructed_vectors, p, deviation_factor, 
            n_iteration_per_split, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, 
            n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, 
            search_mini_indexes_in_parallel, max_search_threads, search_strategy, std::move(backend));
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
    }
    return 0;
}