// System headers
#include <cstddef>
#include <omp.h>
#include <boost/program_options.hpp>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>
#include <cmath>
#include <map>
#include <limits>
#include <utility>

// Backend header
#include "greator_backend.h"

// QVCache header
#include "qvcache/qvcache.h"

namespace po = boost::program_options;

// Metrics structures to match Python format
struct HybridMetrics {
    double hit_ratio;
    size_t hits;
    size_t total_queries;
    uint32_t threads;
    double avg_latency_ms;
    double avg_hit_latency_ms;
    double qps;
    double qps_per_thread;
    size_t memory_active_vectors;
    size_t memory_max_points;
    size_t pca_active_regions;
    std::map<size_t, size_t> index_vectors;  // index_id -> vector_count
    double p90;
    double p95;
    double p99;
};

struct RecallAllMetrics {
    double recall_all;
    uint32_t K;
    size_t low_recall_queries;
    size_t very_low_recall_queries;
};

struct RecallHitMetrics {
    double recall_cache_hits;  // Use -1.0 to represent null
    size_t cache_hit_count;
};

template <typename T, typename TagT = uint32_t>
RecallAllMetrics calculate_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    std::vector<double> recall_by_query;
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();
    
    for (int32_t i = 0; i < query_num; i++) {
        std::set<uint32_t> groundtruth_closest_neighbors;
        std::set<uint32_t> calculated_closest_neighbors;
        for (int32_t j = 0; j < K; j++) {
            groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
        }
        // Filter out invalid IDs (padded results)
        for (int32_t j = 0; j < K; j++) {
            TagT tag = *(query_result_tags.data() + i * K + j);
            if (tag != INVALID_ID) {
                calculated_closest_neighbors.insert(tag);
            }
        }
        uint32_t matching_neighbors = 0;
        for (uint32_t x : calculated_closest_neighbors) {
            if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
        }
        double recall = matching_neighbors / (double)K;
        recall_by_query.push_back(recall);
        total_recall += recall;
    }
    double average_recall = total_recall / (query_num);
    
    // Count queries with low recall
    size_t low_recall_count = 0;
    size_t very_low_recall_count = 0;
    for (double r : recall_by_query) {
        if (r < 0.5) low_recall_count++;
        if (r < 0.1) very_low_recall_count++;
    }
    
    RecallAllMetrics metrics;
    metrics.recall_all = average_recall;
    metrics.K = K;
    metrics.low_recall_queries = low_recall_count;
    metrics.very_low_recall_queries = very_low_recall_count;
    
    return metrics;
}

template <typename T, typename TagT = uint32_t>
RecallHitMetrics calculate_hit_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, 
                         const std::vector<bool>& hit_results, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    size_t hit_count = 0;
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();
    
    for (int32_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            std::set<uint32_t> groundtruth_closest_neighbors;
            std::set<uint32_t> calculated_closest_neighbors;
            for (int32_t j = 0; j < K; j++) {
                groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
            }
            // Filter out invalid IDs (padded results)
            for (int32_t j = 0; j < K; j++) {
                TagT tag = *(query_result_tags.data() + i * K + j);
                if (tag != INVALID_ID) {
                    calculated_closest_neighbors.insert(tag);
                }
            }
            uint32_t matching_neighbors = 0;
            for (uint32_t x : calculated_closest_neighbors) {
                if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
            }
            double recall = matching_neighbors / (double)K;
            total_recall += recall;
            hit_count++;
        }
    }
    
    RecallHitMetrics metrics;
    if (hit_count > 0) {
        metrics.recall_cache_hits = total_recall / hit_count;
    } else {
        metrics.recall_cache_hits = -1.0;  // Use -1.0 to represent null
    }
    metrics.cache_hit_count = hit_count;
    
    return metrics;
}

template <typename T, typename TagT = uint32_t>
void log_split_metrics(const HybridMetrics& hybrid_metrics, const RecallAllMetrics& recall_all_metrics, const RecallHitMetrics& recall_hit_metrics) {
    // Build mini index vector counts string
    std::string mini_index_counts = "";
    for (const auto& [idx, count] : hybrid_metrics.index_vectors) {
        if (!mini_index_counts.empty()) mini_index_counts += ", ";
        mini_index_counts += "\"index_" + std::to_string(idx) + "_vectors\": " + std::to_string(count);
    }
    
    // Log combined metrics in single JSON line (matching Python format)
    if (recall_hit_metrics.recall_cache_hits < 0) {
        // null case
        spdlog::info("{{\"event\": \"split_metrics\", "
                  "\"hit_ratio\": {}, "
                  "\"hits\": {}, "
                  "\"total_queries\": {}, "
                  "\"threads\": {}, "
                  "\"avg_latency_ms\": {}, "
                  "\"avg_hit_latency_ms\": {}, "
                  "\"qps\": {}, "
                  "\"qps_per_thread\": {}, "
                  "\"memory_active_vectors\": {}, "
                  "\"memory_max_points\": {}, "
                  "\"pca_active_regions\": {}, "
                  "{}, "
                  "\"tail_latency_ms\": {{\"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, "
                  "\"K\": {}, "
                  "\"low_recall_queries\": {}, "
                  "\"very_low_recall_queries\": {}, "
                  "\"recall_cache_hits\": null, "
                  "\"cache_hit_count\": {}}}",
                  hybrid_metrics.hit_ratio, hybrid_metrics.hits, hybrid_metrics.total_queries,
                  hybrid_metrics.threads, hybrid_metrics.avg_latency_ms, hybrid_metrics.avg_hit_latency_ms,
                  hybrid_metrics.qps, hybrid_metrics.qps_per_thread,
                  hybrid_metrics.memory_active_vectors, hybrid_metrics.memory_max_points,
                  hybrid_metrics.pca_active_regions, mini_index_counts,
                  hybrid_metrics.p90, hybrid_metrics.p95, hybrid_metrics.p99,
                  recall_all_metrics.recall_all, recall_all_metrics.K,
                  recall_all_metrics.low_recall_queries, recall_all_metrics.very_low_recall_queries,
                  recall_hit_metrics.cache_hit_count);
    } else {
        spdlog::info("{{\"event\": \"split_metrics\", "
                  "\"hit_ratio\": {}, "
                  "\"hits\": {}, "
                  "\"total_queries\": {}, "
                  "\"threads\": {}, "
                  "\"avg_latency_ms\": {}, "
                  "\"avg_hit_latency_ms\": {}, "
                  "\"qps\": {}, "
                  "\"qps_per_thread\": {}, "
                  "\"memory_active_vectors\": {}, "
                  "\"memory_max_points\": {}, "
                  "\"pca_active_regions\": {}, "
                  "{}, "
                  "\"tail_latency_ms\": {{\"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
                  "\"recall_all\": {}, "
                  "\"K\": {}, "
                  "\"low_recall_queries\": {}, "
                  "\"very_low_recall_queries\": {}, "
                  "\"recall_cache_hits\": {}, "
                  "\"cache_hit_count\": {}}}",
                  hybrid_metrics.hit_ratio, hybrid_metrics.hits, hybrid_metrics.total_queries,
                  hybrid_metrics.threads, hybrid_metrics.avg_latency_ms, hybrid_metrics.avg_hit_latency_ms,
                  hybrid_metrics.qps, hybrid_metrics.qps_per_thread,
                  hybrid_metrics.memory_active_vectors, hybrid_metrics.memory_max_points,
                  hybrid_metrics.pca_active_regions, mini_index_counts,
                  hybrid_metrics.p90, hybrid_metrics.p95, hybrid_metrics.p99,
                  recall_all_metrics.recall_all, recall_all_metrics.K,
                  recall_all_metrics.low_recall_queries, recall_all_metrics.very_low_recall_queries,
                  recall_hit_metrics.recall_cache_hits, recall_hit_metrics.cache_hit_count);
    }
}

template <typename T, typename TagT = uint32_t>
std::pair<std::vector<bool>, HybridMetrics> hybrid_search(
    qvcache::QVCache<T>& qvcache,
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
        bool hit = qvcache.search(
            query + i * query_aligned_dim,
            K,
            query_result_tags.data() + i * K,
            res,
            query_result_dists.data() + i * K,
            stats + i
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
    
    // Build index vector counts map
    std::map<size_t, size_t> index_vectors;
    size_t num_mini_indexes = qvcache.get_number_of_mini_indexes();
    for (size_t i = 0; i < num_mini_indexes; ++i) {
        index_vectors[i] = qvcache.get_index_vector_count(i);
    }
    
    HybridMetrics metrics;
    metrics.hit_ratio = final_ratio;
    metrics.hits = hit_count.load(std::memory_order_relaxed);
    metrics.total_queries = query_num;
    metrics.threads = search_threads;
    metrics.avg_latency_ms = avg_latency_ms;
    metrics.avg_hit_latency_ms = avg_hit_latency_ms;
    metrics.qps = qps;
    metrics.qps_per_thread = qps_per_thread;
    metrics.memory_active_vectors = qvcache.get_number_of_vectors_in_memory_index();
    metrics.memory_max_points = qvcache.get_number_of_max_points_in_memory_index();
    metrics.pca_active_regions = qvcache.get_number_of_active_pca_regions();
    metrics.index_vectors = index_vectors;
    metrics.p90 = p90;
    metrics.p95 = p95;
    metrics.p99 = p99;
    
    delete[] stats;
    return std::make_pair(hit_results, metrics);
}

// Main experiment logic for noisy queries

template <typename T = float, typename TagT = uint32_t>
void experiment_split_noisy(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t memory_L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha,
    uint32_t build_threads,
    uint32_t search_threads,
    int disk_index_already_built,
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
    std::unique_ptr<qvcache::BackendInterface<T, TagT>> greator_backend
) {
   qvcache::QVCache<T> qvcache(
       data_path, disk_index_prefix,
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
       std::move(greator_backend)
    );

    // Set the search strategy
    if (search_strategy == "SEQUENTIAL_LRU_STOP_FIRST_HIT") {
        qvcache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL_LRU_STOP_FIRST_HIT);
    } else if (search_strategy == "SEQUENTIAL_LRU_ADAPTIVE") {
        qvcache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL_LRU_ADAPTIVE);
        // Enable adaptive behavior and set parameters
        qvcache.enable_adaptive_strategy(true);
        qvcache.set_hit_ratio_window_size(100);  // Monitor last 100 queries
        qvcache.set_hit_ratio_threshold(0.90);   // Switch to SEQUENTIAL_ALL when hit ratio < 90%
    } else if (search_strategy == "SEQUENTIAL_ALL") {
        qvcache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::SEQUENTIAL_ALL);
    } else if (search_strategy == "PARALLEL") {
        qvcache.set_search_strategy(qvcache::QVCache<T>::SearchStrategy::PARALLEL);
    } else {
        std::cerr << "Unknown search strategy: " << search_strategy << std::endl;
        std::cerr << "Available strategies: SEQUENTIAL_LRU_STOP_FIRST_HIT, SEQUENTIAL_LRU_ADAPTIVE, SEQUENTIAL_ALL, PARALLEL" << std::endl;
        exit(1);
    }

    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);
    size_t total_query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, total_query_num, query_dim, query_aligned_dim);
    std::vector<T *> res = std::vector<T *>();
    // Calculate number of queries per round
    if (total_query_num % n_rounds != 0) {
        std::cerr << "Total number of queries is not divisible by n_rounds!" << std::endl;
        exit(1);
    }
    size_t query_num_per_round = total_query_num / n_rounds;
    if (n_groundtruth % n_rounds != 0) {
        std::cerr << "Total number of groundtruth records is not divisible by n_rounds!" << std::endl;
        exit(1);
    }
    size_t groundtruth_per_round = n_groundtruth / n_rounds;
    size_t split_size = (query_num_per_round + n_splits - 1) / n_splits;
    for (int round = 0; round < n_rounds; ++round) {
        size_t round_query_offset = round * query_num_per_round;
        size_t round_gt_offset = round * groundtruth_per_round;
        spdlog::info("{{\"event\": \"round_start\", \"round\": {}}}", round);
        for (int split = 0; split < n_splits; ++split) {
            size_t start = split * split_size;
            size_t end = std::min(start + split_size, query_num_per_round);
            if (start >= end) break;
            size_t this_split_size = end - start;
            std::vector<TagT> query_result_tags(this_split_size * K);
            for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                auto [hit_results, hybrid_metrics] = hybrid_search(
                    qvcache,
                    query + (round_query_offset + start) * query_aligned_dim,
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
                RecallAllMetrics recall_all = calculate_recall<T, TagT>(K, groundtruth_ids + (round_gt_offset + start) * groundtruth_dim, query_result_tags, this_split_size, groundtruth_dim);
                RecallHitMetrics recall_hits = calculate_hit_recall<T, TagT>(K, groundtruth_ids + (round_gt_offset + start) * groundtruth_dim, query_result_tags, hit_results, this_split_size, groundtruth_dim);
                log_split_metrics<T, TagT>(hybrid_metrics, recall_all, recall_hits);
                query_result_tags.clear();
            }
        }
    }
    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, disk_index_prefix;
    uint32_t R, memory_L, disk_L, K, B, M;
    uint32_t build_threads, search_threads, beamwidth;
    float alpha;
    int single_file_index, disk_index_already_built, use_reconstructed_vectors;
    double hit_rate;
    double p, deviation_factor;
    int n_iteration_per_split;
    uint32_t sector_len = 4096;
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
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query (base + noisy)")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth (base + noisy)")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix to index")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("memory_L", po::value<uint32_t>(&memory_L)->required(), "Value of memory L")
            ("disk_L", po::value<uint32_t>(&disk_L)->required(), "Value of disk L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("use_reconstructed_vectors", po::value<int>(&use_reconstructed_vectors)->default_value(true), "Use reconstructed vectors for insertion to memory index")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("p", po::value<double>(&p)->default_value(0.75), "Value of p")
            ("deviation_factor", po::value<double>(&deviation_factor)->default_value(0.05), "Value of deviation factor")
            ("n_iteration_per_split", po::value<int>(&n_iteration_per_split)->required(), "Number of search iterations per split")
            ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes")
            ("use_regional_theta", po::value<bool>(&use_regional_theta)->default_value(true), "Use regional theta (true) or global theta (false)")
            ("pca_dim", po::value<uint32_t>(&pca_dim)->required(), "Value of PCA dimension")
            ("buckets_per_dim", po::value<uint32_t>(&buckets_per_dim)->required(), "Value of buckets per dimension")
            ("memory_index_max_points", po::value<size_t>(&memory_index_max_points)->required(), "Max points for memory index")
            ("n_splits", po::value<int>(&n_splits)->required(), "Number of splits for queries")
            ("n_rounds", po::value<int>(&n_rounds)->required(), "Number of rounds (base + noisy sets)")
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
    set_sector_len(sector_len);
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");
    logger->info("{{\n"
        "  \"event\": \"params\",\n"
        "  \"data_type\": \"{}\",\n"
        "  \"data_path\": \"{}\",\n"
        "  \"query_path\": \"{}\",\n"
        "  \"groundtruth_path\": \"{}\",\n"
        "  \"disk_index_prefix\": \"{}\",\n"
        "  \"R\": {},\n"
        "  \"memory_L\": {},\n"
        "  \"disk_L\": {},\n"
        "  \"K\": {},\n"
        "  \"B\": {},\n"
        "  \"M\": {},\n"
        "  \"build_threads\": {},\n"
        "  \"search_threads\": {},\n"
        "  \"alpha\": {},\n"
        "  \"use_reconstructed_vectors\": {},\n"
        "  \"disk_index_already_built\": {},\n"
        "  \"beamwidth\": {},\n"
        "  \"p\": {},\n"
        "  \"deviation_factor\": {},\n"
        "  \"n_iteration_per_split\": {},\n"
        "  \"sector_len\": {},\n"
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
        data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, build_threads, search_threads, alpha, use_reconstructed_vectors, disk_index_already_built, beamwidth, p, deviation_factor, n_iteration_per_split, sector_len, use_regional_theta, pca_dim, buckets_per_dim, memory_index_max_points, n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_mini_indexes_in_parallel, max_search_threads, search_strategy);
    if (data_type == "float") {
        std::unique_ptr<qvcache::BackendInterface<float, uint32_t>> greator_backend = std::make_unique<qvcache::GreatorBackend<float>>(
            data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth);
        experiment_split_noisy<float>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, alpha, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, n_iteration_per_split, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_mini_indexes_in_parallel, max_search_threads, search_strategy, std::move(greator_backend));
    } else if (data_type == "int8") {
        std::unique_ptr<qvcache::BackendInterface<int8_t, uint32_t>> greator_backend = std::make_unique<qvcache::GreatorBackend<int8_t>>(
            data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth);
        experiment_split_noisy<int8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, alpha, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, n_iteration_per_split, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_mini_indexes_in_parallel, max_search_threads, search_strategy, std::move(greator_backend));
    } else if (data_type == "uint8") {
        std::unique_ptr<qvcache::BackendInterface<uint8_t, uint32_t>> greator_backend = std::make_unique<qvcache::GreatorBackend<uint8_t>>(
            data_path, disk_index_prefix, R, disk_L, B, M, build_threads, disk_index_already_built, beamwidth);
        experiment_split_noisy<uint8_t>(data_type, data_path, query_path, groundtruth_path, disk_index_prefix, R, memory_L, K, B, M, alpha, build_threads, search_threads, disk_index_already_built, beamwidth, use_reconstructed_vectors, p, deviation_factor, n_iteration_per_split, memory_index_max_points, use_regional_theta, pca_dim, buckets_per_dim, n_splits, n_rounds, n_async_insert_threads, lazy_theta_updates, number_of_mini_indexes, search_mini_indexes_in_parallel, max_search_threads, search_strategy, std::move(greator_backend));
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
    }
    return 0;
} 