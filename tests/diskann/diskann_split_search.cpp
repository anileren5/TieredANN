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
#include <cstring>
#include <fstream>

// DiskANN headers
#include "diskann/utils.h"
#include "diskann/index_factory.h"
#include "diskann/index.h"

namespace po = boost::program_options;

// Metrics structures to match Python format
struct DiskANNMetrics {
    size_t total_queries;
    uint32_t threads;
    double avg_latency_ms;
    double qps;
    double qps_per_thread;
    size_t index_vectors;
    size_t index_max_points;
    double p50;
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
void log_split_metrics(const DiskANNMetrics& metrics, const RecallAllMetrics& recall_all_metrics) {
    // Log combined metrics in single JSON line (matching Python format)
    spdlog::info("{{\"event\": \"split_metrics\", "
              "\"total_queries\": {}, "
              "\"threads\": {}, "
              "\"avg_latency_ms\": {}, "
              "\"qps\": {}, "
              "\"qps_per_thread\": {}, "
              "\"index_vectors\": {}, "
              "\"index_max_points\": {}, "
              "\"tail_latency_ms\": {{\"p50\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}}}, "
              "\"recall_all\": {}, "
              "\"K\": {}, "
              "\"low_recall_queries\": {}, "
              "\"very_low_recall_queries\": {}}}",
              metrics.total_queries,
              metrics.threads, metrics.avg_latency_ms,
              metrics.qps, metrics.qps_per_thread,
              metrics.index_vectors, metrics.index_max_points,
              metrics.p50, metrics.p90, metrics.p95, metrics.p99,
              recall_all_metrics.recall_all, recall_all_metrics.K,
              recall_all_metrics.low_recall_queries, recall_all_metrics.very_low_recall_queries);
}

template <typename T, typename TagT = uint32_t>
DiskANNMetrics diskann_search(
    std::unique_ptr<diskann::AbstractIndex>& index,
    const T* query, size_t query_num, uint32_t query_aligned_dim,
    uint32_t K, uint32_t L, uint32_t search_threads,
    std::vector<uint32_t>& query_result_tags, std::vector<T *>& res
) {
    std::vector<float> query_result_dists(K * query_num);
    std::vector<double> latencies_ms(query_num, 0.0);
    const TagT INVALID_ID = std::numeric_limits<TagT>::max();
    
    // Initialize result tags to invalid values
    std::fill(query_result_tags.begin(), query_result_tags.end(), INVALID_ID);
    std::fill(query_result_dists.begin(), query_result_dists.end(), std::numeric_limits<float>::max());
    
    // Check if index has any vectors
    size_t num_vectors = index->get_number_of_active_vectors();
    if (num_vectors == 0) {
        // Index is empty - return metrics with invalid results
        DiskANNMetrics metrics;
        metrics.total_queries = query_num;
        metrics.threads = search_threads;
        metrics.avg_latency_ms = 0.0;
        metrics.qps = 0.0;
        metrics.qps_per_thread = 0.0;
        metrics.index_vectors = 0;
        metrics.p50 = 0.0;
        metrics.p90 = 0.0;
        metrics.p95 = 0.0;
        metrics.p99 = 0.0;
        return metrics;
    }
    
    auto global_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Initialize this query's results to invalid
        std::fill(query_result_tags.data() + i * K, query_result_tags.data() + (i + 1) * K, INVALID_ID);
        std::fill(query_result_dists.data() + i * K, query_result_dists.data() + (i + 1) * K, std::numeric_limits<float>::max());
        
        // Search in the index
        index->search_with_tags(
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
    
    std::vector<double> sorted_latencies = latencies_ms;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    auto get_percentile = [&](double p) {
        size_t idx = static_cast<size_t>(std::ceil(p * query_num)) - 1;
        if (idx >= query_num) idx = query_num - 1;
        return sorted_latencies[idx];
    };
    double p50 = get_percentile(0.50);
    double p90 = get_percentile(0.90);
    double p95 = get_percentile(0.95);
    double p99 = get_percentile(0.99);
    
    DiskANNMetrics metrics;
    metrics.total_queries = query_num;
    metrics.threads = search_threads;
    metrics.avg_latency_ms = avg_latency_ms;
    metrics.qps = qps;
    metrics.qps_per_thread = qps_per_thread;
    metrics.index_vectors = index->get_number_of_active_vectors();
    metrics.p50 = p50;
    metrics.p90 = p90;
    metrics.p95 = p95;
    metrics.p99 = p99;
    
    return metrics;
}

// Helper function to load a single vector from binary file
template <typename T>
void load_vector_by_index(const std::string& data_path, T* vector, size_t dim, size_t aligned_dim, size_t index) {
    std::ifstream reader(data_path, std::ios::binary);
    if (!reader.is_open()) {
        throw std::runtime_error("Failed to open file: " + data_path);
    }
    
    // Skip metadata (2 * sizeof(uint32_t)) and move to the desired vector
    // Offset = metadata size + (index * vector size)
    size_t offset = 2 * sizeof(uint32_t) + index * dim * sizeof(T);
    reader.seekg(offset, std::ios::beg);
    
    // Read the vector data
    reader.read(reinterpret_cast<char*>(vector), dim * sizeof(T));
    
    // Zero out padding if needed
    if (aligned_dim > dim) {
        std::memset(vector + dim, 0, (aligned_dim - dim) * sizeof(T));
    }
    
    reader.close();
}

// Helper function to insert vectors into diskann index
template <typename T, typename TagT = uint32_t>
void insert_vectors_into_index(
    std::unique_ptr<diskann::AbstractIndex>& index,
    const std::string& data_path,
    size_t dim,
    size_t aligned_dim,
    const std::vector<TagT>& vector_ids
) {
    for (size_t i = 0; i < vector_ids.size(); ++i) {
        TagT vector_id = vector_ids[i];
        
        // Allocate aligned memory for the vector
        T* vector = nullptr;
        diskann::alloc_aligned((void**)&vector, aligned_dim * sizeof(T), 8 * sizeof(T));
        
        // Load vector from file
        load_vector_by_index(data_path, vector, dim, aligned_dim, vector_id);
        
        // Insert into index (tag is vector_id + 1, as diskann uses 1-indexed tags)
        int ret = index->insert_point(vector, 1 + vector_id);
        
        // Free the vector memory
        diskann::aligned_free(vector);
        
        if (ret != 0) {
            std::cerr << "Warning: Failed to insert vector " << vector_id << " (return code: " << ret << ")" << std::endl;
        }
    }
}

// Main experiment logic for split search
template <typename T = float, typename TagT = uint32_t>
void experiment_split(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    uint32_t R,
    uint32_t L,
    uint32_t K,
    float alpha,
    uint32_t build_threads,
    uint32_t search_threads,
    size_t max_points,
    int n_iteration_per_split,
    int n_splits,
    int n_rounds,
    size_t initial_build_points,
    diskann::Metric metric
) {
    // Read metadata from data file
    size_t num_points, dim;
    diskann::get_bin_metadata(data_path, num_points, dim);
    size_t aligned_dim = ROUND_UP(dim, 8);
    
    // Create diskann index
    diskann::IndexWriteParameters write_params = diskann::IndexWriteParametersBuilder(L, aligned_dim)
                                                                    .with_alpha(alpha)
                                                                    .with_num_threads(build_threads)
                                                                    .build();

    diskann::IndexSearchParams search_params = diskann::IndexSearchParams(L, search_threads);

    diskann::IndexConfig index_config = diskann::IndexConfigBuilder()
                                                    .with_metric(metric)
                                                    .with_dimension(dim)
                                                    .with_max_points(max_points)
                                                    .is_dynamic_index(true)
                                                    .with_index_write_params(write_params)
                                                    .with_index_search_params(search_params)
                                                    .with_data_type(diskann_type_to_name<T>())
                                                    .with_tag_type(diskann_type_to_name<TagT>())
                                                    .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                                    .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                                    .is_enable_tags(true)
                                                    .is_filtered(false)
                                                    .with_num_frozen_pts(0)
                                                    .is_concurrent_consolidate(true)
                                                    .build();
    
    diskann::IndexFactory index_factory = diskann::IndexFactory(index_config);
    auto index = index_factory.create_instance();
    index->set_start_points_at_random(static_cast<T>(0));
    
    std::cout << "DiskANN index created successfully!" << std::endl;
    std::cout << "Max points: " << max_points << ", Dimension: " << dim << ", Aligned dim: " << aligned_dim << std::endl;
    
    // Build index with points from the base file
    // Limit to max_points since that's the index capacity
    size_t points_to_build;
    if (initial_build_points > 0) {
        points_to_build = std::min({static_cast<size_t>(initial_build_points), num_points, max_points});
    } else {
        // Default: build up to max_points (or all points if less than max_points)
        points_to_build = std::min(num_points, max_points);
    }
    
    std::cout << "=========================================" << std::endl;
    std::cout << "Building index with " << points_to_build << " / " << num_points << " points from data file..." << std::endl;
    std::cout << "Index capacity (max_points): " << max_points << std::endl;
    if (points_to_build < num_points) {
        std::cout << "Note: Only building " << points_to_build << " points due to index capacity limit." << std::endl;
    }
    std::cout << "This may take a while..." << std::endl;
    
    auto build_start = std::chrono::high_resolution_clock::now();
    
    // Generate tags for the build (sequential 1-indexed tags: 1, 2, 3, ...)
    std::vector<TagT> tags;
    tags.reserve(points_to_build);
    for (size_t i = 0; i < points_to_build; ++i) {
        tags.push_back(static_cast<TagT>(i + 1));
        // Log progress every 10% or every 100k points, whichever is more frequent
        if ((i + 1) % std::max(static_cast<size_t>(1), points_to_build / 10) == 0 || 
            (i + 1) % 100000 == 0) {
            std::cout << "Preparing tags: " << (i + 1) << " / " << points_to_build 
                      << " (" << (100.0 * (i + 1) / points_to_build) << "%)" << std::endl;
        }
    }
    
    std::cout << "Tags prepared. Starting index build..." << std::endl;
    
    // Build the index with all data
    // Cast to concrete Index type (with default LabelT = uint32_t for non-filtered index)
    // to access the filename-based build method
    auto* concrete_index = static_cast<diskann::Index<T, TagT, uint32_t>*>(index.get());
    concrete_index->build(data_path.c_str(), points_to_build, tags);
    
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration<double>(build_end - build_start);
    
    size_t actual_vectors = index->get_number_of_active_vectors();
    double build_time_sec = build_duration.count();
    double build_rate = (build_time_sec > 0) ? (points_to_build / build_time_sec) : 0.0;
    
    std::cout << "=========================================" << std::endl;
    std::cout << "Index build completed!" << std::endl;
    std::cout << "  Points built: " << actual_vectors << " / " << points_to_build << std::endl;
    std::cout << "  Build time: " << build_time_sec << " seconds" << std::endl;
    std::cout << "  Build rate: " << build_rate << " points/second" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Load queries and groundtruth
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);
    
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);
    
    std::vector<T *> res = std::vector<T *>();
    
    std::cout << "Starting split search experiment..." << std::endl;
    std::cout << "  Total queries: " << query_num << std::endl;
    std::cout << "  Index vectors: " << index->get_number_of_active_vectors() << std::endl;
    
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
                    // Search
                    auto metrics = diskann_search<T, TagT>(
                        index,
                        query + start * query_aligned_dim,
                        this_split_size,
                        query_aligned_dim,
                        K,
                        L,
                        search_threads,
                        query_result_tags,
                        res
                    );
                    
                    metrics.index_max_points = max_points;
                    
                    // Calculate recall
                    RecallAllMetrics recall_all = calculate_recall<T, TagT>(
                        K, groundtruth_ids + start * groundtruth_dim, 
                        query_result_tags, this_split_size, groundtruth_dim
                    );
                    
                    // Log metrics
                    log_split_metrics<T, TagT>(metrics, recall_all);
                    
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
                    // Search
                    auto metrics2 = diskann_search<T, TagT>(
                        index,
                        query + start2 * query_aligned_dim,
                        this_split_size2,
                        query_aligned_dim,
                        K,
                        L,
                        search_threads,
                        query_result_tags2,
                        res
                    );
                    
                    metrics2.index_max_points = max_points;
                    
                    // Calculate recall
                    RecallAllMetrics recall_all2 = calculate_recall<T, TagT>(
                        K, groundtruth_ids + start2 * groundtruth_dim, 
                        query_result_tags2, this_split_size2, groundtruth_dim
                    );
                    
                    // Log metrics
                    log_split_metrics<T, TagT>(metrics2, recall_all2);
                    
                    query_result_tags2.clear();
                }
            }
            
            // Second split in the pair again (reverse order)
            if (start2 < end2) {
                size_t this_split_size2 = end2 - start2;
                std::vector<TagT> query_result_tags2(this_split_size2 * K);
                
                for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                    auto metrics2 = diskann_search<T, TagT>(
                        index,
                        query + start2 * query_aligned_dim,
                        this_split_size2,
                        query_aligned_dim,
                        K,
                        L,
                        search_threads,
                        query_result_tags2,
                        res
                    );
                    
                    metrics2.index_max_points = max_points;
                    
                    RecallAllMetrics recall_all2 = calculate_recall<T, TagT>(
                        K, groundtruth_ids + start2 * groundtruth_dim, 
                        query_result_tags2, this_split_size2, groundtruth_dim
                    );
                    
                    log_split_metrics<T, TagT>(metrics2, recall_all2);
                    
                    query_result_tags2.clear();
                }
            }
            
            // First split in the pair again (reverse order)
            if (start < end) {
                size_t this_split_size = end - start;
                std::vector<TagT> query_result_tags(this_split_size * K);
                
                for (int iter = 0; iter < n_iteration_per_split; ++iter) {
                    auto metrics = diskann_search<T, TagT>(
                        index,
                        query + start * query_aligned_dim,
                        this_split_size,
                        query_aligned_dim,
                        K,
                        L,
                        search_threads,
                        query_result_tags,
                        res
                    );
                    
                    metrics.index_max_points = max_points;
                    
                    RecallAllMetrics recall_all = calculate_recall<T, TagT>(
                        K, groundtruth_ids + start * groundtruth_dim, 
                        query_result_tags, this_split_size, groundtruth_dim
                    );
                    
                    log_split_metrics<T, TagT>(metrics, recall_all);
                    
                    query_result_tags.clear();
                }
            }
        }
    }
    
    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path;
    std::string metric_str = "l2";
    uint32_t R, L, K;
    uint32_t build_threads, search_threads;
    float alpha;
    int n_iteration_per_split;
    size_t max_points;
    int n_splits;
    int n_rounds;
    size_t initial_build_points = 0; // 0 means build all points from base file, otherwise build this many points
    
    po::options_description desc;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data (float/int8/uint8)")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R (aligned dimension)")
            ("L", po::value<uint32_t>(&L)->required(), "Value of L (search width)")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K (number of results)")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
            ("max_points", po::value<size_t>(&max_points)->required(), "Max points for index")
            ("n_iteration_per_split", po::value<int>(&n_iteration_per_split)->required(), "Number of search iterations per split")
            ("n_splits", po::value<int>(&n_splits)->required(), "Number of splits for queries")
            ("n_rounds", po::value<int>(&n_rounds)->default_value(1), "Number of rounds to repeat all splits")
            ("initial_build_points", po::value<size_t>(&initial_build_points)->default_value(0), "Number of points to build index with initially (0 = build all points from base file)")
            ("metric", po::value<std::string>(&metric_str)->default_value("l2"), "Distance metric (l2/cosine/inner_product)");
        
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
    
    // Parse metric string to enum
    diskann::Metric metric = diskann::Metric::L2;
    std::string metric_lower = metric_str;
    std::transform(metric_lower.begin(), metric_lower.end(), metric_lower.begin(), ::tolower);
    if (metric_lower == "cosine") {
        metric = diskann::Metric::COSINE;
    } else if (metric_lower == "l2") {
        metric = diskann::Metric::L2;
    } else if (metric_lower == "inner_product" || metric_lower == "ip") {
        metric = diskann::Metric::INNER_PRODUCT;
    } else {
        std::cerr << "Warning: Unknown metric '" << metric_str << "', defaulting to L2" << std::endl;
        metric = diskann::Metric::L2;
    }
    
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_pattern("%v");
    logger->info("{{\n"
        "  \"event\": \"params\",\n"
        "  \"data_type\": \"{}\",\n"
        "  \"data_path\": \"{}\",\n"
        "  \"query_path\": \"{}\",\n"
        "  \"groundtruth_path\": \"{}\",\n"
        "  \"R\": {},\n"
        "  \"L\": {},\n"
        "  \"K\": {},\n"
        "  \"alpha\": {},\n"
        "  \"build_threads\": {},\n"
        "  \"search_threads\": {},\n"
        "  \"max_points\": {},\n"
        "  \"n_iteration_per_split\": {},\n"
        "  \"n_splits\": {},\n"
        "  \"n_rounds\": {},\n"
        "  \"initial_build_points\": {},\n"
        "  \"metric\": \"{}\"\n"
        "}}",
        data_type, data_path, query_path, groundtruth_path, R, L, K, alpha, 
        build_threads, search_threads, max_points, n_iteration_per_split, 
        n_splits, n_rounds, initial_build_points, metric_str);
    
    if (data_type == "float") {
        experiment_split<float>(data_type, data_path, query_path, groundtruth_path, 
                               R, L, K, alpha, build_threads, search_threads, 
                               max_points, n_iteration_per_split, n_splits, n_rounds,
                               initial_build_points, metric);
    } else if (data_type == "int8") {
        experiment_split<int8_t>(data_type, data_path, query_path, groundtruth_path, 
                                R, L, K, alpha, build_threads, search_threads, 
                                max_points, n_iteration_per_split, n_splits, n_rounds,
                                initial_build_points, metric);
    } else if (data_type == "uint8") {
        experiment_split<uint8_t>(data_type, data_path, query_path, groundtruth_path, 
                                 R, L, K, alpha, build_threads, search_threads, 
                                 max_points, n_iteration_per_split, n_splits, n_rounds,
                                 initial_build_points, metric);
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
        return 1;
    }
    
    return 0;
}

