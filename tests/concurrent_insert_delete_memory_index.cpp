// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <boost/program_options.hpp>
#include <future>

#include "diskann/index.h"
#include "diskann/utils.h"
#include "diskann/filter_utils.h"
#include "diskann/program_options_utils.hpp"
#include "diskann/index_factory.h"
#include "diskann/memory_mapper.h"

namespace po = boost::program_options;

void print_experiment_settings(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& chunks_groundtruth_path,
    size_t chunk_size,
    uint32_t R, uint32_t L, uint32_t K,
    float alpha,
    uint32_t insert_threads, uint32_t consolidate_threads,
    uint32_t build_threads, uint32_t search_threads)
{
    const int width = 70;
    std::string line(width, '=');

    std::cout << "\n\n" << line << '\n';
    std::cout << " Experiment Settings\n";
    std::cout << line << '\n';

    auto print_setting = [&](const std::string& name, const auto& value)
    {
        std::ostringstream oss;
        oss << name << ": ";

        std::string val_str;
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) {
            val_str = value;
        } else if constexpr (std::is_floating_point_v<std::decay_t<decltype(value)>>) {
            std::ostringstream tmp;
            tmp.precision(2);
            tmp << std::fixed << value;
            val_str = tmp.str();
        } else {
            val_str = std::to_string(value);
        }

        oss << val_str;
        std::string line_str = oss.str();

        if ((int)line_str.size() < width)
            line_str += std::string(width - line_str.size(), ' ');

        std::cout << line_str << '\n';
    };

    print_setting("data_type", data_type);
    print_setting("data_path", data_path);
    print_setting("query_path", query_path);
    print_setting("chunks_groundtruth_path", chunks_groundtruth_path);
    print_setting("chunk_size", chunk_size);

    print_setting("R", R);
    print_setting("L", L);
    print_setting("K", K);
    print_setting("alpha", alpha);

    print_setting("insert_threads", insert_threads);
    print_setting("consolidate_threads", consolidate_threads);
    print_setting("build_threads", build_threads);
    print_setting("search_threads", search_threads);

    std::cout << line << "\n\n";
}

template <typename T, typename TagT = uint32_t>
void calculate_recall(const uint32_t K, TagT*& groundtruth_ids, std::vector<TagT>& query_result_tags, const uint32_t query_num, const uint32_t groundtruth_dim, uint32_t chunk_index = 1, size_t chunk_size = 100000) {
    double total_recall = 0.0;

    for (int32_t i = 0; i < query_num; i++) {
        std::set<uint32_t> groundtruth_closest_neighbors;
        std::set<uint32_t> calculated_closest_neighbors;
        for (int32_t j = 0; j < K; j++) {
            calculated_closest_neighbors.insert(*(query_result_tags.data() + i * K + j) - (chunk_index - 1) * chunk_size);
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
void search(diskann::AbstractIndex &index, const T* query, size_t query_num, uint32_t query_aligned_dim, uint32_t K, uint32_t L, uint32_t search_threads, std::vector<uint32_t>& query_result_tags, std::vector<T *>& res) {
    std::vector<double> latencies_ms(query_num, 0.0);

    auto global_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        index.search_with_tags(query + i * query_aligned_dim,
                                K, L,
                                query_result_tags.data() + i * K,
                                nullptr, res);

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

    std::cout << "search(): " << query_num << " queries using "
              << search_threads << " threads\n";
    std::cout << "  Total time:     " << total_time_ms << " ms\n";
    std::cout << "  Avg latency:    " << avg_latency_ms << " ms/query\n";
    std::cout << "  QPS:            " << qps << "\n";
    std::cout << "  QPS/thread:     " << qps_per_thread << "\n";
}

template <typename T, typename TagT = uint32_t>
void build(diskann::AbstractIndex &index, size_t end, int32_t thread_count, T *data, size_t aligned_dim) {
    #pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t j = 0; j < (int64_t)end; j++) {
        index.insert_point(data + j * aligned_dim, 1 + static_cast<TagT>(j));
    }
    std::cout << "built using [" << 0 << "," << end-1 << "]" << std::endl;
}

template <typename T, typename TagT = uint32_t>
void insert(diskann::AbstractIndex &index, size_t start, size_t end,
            int32_t thread_count, T* data, size_t aligned_dim) {

    size_t count = end - start;
    std::vector<double> latencies_ms(count, 0.0);

    auto global_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t j = start; j < (int64_t)end; j++) {
        auto local_start = std::chrono::high_resolution_clock::now();

        index.insert_point(data + j * aligned_dim, 1 + static_cast<TagT>(j));

        auto local_end = std::chrono::high_resolution_clock::now();
        latencies_ms[j - start] = std::chrono::duration<double, std::milli>(local_end - local_start).count();
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(global_end - global_start).count();
    double total_time_sec = total_time_ms / 1000.0;

    double total_latency = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    double avg_latency_ms = total_latency / static_cast<double>(count);
    double qps = static_cast<double>(count) / total_time_sec;
    double qps_per_thread = qps / static_cast<double>(thread_count);

    std::cout << "insert(): inserted [" << start << "," << end - 1 << "] using "
              << thread_count << " threads\n";
    std::cout << "  Total time:     " << total_time_ms << " ms\n";
    std::cout << "  Avg latency:    " << avg_latency_ms << " ms/point\n";
    std::cout << "  QPS:            " << qps << "\n";
    std::cout << "  QPS/thread:     " << qps_per_thread << "\n";
}

template <typename T, typename TagT = uint32_t>
void delete_and_consolidate(diskann::AbstractIndex &index,
                             const diskann::IndexWriteParameters &delete_params,
                             size_t start, size_t end) {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = start; i < end; i++) {
        index.lazy_delete(static_cast<TagT>(1 + i));
    }

    index.consolidate_deletes(delete_params);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "deleted and consolidated [" << start << "," << end - 1
              << "] in " << duration_ms << " ms" << std::endl;
}

template <typename T, typename TagT = uint32_t>
void experiment(const std::string &data_path,
                const std::string &query_path,
                const std::string &chunks_groundtruth_path, 
                size_t chunk_size, uint32_t R, 
                uint32_t L, uint32_t K, float alpha,
                uint32_t build_threads, uint32_t insert_threads,
                uint32_t consolidate_threads, uint32_t search_threads) {

    size_t dim, aligned_dim;
    size_t num_points;
    diskann::get_bin_metadata(data_path, num_points, dim);
    aligned_dim = ROUND_UP(dim, 8);
    size_t num_chunks = (num_points / chunk_size);

    diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(L, R)
                                                .with_alpha(alpha)
                                                .with_num_threads(consolidate_threads)
                                                .build();

    auto index_search_params = diskann::IndexSearchParams(L, search_threads);

    diskann::IndexConfig index_config = diskann::IndexConfigBuilder()
                                            .with_metric(diskann::L2)
                                            .with_dimension(dim)
                                            .with_max_points(num_points)
                                            .is_dynamic_index(true)
                                            .with_index_write_params(params)
                                            .with_index_search_params(index_search_params)
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

    T *data = nullptr;
    diskann::alloc_aligned((void **)&data, num_points * aligned_dim * sizeof(T), 8 * sizeof(T));
    diskann::load_aligned_bin<T>(data_path, data, num_points, dim, aligned_dim);

    // Build the index using the first chunk
    index->set_start_points_at_random(static_cast<T>(0));
    build<T, TagT>(*index, chunk_size, build_threads, data, aligned_dim);

    // Load queries
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);

    // I don't know why it is need but it is somehow necessary to call seach_with_tags.
    std::vector<T *> res = std::vector<T *>();

    // Allocate the space to store result of the queries
    std::vector<TagT> query_result_tags;
    query_result_tags.resize(query_num * K);

    // Perform a search on the first chunk
    search(*index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);

    // Load groundtruth ids for the results of the search on the first chunk
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(chunks_groundtruth_path + std::string("1_groundtruth.bin"), groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

    // Calculate recall for the first chunk
    calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim, 1, chunk_size);

    std::future<void> insert_task;
    std::future<void> delete_task;

    for (size_t current_chunk = 2; current_chunk <= num_chunks; current_chunk++) {
        // Calculate batch boundaries
        size_t insert_start = (current_chunk - 1) * chunk_size;
        size_t insert_end = current_chunk * chunk_size;
    
        size_t delete_start = (current_chunk - 2) * chunk_size;
        size_t delete_end = (current_chunk - 1) * chunk_size;
    
        // Launch insert
        insert_task = std::async(std::launch::async,
            [&index, insert_start, insert_end, data, aligned_dim, insert_threads]() {
                insert<T, TagT>(*index, insert_start, insert_end, insert_threads, data, aligned_dim);
            });
    
        // Build delete parameters
        diskann::IndexWriteParameters delete_params = diskann::IndexWriteParametersBuilder(params).with_num_threads(consolidate_threads).build();

        // Launch delete
        delete_task = std::async(std::launch::async,
            [&index, delete_params, delete_start, delete_end]() {
                delete_and_consolidate<T, TagT>(*index, delete_params, delete_start, delete_end);
            });
            
        // Wait for the current batches to be done
        insert_task.wait();
        delete_task.wait();

        // Perform a search on the current chunk availabe in the index
        search(*index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);

        // Load groundtruth ids for the results of the search on the current chunk
        diskann::load_truthset(chunks_groundtruth_path + std::to_string(current_chunk) + std::string("_groundtruth.bin"), groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

        // Calculate recall for the current chunk
        calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim, current_chunk, chunk_size);
    }
    
    diskann::aligned_free(data);
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, chunks_groundtruth_path;
    uint32_t R, L, K, build_threads, insert_threads, consolidate_threads, search_threads;
    float alpha;
    size_t chunk_size;

    po::options_description desc;

    try
    {
        po::options_description desc("Allowed options");

        desc.add_options()
            ("help,h", "Print information on arguments")
            ("data_type", po::value<std::string>(&data_type)->required(), "Type of data")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to query")
            ("chunks_groundtruth_path", po::value<std::string>(&chunks_groundtruth_path)->required(), "Path to groundtruth for chunks")
            ("chunk_size", po::value<size_t>(&chunk_size)->required(), "Chunk size")
            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("L", po::value<uint32_t>(&L)->required(), "Value of L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("insert_threads", po::value<uint32_t>(&insert_threads)->required(), "Threads for insertion")
            ("consolidate_threads", po::value<uint32_t>(&consolidate_threads)->required(), "Threads for consolidation")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching");
    
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

    print_experiment_settings(data_type, data_path, query_path, chunks_groundtruth_path,
        chunk_size, R, L, K, alpha,
        build_threads, insert_threads, consolidate_threads, search_threads);

    if (data_type == std::string("int8"))
        experiment<int8_t>(data_path, query_path, chunks_groundtruth_path, chunk_size, R, L, K, alpha, build_threads, insert_threads, consolidate_threads, search_threads);
    else if (data_type == std::string("uint8"))
        experiment<uint8_t>(data_path, query_path, chunks_groundtruth_path, chunk_size, R, L, K, alpha, build_threads, insert_threads, consolidate_threads, search_threads);
    else if (data_type == std::string("float"))
        experiment<float>(data_path, query_path, chunks_groundtruth_path, chunk_size, R, L, K, alpha, build_threads, insert_threads, consolidate_threads, search_threads);
    else
        std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;

    return 0;
}
