// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <boost/program_options.hpp>
#include <future>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>

#include "utils.h"
#include "filter_utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"

#include "memory_mapper.h"

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

inline std::string get_current_timestamp() {
    using namespace std::chrono;

    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::time_t now_time = system_clock::to_time_t(now);
    std::tm local_tm = *std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
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

    std::cout << "[" << get_current_timestamp() << "] "
              << "search(): queries=" << query_num
              << " threads=" << search_threads
              << " total_time_ms=" << total_time_ms
              << " avg_latency_ms=" << avg_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << std::endl;
}

template <typename T, typename TagT = uint32_t>
void build(diskann::AbstractIndex &index, size_t end, int32_t thread_count, T *data, size_t aligned_dim) {
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t j = 0; j < (int64_t)end; j++) {
        index.insert_point(data + j * aligned_dim, 1 + static_cast<TagT>(j));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "[" << get_current_timestamp() << "] "
              << "build(): range=[0," << end - 1 << "]"
              << " threads=" << thread_count
              << " duration_ms=" << duration_ms
              << std::endl;
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

    std::cout << "[" << get_current_timestamp() << "] "
              << "insert(): range=[" << start << "," << end - 1 << "]"
              << " threads=" << thread_count
              << " total_time_ms=" << total_time_ms
              << " avg_latency_ms=" << avg_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << std::endl;
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

    std::cout << "[" << get_current_timestamp() << "] "
              << "delete_and_consolidate(): range=[" << start << "," << end - 1
              << "] duration_ms=" << duration_ms << std::endl;
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
    build<T, TagT>(*index, num_points, build_threads, data, aligned_dim);

    // Load queries
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);

    // I don't know why it is need but it is somehow necessary to call seach_with_tags.
    std::vector<T *> res = std::vector<T *>();

    // Allocate the space to store result of the queries
    std::vector<TagT> query_result_tags;
    query_result_tags.resize(query_num * K);

    std::future<void> insert_task;
    std::future<void> delete_task;
    std::future<void> search_task;              

    search_task = std::async(std::launch::async,
        [&index, query, query_num, query_aligned_dim, K, L, search_threads, &query_result_tags, &res]() {
            for (size_t i = 0; i < 1000; i++) {
                search<T, TagT>(*index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);
            }
        });

    
    // Do some isolated search before starting the dynamic phase
    std::this_thread::sleep_for(std::chrono::seconds(5));

    for (size_t current_chunk = 2; current_chunk <= num_chunks; current_chunk++) {
        // Calculate batch boundaries

        size_t delete_start = (current_chunk - 2) * chunk_size;
        size_t delete_end = (current_chunk - 1) * chunk_size;
    
        std::cout << "[" << get_current_timestamp() << "] " << "only_delete_phase_start()" << std::endl;

        // Build delete parameters
        diskann::IndexWriteParameters delete_params = diskann::IndexWriteParametersBuilder(params).with_num_threads(consolidate_threads).build();

        // Launch delete
        delete_task = std::async(std::launch::async,
            [&index, delete_params, delete_start, delete_end]() {
                delete_and_consolidate<T, TagT>(*index, delete_params, delete_start, delete_end); // Delete the chunk that was just inserted
            });
        delete_task.wait(); // Wait for delete to finish
        
        std::this_thread::sleep_for(std::chrono::seconds(2));


        // Only insert phase
        std::cout << "[" << get_current_timestamp() << "] " << "exclude_start()" << std::endl;
        
        // Launch insert
        insert_task = std::async(std::launch::async,
            [&index, delete_start, delete_end, data, aligned_dim, insert_threads]() {
                insert<T, TagT>(*index, delete_start, delete_end, insert_threads, data, aligned_dim);
            });

        insert_task.wait(); 

        std::cout << "[" << get_current_timestamp() << "] " << "exclude_end()" << std::endl;


        // No insert and delete phase
        std::cout << "[" << get_current_timestamp() << "] " << "sleep_insert_delete_phase_start()" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));

    }

    std::cout << "[" << get_current_timestamp() << "] " << "dynamic_phase_end()" << std::endl;
    
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
