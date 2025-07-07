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
#include <abstract_index.h>
#include <index_factory.h>

#include "utils.h"
#include "filter_utils.h"
#include "program_options_utils.hpp"

#include "memory_mapper.h"

namespace po = boost::program_options;

void print_experiment_settings(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    uint32_t R, uint32_t L, uint32_t K,
    float alpha,
    uint32_t insert_threads, uint32_t consolidate_threads,
    uint32_t build_threads, uint32_t search_threads,
    float batch_size_in_percentage,
    uint32_t n_iterations)
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
            tmp.precision(1);
            tmp << std::fixed << value;
            val_str = tmp.str();
        } else {
            val_str = std::to_string(value);
        }

        oss << val_str;
        std::string line_str = oss.str();

        // pad right with spaces to fill width
        if ((int)line_str.size() < width)
            line_str += std::string(width - line_str.size(), ' ');

        std::cout << line_str << '\n';
    };

    print_setting("data_type", data_type);
    print_setting("data_path", data_path);
    print_setting("query_path", query_path);
    print_setting("groundtruth_path", groundtruth_path);

    print_setting("R", R);
    print_setting("L", L);
    print_setting("K", K);
    print_setting("alpha", alpha);

    print_setting("insert_threads", insert_threads);
    print_setting("consolidate_threads", consolidate_threads);
    print_setting("build_threads", build_threads);
    print_setting("search_threads", search_threads);

    print_setting("batch_size_in_percentage", batch_size_in_percentage);
    print_setting("iterations", n_iterations);

    std::cout << line << "\n\n";
}


std::vector<uint32_t> generate_random_uint32(size_t n_delete, uint32_t n_points) {
    if (n_delete > n_points) {
        throw std::invalid_argument("Cannot generate more unique numbers than the range allows.");
    }

    std::unordered_set<uint32_t> unique_numbers;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, n_points - 1);

    while (unique_numbers.size() < n_delete) {
        unique_numbers.insert(dist(gen));
    }

    return std::vector<uint32_t>(unique_numbers.begin(), unique_numbers.end());
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
void build(diskann::AbstractIndex &index, uint32_t num_vectors, size_t build_threads, T* data, uint32_t vector_aligned_dim) {
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)build_threads) schedule(dynamic)
    for (int64_t j = 0; j < (int64_t)num_vectors; j++) {
        index.insert_point(&data[j * vector_aligned_dim], 1 + static_cast<TagT>(j));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "build() took " << duration_ms << " ms\n";
}

template <typename T, typename TagT = uint32_t>
void insert_batch(diskann::AbstractIndex &index, size_t insert_threads, T *data, size_t vector_aligned_dim, std::vector<uint32_t>& tags_to_be_inserted) {
    size_t n = tags_to_be_inserted.size();
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)insert_threads) schedule(dynamic)
    for (int64_t j = 0; j < (int64_t)n; j++) {
        index.insert_point(&data[tags_to_be_inserted[j] * vector_aligned_dim], 1 + static_cast<TagT>(tags_to_be_inserted[j]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration<double>(end - start).count(); // in seconds

    double qps = static_cast<double>(n) / duration_sec;
    double qps_per_thread = qps / static_cast<double>(insert_threads);

    std::cout << "insert_batch(): " << n << " inserts in "
              << duration_sec << " sec\n";
    std::cout << "IQPS: " << qps << "\n";
    std::cout << "IQPS/thread: " << qps_per_thread << "\n";
}

template <typename T, typename TagT = uint32_t>
void delete_and_consolidate(diskann::AbstractIndex &index,
                            diskann::IndexWriteParameters &delete_params,
                            std::vector<uint32_t>& tags_to_be_deleted) {
    auto start = std::chrono::high_resolution_clock::now();

    for (auto tag : tags_to_be_deleted)
        index.lazy_delete(static_cast<TagT>(1 + tag));

    index.consolidate_deletes(delete_params);

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "delete_and_consolidate() took " << duration_ms << " ms\n";
}

template <typename T, typename TagT = uint32_t>
void calculate_recall(const uint32_t K, TagT*& groundtruth_ids, std::vector<TagT>& query_result_tags, const uint32_t query_num, const uint32_t groundtruth_dim) {
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
void experiment(const std::string &data_path, const std::string &query_path, const std::string &groundtruth_path,
                const uint32_t L, const uint32_t R, const uint32_t K, const float alpha, const uint32_t insert_threads,
                const uint32_t consolidate_threads, const uint32_t search_threads, const uint32_t build_threads,
                const float batch_size_in_percentage, const uint32_t n_iterations)
{
    // Prepare the index
    auto build_params = diskann::IndexWriteParametersBuilder(L, R)
                                                     .with_alpha(alpha)
                                                     .with_num_threads(build_threads)
                                                     .with_filter_list_size(L)
                                                     .build();

    
    // Search Paramas
    auto search_params = diskann::IndexSearchParams(L, search_threads);
    
    // Delete Params
    auto delete_params = diskann::IndexWriteParametersBuilder(L, R)
                                                      .with_alpha(alpha)
                                                      .with_num_threads(consolidate_threads)
                                                      .with_filter_list_size(L)
                                                      .build();

    // Load the metadata about the file containing vectors
    size_t vector_dim, vector_aligned_dim;
    size_t num_vectors;
    diskann::get_bin_metadata(data_path, num_vectors, vector_dim);
    vector_aligned_dim = ROUND_UP(vector_dim, 8);


    // Build Params
    auto index_config = diskann::IndexConfigBuilder()
                            .with_metric(diskann::L2)
                            .with_dimension(vector_dim)
                            .with_max_points(num_vectors)
                            .is_dynamic_index(true)
                            .is_enable_tags(true)
                            .is_use_opq(false)
                            .with_num_pq_chunks(false)
                            .is_pq_dist_build(false)
                            .with_tag_type(diskann_type_to_name<TagT>())
                            .with_data_type(diskann_type_to_name<T>())
                            .with_index_write_params(build_params)
                            .with_index_search_params(search_params)
                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                            .is_concurrent_consolidate(true)
                            .build();

    // Create the index object
    diskann::IndexFactory index_factory = diskann::IndexFactory(index_config);
    auto index = index_factory.create_instance();

    // Load the vectors
    T *data = nullptr;
    diskann::alloc_aligned((void **)&data, num_vectors * vector_aligned_dim * sizeof(T), 8 * sizeof(T));
    diskann::load_aligned_bin<T>(data_path, data, num_vectors, vector_dim, vector_aligned_dim);

    // Generate dummy tags
    std::vector<TagT> tags(num_vectors);
    std::iota(tags.begin(), tags.end(), static_cast<TagT>(0));

    // Build the index
    build<T, TagT>(*index, num_vectors, build_params.num_threads, data, vector_aligned_dim);

    // Load queries
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);
    
    // I don't know why it is need but it is somehow necessary to call seach_with_tags.
    std::vector<T *> res = std::vector<T *>();

    // Allocate the space to store result of the queries
    std::vector<TagT> query_result_tags;
    query_result_tags.resize(query_num * K);

    // Load groundtruth ids for the results
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

    // Perform a search before doing any deletion or insertions
    std::cout << std::endl << std::endl << "====================================== Initial Search ===============================" << std::endl << std::endl;
    search(*index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);
    
    // Calculate the initial recall
    calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim);
    std::cout << std::endl << std::endl << "=====================================================================================" << std::endl << std::endl;



    // Perform the iterations
    for (size_t iteration = 0; iteration < n_iterations; iteration++) {
        std::cout << std::endl << std::endl << "========================================Cycle " << iteration + 1 << "======================================" << std::endl << std::endl;
        // Generate the random tags which will be first deleted then re-inserted.
        auto tags_in_this_batch = generate_random_uint32(num_vectors * (batch_size_in_percentage / 100.0),  num_vectors);

        // Delete the batch
        delete_and_consolidate<T, TagT>(*index, delete_params, tags_in_this_batch);

        // Reinsert the batch
        insert_batch<T, TagT>(*index, build_params.num_threads, data, vector_aligned_dim, tags_in_this_batch);
 
        // Perform a search for all the queries
        search(*index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);

        // Calculate recalls
        calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim);
        std::cout << std::endl << std::endl << "=====================================================================================" << std::endl << std::endl;
    }

    // Free the loaded vectors
    diskann::aligned_free(data);
}

int main(int argc, char **argv)
{
    std::string data_type, data_path, query_path, groundtruth_path;
    uint32_t insert_threads, consolidate_threads, search_threads, build_threads, R, L, K, n_iterations;
    float alpha, batch_size_in_percentage;
    po::options_description desc;

    try
    {

        desc.add_options()
            ("data_type", po::value<std::string>(&data_type)->required(),"Data type")
            ("data_path", po::value<std::string>(&data_path)->required(), "Path to data")
            ("query_path", po::value<std::string>(&query_path)->required(), "Path to queries")
            ("groundtruth_path", po::value<std::string>(&groundtruth_path)->required(), "Path to groundtruth")
            ("insert_threads", po::value<uint32_t>(&insert_threads)->required(),"Insert threads")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Search threads")
            ("build_threads", po::value<uint32_t>(&build_threads)->required(),"Build threads")
            ("consolidate_threads", po::value<uint32_t>(&consolidate_threads)->required(),"Consolidate threads")
            ("R", po::value<uint32_t>(&R)->required(), "R parameter")
            ("L", po::value<uint32_t>(&L)->required(), "L parameter")
            ("K", po::value<uint32_t>(&K)->required(),"K parameter")
            ("batch_size_in_percentage", po::value<float>(&batch_size_in_percentage)->required(),"Batch size in percentage")
            ("iterations", po::value<uint32_t>(&n_iterations)->required(), "Number of iterations")
            ("alpha", po::value<float>(&alpha)->required(), "Alpha");

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

    print_experiment_settings(data_type, data_path, query_path, groundtruth_path,
        R, L, K, alpha,
        insert_threads, consolidate_threads,
        build_threads, search_threads,
        batch_size_in_percentage, n_iterations);

    if (data_type == std::string("uint8"))
        experiment<uint8_t>(data_path, query_path, groundtruth_path, L, R, K, alpha, insert_threads,
                            consolidate_threads, search_threads, build_threads, batch_size_in_percentage, n_iterations);

    else if (data_type == std::string("int8"))
        experiment<int8_t>(data_path, query_path, groundtruth_path, L, R, K, alpha, insert_threads, consolidate_threads,
                           search_threads, build_threads, batch_size_in_percentage, n_iterations);

    else if (data_type == std::string("float"))
        experiment<float>(data_path, query_path, groundtruth_path, L, R, K, alpha, insert_threads, consolidate_threads,
                          search_threads, build_threads, batch_size_in_percentage, n_iterations);
    else
    {
        std::cout << "Unsupported datatype" << std::endl;
        return -1;
    }

    return 0;
}
