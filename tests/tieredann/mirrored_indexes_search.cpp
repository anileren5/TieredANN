// System headers
#include <iomanip>
#include <omp.h>
#include <boost/program_options.hpp>
#include <future>

// Greator (Disk index) headers
#include "greator/pq_flash_index.h"
#include "greator/aux_utils.h"
#include "greator/linux_aligned_file_reader.h"
#include "greator/utils.h"

// DiskANN (Memory index) headers
#include "diskann/index_factory.h"

#define N_SEARCH_ITER 1

namespace po = boost::program_options;

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

void print_experiment_settings(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& resulsts_prefix,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha, double hit_rate,
    uint32_t insert_threads,
    uint32_t consolidate_threads,
    uint32_t build_threads,
    uint32_t search_threads,
    const std::string& distance_metric,
    int single_file_index,
    int disk_index_already_built,
    int tags_enabled,
    uint32_t beamwidth,
    uint64_t num_nodes_to_cache)
{
    const int width = 70;
    std::string line(width, '=');

    std::cout << "\n" << line << '\n';
    std::cout << "Experiment Settings\n";
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
    print_setting("groundtruth_path", groundtruth_path);
    print_setting("results_prefix", resulsts_prefix);
    print_setting("disk_index_prefix", disk_index_prefix);

    print_setting("R", R);
    print_setting("L", L);
    print_setting("K", K);
    print_setting("B", B);
    print_setting("M", M);

    print_setting("alpha", alpha);
    print_setting("hit_rate", hit_rate);

    print_setting("build_threads", build_threads);
    print_setting("insert_threads", insert_threads);
    print_setting("consolidate_threads", consolidate_threads);
    print_setting("search_threads", search_threads);

    print_setting("distance_metric", distance_metric);
    print_setting("single_file_index", single_file_index);
    print_setting("disk_index_already_built", disk_index_already_built);
    print_setting("tags_enabled", tags_enabled);
    print_setting("beamwidth", beamwidth);
    print_setting("num_nodes_to_cache", num_nodes_to_cache);

    std::cout << line << "\n\n";
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
void memory_index_search(diskann::AbstractIndex &index, const T* query, size_t query_num, uint32_t query_aligned_dim, uint32_t K, uint32_t L, uint32_t search_threads, std::vector<uint32_t>& query_result_tags, std::vector<T *>& res) {
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
              << "memory_search(): queries=" << query_num
              << " threads=" << search_threads
              << " total_time_ms=" << total_time_ms
              << " avg_latency_ms=" << avg_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << std::endl;
}

template <typename T, typename TagT = uint32_t>
void memory_index_build(diskann::AbstractIndex &index, size_t start, size_t end, int32_t thread_count, T *data, size_t aligned_dim) {
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t j = start; j < (int64_t)end; j++) {
        index.insert_point(data + j * aligned_dim, 1 + static_cast<TagT>(j)); // If indexes are added directly (without + 1), it fails. I don't know why.
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "[" << get_current_timestamp() << "] "
              << "build_memory_index(): "
              << " duration_ms=" << duration_ms
              << std::endl;
}

template <typename T, typename TagT = uint32_t>
void disk_index_build(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters, greator::Metric m, bool singleFile) 
{
    auto start_time = std::chrono::high_resolution_clock::now();


    greator::build_disk_index<T>(dataFilePath, indexFilePath,
                                      indexBuildParameters, m, singleFile);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                                  
    std::cout << "[" << get_current_timestamp() << "] "
            << "build_disk_index(): "
            << " duration_ms=" << duration_ms
            << std::endl;
}

template <typename T, typename TagT = uint32_t>
void disk_index_search(std::unique_ptr<greator::PQFlashIndex<T>>& index, const T* query, size_t query_num, uint32_t query_aligned_dim, uint32_t K, uint32_t L, uint32_t search_threads, std::vector<uint32_t>& query_result_tags, uint32_t beamwidth) {
    std::vector<float> query_result_dists(K * query_num);
    greator::QueryStats *stats = new greator::QueryStats[query_num];

    std::vector<double> latencies_ms(query_num, 0.0);
    auto global_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        index->cached_beam_search(
            query + (i * query_aligned_dim),
            (uint64_t) K, (uint64_t) L,
            query_result_tags.data() + (i * K),
            query_result_dists.data() + (i * K),
            (uint64_t)beamwidth, stats + i
        );

        auto end = std::chrono::high_resolution_clock::now();
        latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num * K; i++) {
        query_result_tags[i] += 1;
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
              << "disk_search(): queries=" << query_num
              << " threads=" << search_threads
              << " total_time_ms=" << total_time_ms
              << " avg_latency_ms=" << avg_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << std::endl;
            

}

#include <cstdlib> 
bool isHitDummy(double hit_rate) {
    return static_cast<double>(rand()) / RAND_MAX < hit_rate;
}

template <typename T, typename TagT = uint32_t>
void hybrid_search(double hit_rate, diskann::AbstractIndex &memory_index, std::unique_ptr<greator::PQFlashIndex<T>>& disk_index, const T* query, size_t query_num, uint32_t query_aligned_dim, uint32_t K, uint32_t L, uint32_t search_threads, std::vector<uint32_t>& query_result_tags, std::vector<T *>& res, uint32_t beamwidth) {
    std::vector<float> query_result_dists(K * query_num);
    greator::QueryStats *stats = new greator::QueryStats[query_num];

    std::vector<double> latencies_ms(query_num, 0.0);
    auto global_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads((int32_t)search_threads) schedule(dynamic)
    for (size_t i = 0; i < query_num; i++) {

        if (isHitDummy(hit_rate)) {
            auto start = std::chrono::high_resolution_clock::now();
        
            memory_index.search_with_tags(query + i * query_aligned_dim,
                                   K, L,
                                   query_result_tags.data() + i * K,
                                   nullptr, res);
    
            auto end = std::chrono::high_resolution_clock::now();
            latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
        } 
        else {
            auto start = std::chrono::high_resolution_clock::now();
            disk_index->cached_beam_search(
                query + (i * query_aligned_dim),
                (uint64_t) K, (uint64_t) L,
                query_result_tags.data() + (i * K),
                query_result_dists.data() + (i * K),
                (uint64_t)beamwidth, stats + i
            );
            auto end = std::chrono::high_resolution_clock::now();
            latencies_ms[i] = std::chrono::duration<double, std::milli>(end - start).count();
            for (size_t j = 0; j < K; j++) (query_result_tags.data() + (i * K))[j] += 1;
        }
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
              << "hybrid_search(): queries=" << query_num
              << " threads=" << search_threads
              << " total_time_ms=" << total_time_ms
              << " avg_latency_ms=" << avg_latency_ms
              << " qps=" << qps
              << " qps_per_thread=" << qps_per_thread
              << std::endl;

}

template <typename T = float, typename TagT = uint32_t>
void experiment(
    const std::string& data_type,
    const std::string& data_path,
    const std::string& query_path,
    const std::string& groundtruth_path,
    const std::string& resulsts_prefix,
    const std::string& disk_index_prefix,
    uint32_t R, uint32_t L, uint32_t K,
    uint32_t B, uint32_t M,
    float alpha,
    double hit_rate,
    uint32_t insert_threads,
    uint32_t consolidate_threads,
    uint32_t build_threads,
    uint32_t search_threads,
    const std::string& distance_metric,
    int single_file_index,
    int disk_index_already_built,
    int tags_enabled,
    uint32_t beamwidth,
    uint64_t num_nodes_to_cache) 
{
    // Read metadata
    size_t dim, aligned_dim;
    size_t num_points;
    diskann::get_bin_metadata(data_path, num_points, dim);
    aligned_dim = ROUND_UP(dim, 8);

    // Read the data
    T *data = nullptr;
    diskann::alloc_aligned((void **)&data, num_points * aligned_dim * sizeof(T), 8 * sizeof(T));
    diskann::load_aligned_bin<T>(data_path, data, num_points, dim, aligned_dim);

    // Load groundtruth ids for the results
    TagT *groundtruth_ids = nullptr;
    float *groundtruth_dists = nullptr;
    size_t n_groundtruth, groundtruth_dim;
    diskann::load_truthset(groundtruth_path, groundtruth_ids, groundtruth_dists, n_groundtruth, groundtruth_dim);

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
    auto memory_index = memory_index_factory.create_instance();

    memory_index->set_start_points_at_random(static_cast<T>(0));

    // Use all the points to build the index
    memory_index_build<T, TagT>(*memory_index, 0, num_points, search_threads, data, aligned_dim);

    delete[] data;

    // Load queries
    size_t query_num, query_dim, query_aligned_dim;
    T *query = nullptr;
    diskann::load_aligned_bin<T>(query_path, query, query_num, query_dim, query_aligned_dim);

    // I don't know why it is need but it is somehow necessary to call seach_with_tags.
    std::vector<T *> res = std::vector<T *>();

    // Allocate the space to store result of the queries
    std::vector<TagT> query_result_tags;
    query_result_tags.resize(query_num * K);

    // Perform search in memory index
    std::future<void> memory_search_task;   

    memory_search_task = std::async(std::launch::async,
        [&memory_index, query, query_num, query_aligned_dim, K, L, search_threads, &query_result_tags, &res, &groundtruth_ids, groundtruth_dim]() {
            for (size_t i = 0; i < N_SEARCH_ITER; i++) {
                memory_index_search<T, TagT>(*memory_index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res);
                calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim);
            }
        });

    memory_search_task.wait();

    // Build disk index
    if (disk_index_already_built == 0) {
        std::string disk_index_params = std::to_string(R) + " " + std::to_string(L) + " " + std::to_string(B) + " " + std::to_string(M) + " " + std::to_string(build_threads);
        disk_index_build<T>(data_path.c_str(), disk_index_prefix.c_str(), disk_index_params.c_str() , greator::Metric::L2, false);
    }

    // Load disk index
    std::shared_ptr<greator::AlignedFileReader> reader = nullptr;
    reader.reset(new greator::LinuxAlignedFileReader());
    std::unique_ptr<greator::PQFlashIndex<T>> disk_index(new greator::PQFlashIndex<T>(greator::Metric::L2, reader, single_file_index, false));
    disk_index->load(disk_index_prefix.c_str(), build_threads);

    // Perform search on disk index
    std::future<void> disk_search_task;   

    disk_search_task = std::async(std::launch::async,
        [&disk_index, query, query_num, query_aligned_dim, K, L, search_threads, &query_result_tags, &res, beamwidth, &groundtruth_ids, groundtruth_dim]() {
            for (size_t i = 0; i < N_SEARCH_ITER; i++) {
                disk_index_search(disk_index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, beamwidth);
                calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim);
            }
        });

    disk_search_task.wait();

    // Perform hybrid search: Route queries to memory index and disk index with 50% probability for each.
    std::future<void> hybrid_search_task;   

    hybrid_search_task = std::async(std::launch::async,
        [&memory_index, &disk_index, query, query_num, query_aligned_dim, K, L, search_threads, &query_result_tags, &res, beamwidth, &groundtruth_ids, groundtruth_dim, hit_rate]() {
            for (size_t i = 0; i < N_SEARCH_ITER; i++) {
                hybrid_search(hit_rate, *memory_index, disk_index, query, query_num, query_aligned_dim, K, L, search_threads, query_result_tags, res, beamwidth);
                calculate_recall<T>(K, groundtruth_ids, query_result_tags, query_num, groundtruth_dim);
            }
        });

    hybrid_search_task.wait();

    if (groundtruth_dists) delete[] groundtruth_dists;
    if (groundtruth_ids) delete[] groundtruth_ids;
}

int main(int argc, char **argv) {
    std::string data_type, data_path, query_path, groundtruth_path, results_prefix, disk_index_prefix;
    uint32_t R, L, K, B, M;
    uint32_t build_threads, insert_threads, consolidate_threads, search_threads;
    float alpha;
    std::string distance_metric;
    int single_file_index, tags_enabled, disk_index_already_built;
    uint32_t beamwidth;
    uint64_t num_nodes_to_cache;
    double hit_rate;

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
            ("results_prefix", po::value<std::string>(&results_prefix)->required(), "Prefix to disk search results")
            ("disk_index_prefix", po::value<std::string>(&disk_index_prefix)->required(), "Prefix to index")

            ("R", po::value<uint32_t>(&R)->required(), "Value of R")
            ("L", po::value<uint32_t>(&L)->required(), "Value of L")
            ("K", po::value<uint32_t>(&K)->required(), "Value of K")
        
            ("B", po::value<uint32_t>(&B)->default_value(8), "Value of B")
            ("M", po::value<uint32_t>(&M)->default_value(8), "Value of M")
        
            ("build_threads", po::value<uint32_t>(&build_threads)->required(), "Threads for building")
            ("insert_threads", po::value<uint32_t>(&insert_threads)->required(), "Threads for inserting")
            ("consolidate_threads", po::value<uint32_t>(&consolidate_threads)->required(), "Threads for consolidation")
            ("search_threads", po::value<uint32_t>(&search_threads)->required(), "Threads for searching")
        
            ("alpha", po::value<float>(&alpha)->required(), "Alpha parameter")
            ("hit_rate", po::value<double>(&hit_rate)->required(), "Hit rate for hybrid search")
        
            ("distance_metric", po::value<std::string>(&distance_metric)->default_value("l2"), "Distance metric")
            ("single_file_index", po::value<int>(&single_file_index)->default_value(0), "Single file index (0/1)")
            ("disk_index_already_built", po::value<int>(&disk_index_already_built)->default_value(1), "Disk index already built (0/1)")
            ("tags_enabled", po::value<int>(&tags_enabled)->default_value(0), "Tags enabled (0/1)")
            ("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(2), "Beamwidth")
            ("num_nodes_to_cache", po::value<uint64_t>(&num_nodes_to_cache)->default_value(500), "Number of nodes to cache around medoid");
    
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

    // Print experiment parameters
    print_experiment_settings(
        data_type, data_path, query_path, groundtruth_path, results_prefix, disk_index_prefix,
        R, L, K, B, M,
        alpha, hit_rate,
        insert_threads, consolidate_threads, build_threads, search_threads,
        distance_metric, single_file_index, disk_index_already_built, tags_enabled, beamwidth, num_nodes_to_cache
    );

    // Run the experiment
    experiment(
        data_type, data_path, query_path, groundtruth_path, results_prefix, disk_index_prefix,
        R, L, K, B, M,
        alpha, hit_rate,
        insert_threads, consolidate_threads, build_threads, search_threads,
        distance_metric, single_file_index, disk_index_already_built, tags_enabled, beamwidth, num_nodes_to_cache
    );
}