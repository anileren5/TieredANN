// Greator (Disk index) headers
#include "greator/pq_flash_index.h"
#include "greator/aux_utils.h"
#include "greator/linux_aligned_file_reader.h"
#include "greator/utils.h"

// DiskANN (Memory index) headers
#include "diskann/index_factory.h"

// TieredANN headers
#include "percentile_stats.h"
#include <cstdint>
#include "tieredann/insert_thread_pool.h"
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace tieredann {

    template <typename T, typename TagT = uint32_t>
    class TieredIndex {
        
        private:
            std::unique_ptr<greator::PQFlashIndex<T, TagT>> disk_index;
            std::unique_ptr<diskann::AbstractIndex> memory_index;
            std::string data_path;
            std::string disk_index_prefix;
            size_t dim, aligned_dim;
            size_t num_points;
            uint32_t search_threads;
            bool use_reconstructed_vectors;
            std::unique_ptr<tieredann::InsertThreadPool<T, TagT>> insert_pool;
            std::unordered_map<uint32_t, double> theta_map;
            std::mutex theta_map_mutex;
            double p, deviation_factor;
            std::atomic<size_t> num_points_in_memory_index{0};
            
            void memory_index_insert_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim) {
                std::vector<T*> vectors;
                vectors.reserve(to_be_inserted.size());
                size_t successful_inserts = 0;
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    T* vector = nullptr;
                    diskann::alloc_aligned((void**)&vector, aligned_dim * sizeof(T), 8 * sizeof(T));
                    diskann::load_vector_by_index(data_path, vector, dim, to_be_inserted[i]);
                    vectors.push_back(vector);
                }
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    int ret = index->insert_point(vectors[i], 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                num_points_in_memory_index.fetch_add(successful_inserts, std::memory_order_relaxed);
                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const size_t dim) {
                std::vector<std::vector<T>> reconstructed_vectors = this->disk_index->inflate_vectors_by_tags(to_be_inserted);
                size_t successful_inserts = 0;
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    int ret = index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                num_points_in_memory_index.fetch_add(successful_inserts, std::memory_order_relaxed);
            }

            bool isHit(const T* query_ptr, uint32_t K, uint32_t L, const float* distances) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (this->get_number_of_vectors_in_memory_index() < K){
                    return false;
                }                
                else if (distances[K - 1] > (1 + deviation_factor)*theta_map[K]) {
                    return false;
                }
                return true;
            }

        
        public:
            TieredIndex(const std::string& data_path,
                        const std::string& disk_index_prefix,
                        uint32_t R, uint32_t L,
                        uint32_t B, uint32_t M,
                        float alpha,
                        uint32_t consolidate_threads,
                        uint32_t build_threads,
                        uint32_t search_threads,
                        int disk_index_already_built,
                        bool use_reconstructed_vectors,
                        double p,
                        double deviation_factor, 
                        uint32_t n_theta_estimation_queries):
                        data_path(data_path),
                        disk_index_prefix(disk_index_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor)
            {                
                // Read metadata
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
                memory_index = memory_index_factory.create_instance();
                memory_index->set_start_points_at_random(static_cast<T>(0));
            
                std::cout << "TieredIndex memory index built successfully!" << std::endl;

                // Build disk index
                if (disk_index_already_built == 0) {
                    std::string disk_index_params = std::to_string(R) + " " + std::to_string(L) + " " + std::to_string(B) + " " + std::to_string(M) + " " + std::to_string(build_threads);
                    greator::build_disk_index<T>(data_path.c_str(), disk_index_prefix.c_str(), disk_index_params.c_str(), greator::Metric::L2, false);
                }

                // Load disk index
                std::shared_ptr<greator::AlignedFileReader> reader = nullptr;
                reader.reset(new greator::LinuxAlignedFileReader());
                std::unique_ptr<greator::PQFlashIndex<T>> temp_disk_index(new greator::PQFlashIndex<T>(greator::Metric::L2, reader, false, false));
                disk_index = std::move(temp_disk_index);
                disk_index->load(disk_index_prefix.c_str(), build_threads);
                
                // Cache vectors near the centroid of the disk index.
                std::vector<uint32_t> node_list;
                disk_index->cache_bfs_levels(500, node_list);
                disk_index->load_cache_list(node_list);
                node_list.clear();
                node_list.shrink_to_fit();

                std::cout << "TieredIndex disk index built successfully!" << std::endl;

                // Initialize insert thread pool for async insertions
                if (use_reconstructed_vectors) {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim) {
                        this->memory_index_insert_reconstructed_sync(index, to_be_inserted, dim);
                    };
                    insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(4, task);
                } else {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim) {
                        this->memory_index_insert_sync(index, to_be_inserted, data_path, dim);
                    };
                    insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(4, task);
                }

                std::cout << "TieredIndex built successfully!" << std::endl;

                // Calculate an estimation of theta from the disk index
                std::unordered_map<uint32_t, double> theta_sums = {
                    {1, 0.0}, {5, 0.0}, {10, 0.0}, {50, 0.0}, {100, 0.0}
                };
                for (size_t i = 0; i < n_theta_estimation_queries; i++) {
                    // Generate a random query by taking a weighted average of two random vectors from the disk index
                    std::vector<T> query(this->dim);
                    std::vector<T> query_2(this->dim);
                    diskann::load_vector_by_index(data_path, query.data(), this->dim, rand() % this->num_points);
                    diskann::load_vector_by_index(data_path, query_2.data(), this->dim, rand() % this->num_points);
                    double random_weight = (double)rand() / RAND_MAX * 0.2;
                    for (size_t j = 0; j < this->dim; j++) {
                        query[j] = random_weight * query[j] + (1 - random_weight) * query_2[j];
                    }

                    // Search the disk index
                    std::vector<TagT> query_result_tags(100);
                    std::vector<float> query_result_dists(100);
                    greator::QueryStats* stats = new greator::QueryStats;
                    this->disk_index->cached_beam_search(query.data(), 100, L, query_result_tags.data(), query_result_dists.data(), 2, stats);

                    // Accumulate thetas for different K
                    theta_sums[1]   += query_result_dists[0];
                    theta_sums[5]   += query_result_dists[4];
                    theta_sums[10]  += query_result_dists[9];
                    theta_sums[50]  += query_result_dists[49];
                    theta_sums[100] += query_result_dists[99];
                }
                // Compute averages and store in theta_map
                for (auto& kv : theta_sums) {
                    theta_map[kv.first] = kv.second / n_theta_estimation_queries;
                }
                std::cout << "Training for theta map completed!" << std::endl;
            }


            bool search(const T* query_ptr, uint32_t K, uint32_t L, uint32_t* query_result_tags_ptr, std::vector<T *>& res, uint32_t beamwidth, float* query_result_dists_ptr, greator::QueryStats* stat) {
                // Search in memory index
                this->memory_index->search_with_tags(query_ptr, K, L, query_result_tags_ptr, query_result_dists_ptr, res);
                if (this->isHit(query_ptr, K, L, query_result_dists_ptr)) {
                    return true; // Return true if the query is hit in the memory index
                }
                else {
                    this->disk_index->cached_beam_search(query_ptr, (uint64_t)K, (uint64_t)L, query_result_tags_ptr, query_result_dists_ptr, (uint64_t)beamwidth, stat);
                    std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                    insert_pool->submit(this->memory_index, tags_to_insert, data_path, this->dim);
                    for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                    {
                        std::lock_guard<std::mutex> lock(theta_map_mutex);
                        theta_map[K] = p * query_result_dists_ptr[K - 1] + (1 - p) * theta_map[K];
                    }
                    return false; // Return false if the query is missed in the memory index
                }
            }

            size_t get_number_of_vectors_in_memory_index() const {
                return num_points_in_memory_index.load(std::memory_order_relaxed);
            }
    };
}