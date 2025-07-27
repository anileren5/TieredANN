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
#include "tieredann/pca_utils.h"
#include <set>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <type_traits>

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
            size_t memory_index_max_points;
            uint32_t search_threads;
            bool use_reconstructed_vectors;
            std::unique_ptr<tieredann::InsertThreadPool<T, TagT>> insert_pool;
            std::unordered_map<uint32_t, double> theta_map;
            std::mutex theta_map_mutex;
            double p, deviation_factor;
            uint32_t memory_L; 
            uint32_t disk_L;    
            bool use_regional_theta = true;
            diskann::IndexWriteParameters memory_index_delete_params;
            uint32_t n_async_insert_threads = 4;

            // --- PCA utilities ---
            std::unique_ptr<PCAUtils<T>> pca_utils;

            void memory_index_insert_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K = 0, float query_distance = 0.0f) {
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
                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const size_t dim, uint32_t K = 0, float query_distance = 0.0f) {
                std::vector<std::vector<T>> reconstructed_vectors = this->disk_index->inflate_vectors_by_tags(to_be_inserted);
                size_t successful_inserts = 0;
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    int ret = index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
            }

            bool isHit(const T* query_ptr, uint32_t K, uint32_t L, const float* distances) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (use_regional_theta) {
                    return pca_utils->isHit(query_ptr, K, L, distances, this->get_number_of_vectors_in_memory_index(), deviation_factor);
                } else {
                    if (this->get_number_of_vectors_in_memory_index() < K){
                        return false;
                    }
                    else if (distances[K - 1] > (1 + deviation_factor)*theta_map[K]) {
                        return false;
                    }
                    return true;
                }
            }

            void update_theta(const T* query_ptr, uint32_t K, float query_distance) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (use_regional_theta) {
                    pca_utils->update_theta(query_ptr, K, query_distance, p);
                } else {
                    theta_map[K] = p * query_distance + (1 - p) * theta_map[K];
                }
            }

        
        public:
            template <typename... Args>
            TieredIndex(const std::string& data_path,
                        const std::string& disk_index_prefix,
                        uint32_t R, uint32_t memory_L, uint32_t disk_L,
                        uint32_t B, uint32_t M,
                        float alpha,
                        uint32_t consolidate_threads,
                        uint32_t build_threads,
                        uint32_t search_threads,
                        int disk_index_already_built,
                        bool use_reconstructed_vectors,
                        double p,
                        double deviation_factor, 
                        uint32_t n_theta_estimation_queries,
                        size_t memory_index_max_points,
                        bool use_regional_theta = true,
                        size_t pca_dim = 16,
                        size_t buckets_per_dim = 4,
                        uint32_t n_async_insert_threads_ = 4)
                        : data_path(data_path),
                        disk_index_prefix(disk_index_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor),
                        memory_L(memory_L),
                        disk_L(disk_L),
                        use_regional_theta(use_regional_theta),
                        memory_index_max_points(memory_index_max_points),
                        n_async_insert_threads(n_async_insert_threads_),
                        memory_index_delete_params(diskann::IndexWriteParametersBuilder(memory_L, R)
                                                .with_alpha(alpha)
                                                .with_num_threads(consolidate_threads)
                                                .with_filter_list_size(memory_L)
                                                .build())
            {                
                // Read metadata
                diskann::get_bin_metadata(data_path, num_points, dim);
                aligned_dim = ROUND_UP(dim, 8);

                // memory_index_max_points is now required and must be set by caller

                // Build memory index
                diskann::IndexWriteParameters memory_index_write_params = diskann::IndexWriteParametersBuilder(memory_L, R)
                                                                    .with_alpha(alpha)
                                                                    .with_num_threads(consolidate_threads)
                                                                    .build();

                diskann::IndexSearchParams memory_index_search_params = diskann::IndexSearchParams(memory_L, search_threads);

                diskann::IndexConfig memory_index_config = diskann::IndexConfigBuilder()
                                                            .with_metric(diskann::L2)
                                                            .with_dimension(dim)
                                                            .with_max_points(memory_index_max_points)
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
                    std::string disk_index_params = std::to_string(R) + " " + std::to_string(disk_L) + " " + std::to_string(B) + " " + std::to_string(M) + " " + std::to_string(build_threads);
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
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_reconstructed_sync(index, to_be_inserted, dim, K, query_distance);
                    };
                    insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                } else {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_sync(index, to_be_inserted, data_path, dim, K, query_distance);
                    };
                    insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                }

                std::cout << "TieredIndex built successfully!" << std::endl;

                // PCA is constructed at construction time using Eigen. Eigen is required.
                if (use_regional_theta) {
                    pca_utils = std::make_unique<PCAUtils<T>>(dim, pca_dim, buckets_per_dim, disk_index_prefix);
                    bool loaded = false;
                    if constexpr (std::is_floating_point<T>::value) {
                        loaded = pca_utils->load_pca_from_file(false);
                    } else {
                        loaded = pca_utils->load_pca_from_file(true);
                    }
                    if (loaded) {
                        std::cout << "[TieredIndex] Loaded PCA from file: " << pca_utils->get_pca_filename_for_logging() << std::endl;
                    } else {
                        std::cout << "[TieredIndex] No PCA file found or mismatch, running PCA..." << std::endl;
                        T* data = nullptr;
                        diskann::alloc_aligned((void**)&data, num_points * aligned_dim * sizeof(T), 8 * sizeof(T));
                        diskann::load_aligned_bin<T>(data_path, data, num_points, dim, aligned_dim);
                        pca_utils->construct_pca_from_data(data, num_points, aligned_dim, disk_index_prefix);
                        diskann::aligned_free(data);
                    }
                } else {
                    std::cout << "[TieredIndex] Skipping PCA construction (use_regional_theta is false)." << std::endl;
                }

                if (!use_regional_theta) {
                    // Initialize global theta_map for K=1,5,10,100
                    theta_map[1] = std::numeric_limits<double>::min();
                    theta_map[5] = std::numeric_limits<double>::min();
                    theta_map[10] = std::numeric_limits<double>::min();
                    theta_map[100] = std::numeric_limits<double>::min();
                }
            }


            bool search(const T* query_ptr, uint32_t K, uint32_t L, uint32_t* query_result_tags_ptr, std::vector<T *>& res, uint32_t beamwidth, float* query_result_dists_ptr, greator::QueryStats* stat) {
                // Search in memory index
                this->memory_index->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                if (this->isHit(query_ptr, K, memory_L, query_result_dists_ptr)) {
                    return true; // Return true if the query is hit in the memory index
                }
                else {
                    this->disk_index->cached_beam_search(query_ptr, (uint64_t)K, (uint64_t)disk_L, query_result_tags_ptr, query_result_dists_ptr, (uint64_t)beamwidth, stat);
                    std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                    insert_pool->submit(this->memory_index, tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                    for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                    // Always update theta in main thread
                    update_theta(query_ptr, K, query_result_dists_ptr[K - 1]);
                    return false; // Return false if the query is missed in the memory index
                }
            }

            size_t get_number_of_vectors_in_memory_index() const {
                return this->memory_index->get_number_of_active_vectors();
            }

            size_t get_number_of_lazy_deleted_vectors_in_memory_index() const {
                return this->memory_index->get_number_of_lazy_deleted_points();
            }

            size_t get_number_of_max_points_in_memory_index() const {
                return this->memory_index->get_max_points();
            }
    };
}