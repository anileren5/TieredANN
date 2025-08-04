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
#include <unordered_map>
#include <mutex>
#include <type_traits>
#include <memory>
#include <cstring>
#include <atomic>
#include <future>

namespace tieredann {

    template <typename T, typename TagT = uint32_t>
    class TieredIndex {
        
        private:
            std::unique_ptr<greator::PQFlashIndex<T, TagT>> disk_index;
            
            // Cycling shadow paging: Two memory indices that cycle between active and query-only
            std::unique_ptr<diskann::AbstractIndex> index_a;
            std::unique_ptr<diskann::AbstractIndex> index_b;
            std::atomic<diskann::AbstractIndex*> active_insert_index; // Index for new insertions

            // --- Index parameters ---
            std::string data_path;
            std::string disk_index_prefix;
            size_t dim, aligned_dim;
            size_t num_points;
            size_t memory_index_max_points_per_index; // Capacity per index
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
            bool lazy_theta_updates = true;

            // --- Consolidation parameters ---
            double consolidation_ratio_;

            // --- Cycling shadow paging state ---
            std::atomic<bool> cycling_in_progress{false};
            std::future<void> cycling_future;
            std::mutex cycling_mutex;

            // --- PCA utilities ---
            std::unique_ptr<PCAUtils<T>> pca_utils;


            
            // Helper function to create a memory index with given max points
            std::unique_ptr<diskann::AbstractIndex> create_memory_index(size_t max_points) {
                diskann::IndexWriteParameters memory_index_write_params = diskann::IndexWriteParametersBuilder(memory_L, aligned_dim)
                                                                    .with_alpha(1.2f) // Default alpha
                                                                    .with_num_threads(4) // Default threads
                                                                    .build();

                diskann::IndexSearchParams memory_index_search_params = diskann::IndexSearchParams(memory_L, search_threads);

                diskann::IndexConfig memory_index_config = diskann::IndexConfigBuilder()
                                                            .with_metric(diskann::L2)
                                                            .with_dimension(dim)
                                                            .with_max_points(max_points)
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
                auto index = memory_index_factory.create_instance();
                index->set_start_points_at_random(static_cast<T>(0));
                return index;
            }

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

                // Insert the new vectors into the provided index (which should be the current active one)
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    int ret = index->insert_point(vectors[i], 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                
                // Check if we need to trigger cycling shadow paging AFTER insertion
                // Only trigger if the currently active index is full
                diskann::AbstractIndex* current_active = active_insert_index.load();
                if (index.get() == current_active && index->get_number_of_active_vectors() >= memory_index_max_points_per_index) {
                    // Always cycle when the currently active index is full
                    if (!cycling_in_progress.load()) {
                        // Start cycling shadow paging
                        trigger_cycling_shadow_paging();
                    }
                }
                
                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const size_t dim, uint32_t K = 0, float query_distance = 0.0f) {
                std::vector<std::vector<T>> reconstructed_vectors = this->disk_index->inflate_vectors_by_tags(to_be_inserted);
                size_t successful_inserts = 0;
                
                // Insert the reconstructed vectors into the provided index
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    int ret = index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                
                // Check if we need to trigger cycling shadow paging AFTER insertion
                // Only trigger if the currently active index is full
                diskann::AbstractIndex* current_active = active_insert_index.load();
                if (index.get() == current_active && index->get_number_of_active_vectors() >= memory_index_max_points_per_index) {
                    // Always cycle when the currently active index is full
                    if (!cycling_in_progress.load()) {
                        trigger_cycling_shadow_paging();
                    }
                }
            }

            // Cycling shadow paging: switch to other index and clear the old one
            void trigger_cycling_shadow_paging() {
                std::lock_guard<std::mutex> lock(cycling_mutex);
                
                if (cycling_in_progress.load()) {
                    return; // Already in progress
                }
                
                cycling_in_progress.store(true);
                
                // Start background cycling
                cycling_future = std::async(std::launch::async, [this]() {
                    this->perform_cycling_shadow_paging();
                });
            }

            void perform_cycling_shadow_paging() {
                // Get current active insert index
                diskann::AbstractIndex* current_active = active_insert_index.load();
                bool switching_to_a = (current_active == index_b.get());
                        
                // STEP 1: Replace the target index with a fresh empty one
                // This preserves the current active index's hit points
                if (switching_to_a) {
                    index_a = create_memory_index(memory_index_max_points_per_index);
                    std::cout << "[CYCLING] Replaced index A with fresh empty index" << std::endl;
                } else {
                    index_b = create_memory_index(memory_index_max_points_per_index);
                    std::cout << "[CYCLING] Replaced index B with fresh empty index" << std::endl;
                }
                
                // STEP 2: Switch active insert index to the fresh empty index
                // New insertions go to the fresh index, but we can still hit from the preserved index
                if (switching_to_a) {
                    active_insert_index.store(index_a.get());
                } else {
                    active_insert_index.store(index_b.get());
                }
                
                std::cout << "[CYCLING] Switched active index to " << (switching_to_a ? "A" : "B") << std::endl;
                std::cout << "[CYCLING] Index A vectors: " << index_a->get_number_of_active_vectors() 
                          << ", Index B vectors: " << index_b->get_number_of_active_vectors() << std::endl;
                
                // Cycling complete
                cycling_in_progress.store(false);
                std::cout << "[CYCLING] Cycling complete" << std::endl;
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
                        uint32_t n_async_insert_threads_ = 4,
                        bool lazy_theta_updates_ = true,
                        double consolidation_ratio = 0.2,
                        uint32_t lru_async_threads = 4)
                        : data_path(data_path),
                        disk_index_prefix(disk_index_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor),
                        memory_L(memory_L),
                        disk_L(disk_L),
                        use_regional_theta(use_regional_theta),
                        memory_index_max_points_per_index(memory_index_max_points / 2), // Half for each index
                        n_async_insert_threads(n_async_insert_threads_),
                        lazy_theta_updates(lazy_theta_updates_),
                        consolidation_ratio_(consolidation_ratio),
                        memory_index_delete_params(diskann::IndexWriteParametersBuilder(memory_L, R)
                                                .with_alpha(alpha)
                                                .with_num_threads(consolidate_threads)
                                                .with_filter_list_size(memory_L)
                                                .build())
            {                
                // Read metadata
                diskann::get_bin_metadata(data_path, num_points, dim);
                aligned_dim = ROUND_UP(dim, 8);

                // Build cycling shadow paging memory indices
                index_a = create_memory_index(memory_index_max_points_per_index);
                index_b = create_memory_index(memory_index_max_points_per_index);
                
                // Set index_a as active initially for insertions
                active_insert_index.store(index_a.get());
            
                std::cout << "TieredIndex shadow paging memory indices built successfully!" << std::endl;
                std::cout << "Each index can hold up to " << memory_index_max_points_per_index << " vectors" << std::endl;

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
                    if (lazy_theta_updates) {
                        auto theta_update_task = [this](T* query_ptr, uint32_t K, float query_distance) {
                            this->update_theta(query_ptr, K, query_distance);
                        };
                        insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(n_async_insert_threads, task, theta_update_task);
                    } else {
                        insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                    }
                } else {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_sync(index, to_be_inserted, data_path, dim, K, query_distance);
                    };
                    if (lazy_theta_updates) {
                        auto theta_update_task = [this](T* query_ptr, uint32_t K, float query_distance) {
                            this->update_theta(query_ptr, K, query_distance);
                        };
                        insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(n_async_insert_threads, task, theta_update_task);
                    } else {
                        insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(n_async_insert_threads, task);
                    }
                }

                std::cout << "TieredIndex built successfully with shadow paging!" << std::endl;

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
                // Get current active insert index (atomic read)
                diskann::AbstractIndex* current_active = active_insert_index.load();
                
                // Always search in both indices
                bool is_hit = false;
                
                // Try current active index first
                if (current_active->get_number_of_active_vectors() > 0) {
                    current_active->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                    is_hit = this->isHit(query_ptr, K, memory_L, query_result_dists_ptr);
                }
                
                // If miss in current active index, try the other index
                if (!is_hit) {
                    diskann::AbstractIndex* other_index = (current_active == index_a.get()) ? index_b.get() : index_a.get();
                    if (other_index->get_number_of_active_vectors() > 0) {
                        other_index->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                        is_hit = this->isHit(query_ptr, K, memory_L, query_result_dists_ptr);
                    }
                }
                
                if (is_hit) {
                    return true; // Return true if the query is hit in the memory index
                }
                else {
                    
                    this->disk_index->cached_beam_search(query_ptr, (uint64_t)K, (uint64_t)disk_L, query_result_tags_ptr, query_result_dists_ptr, (uint64_t)beamwidth, stat);
                    std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                    
                    if (lazy_theta_updates) {
                        // Copy query pointer for async insertion and theta update
                        T* query_copy = nullptr;
                        diskann::alloc_aligned((void**)&query_copy, this->aligned_dim * sizeof(T), 8 * sizeof(T));
                        std::memcpy(query_copy, query_ptr, this->aligned_dim * sizeof(T));
                        
                        // Submit to insert pool with current active index
                        if (current_active == index_a.get()) {
                            insert_pool->submit(index_a, tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1], query_copy);
                        } else {
                            insert_pool->submit(index_b, tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1], query_copy);
                        }
                    } else {
                        // Immediate theta update in main thread
                        update_theta(query_ptr, K, query_result_dists_ptr[K - 1]);
                        if (current_active == index_a.get()) {
                            insert_pool->submit(index_a, tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                        } else {
                            insert_pool->submit(index_b, tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                        }
                    }
                    
                    for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                    return false; // Return false if the query is missed in the memory index
                }
            }

            size_t get_number_of_vectors_in_memory_index() const {
                return index_a->get_number_of_active_vectors() + index_b->get_number_of_active_vectors();
            }

            size_t get_number_of_max_points_in_memory_index() const {
                return memory_index_max_points_per_index * 2; // Total capacity across both indices
            }

            // New methods for cycling shadow paging status
            bool is_cycling_in_progress() const {
                return cycling_in_progress.load();
            }

            size_t get_index_a_vector_count() const {
                return index_a->get_number_of_active_vectors();
            }

            size_t get_index_b_vector_count() const {
                return index_b->get_number_of_active_vectors();
            }

            bool is_index_a_active() const {
                return active_insert_index.load() == index_a.get();
            }

            size_t get_number_of_active_pca_regions() const {
                if (use_regional_theta && pca_utils) {
                    return pca_utils->get_number_of_active_regions();
                }
                return 0;
            }


    };
}