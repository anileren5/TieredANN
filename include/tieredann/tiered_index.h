// Greator (Disk index) headers
#include "greator/pq_flash_index.h"
#include "greator/aux_utils.h"
#include "greator/linux_aligned_file_reader.h"
#include "greator/utils.h"
#include "greator/ctpl_stl.h"

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
#include <vector>
#include <algorithm>

namespace tieredann {

    template <typename T, typename TagT = uint32_t>
    class TieredIndex {
        
        private:
            std::unique_ptr<greator::PQFlashIndex<T, TagT>> disk_index;
            
            // Cycling shadow paging: n memory indices that cycle between active and query-only
            std::vector<std::unique_ptr<diskann::AbstractIndex>> memory_indices;
            std::atomic<size_t> active_insert_index_id; // Index ID for new insertions

            // --- Index parameters ---
            std::string data_path;
            std::string disk_index_prefix;
            size_t dim, aligned_dim;
            size_t num_points;
            size_t memory_index_max_points_per_index; // Capacity per index
            size_t number_of_mini_indexes; // Number of mini indexes
            uint32_t search_threads;
            bool use_reconstructed_vectors;
            std::unique_ptr<tieredann::InsertThreadPool<T, TagT>> insert_pool;
            std::unordered_map<uint32_t, double> theta_map;
            std::mutex theta_map_mutex;
            double p, deviation_factor;
            uint32_t memory_L; 
            uint32_t disk_L;    
            bool use_regional_theta = true;

            uint32_t n_async_insert_threads = 4;
            bool lazy_theta_updates = true;
            bool search_mini_indexes_in_parallel = false; // Control parallel vs sequential search
            size_t max_search_threads = 32; // Maximum threads for parallel search (should be > query processing threads)

            // --- Cycling shadow paging state ---
            std::atomic<bool> cycling_in_progress{false};
            std::future<void> cycling_future;
            std::mutex cycling_mutex;

            // --- PCA utilities ---
            std::unique_ptr<PCAUtils<T>> pca_utils;

            // --- Thread pool for parallel search ---
            std::unique_ptr<ctpl::thread_pool> search_thread_pool;
            std::mutex search_pool_mutex;


            
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
                size_t current_active_id = active_insert_index_id.load();
                if (index.get() == memory_indices[current_active_id].get() && index->get_number_of_active_vectors() >= memory_index_max_points_per_index) {
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
                size_t current_active_id = active_insert_index_id.load();
                if (index.get() == memory_indices[current_active_id].get() && index->get_number_of_active_vectors() >= memory_index_max_points_per_index) {
                    // Always cycle when the currently active index is full
                    if (!cycling_in_progress.load()) {
                        trigger_cycling_shadow_paging();
                    }
                }
            }

            // Cycling shadow paging: switch to next index and clear the old one
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
                // Get current active insert index ID
                size_t current_active_id = active_insert_index_id.load();
                size_t next_active_id = (current_active_id + 1) % number_of_mini_indexes;
                        
                // STEP 1: Replace the target index with a fresh empty one
                // This preserves the current active index's hit points
                memory_indices[next_active_id] = create_memory_index(memory_index_max_points_per_index);
                
                // STEP 2: Switch active insert index to the fresh empty index
                // New insertions go to the fresh index, but we can still hit from the preserved indices
                active_insert_index_id.store(next_active_id);
                                                
                // Cycling complete
                cycling_in_progress.store(false);
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

            // Helper function to search a single memory index
            bool search_single_index(size_t index_id, const T* query_ptr, uint32_t K, uint32_t L, 
                                   uint32_t* query_result_tags_ptr, std::vector<T*>& res, 
                                   float* query_result_dists_ptr) {
                if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                    memory_indices[index_id]->search_with_tags(query_ptr, K, L, query_result_tags_ptr, query_result_dists_ptr, res);
                    return this->isHit(query_ptr, K, L, query_result_dists_ptr);
                }
                return false;
            }

            // Parallel search across all memory indices using thread pool
            bool parallel_search_memory_indices(const T* query_ptr, uint32_t K, uint32_t L,
                                              uint32_t* query_result_tags_ptr, std::vector<T*>& res,
                                              float* query_result_dists_ptr) {
                if (number_of_mini_indexes == 1) {
                    // Single index case - no need for parallelization
                    return search_single_index(0, query_ptr, K, L, query_result_tags_ptr, res, query_result_dists_ptr);
                }

                // Ensure thread pool is initialized
                {
                    std::lock_guard<std::mutex> lock(search_pool_mutex);
                    if (!search_thread_pool) {
                        // Create thread pool with enough threads to handle concurrent searches
                        // The pool should be sized to handle the maximum number of concurrent searches
                        // For n memory indices, we need at least n threads to search them all in parallel
                        // The max_search_threads parameter provides an upper limit
                        size_t pool_size = std::min(max_search_threads, std::max(number_of_mini_indexes, static_cast<size_t>(8)));
                        search_thread_pool = std::make_unique<ctpl::thread_pool>(static_cast<int>(pool_size));
                    }
                }

                // Prepare results for each index
                std::vector<std::vector<uint32_t>> all_tags(number_of_mini_indexes);
                std::vector<std::vector<float>> all_dists(number_of_mini_indexes);
                std::vector<std::vector<T*>> all_res(number_of_mini_indexes);
                std::vector<bool> hit_results(number_of_mini_indexes, false);
                std::vector<std::mutex> result_mutexes(number_of_mini_indexes);
                std::vector<std::future<void>> futures;

                // Lambda function for parallel search
                auto search_worker = [&](int thread_id, size_t index_id) {
                    if (memory_indices[index_id]->get_number_of_active_vectors() > 0) {
                        // Allocate temporary storage for this thread
                        std::vector<uint32_t> temp_tags(K);
                        std::vector<float> temp_dists(K);
                        std::vector<T*> temp_res;

                        // Search in this index
                        memory_indices[index_id]->search_with_tags(query_ptr, K, L, temp_tags.data(), temp_dists.data(), temp_res);
                        
                        // Check if this is a hit
                        bool is_hit = this->isHit(query_ptr, K, L, temp_dists.data());
                        
                        // Store results atomically
                        {
                            std::lock_guard<std::mutex> lock(result_mutexes[index_id]);
                            all_tags[index_id] = std::move(temp_tags);
                            all_dists[index_id] = std::move(temp_dists);
                            all_res[index_id] = std::move(temp_res);
                            hit_results[index_id] = is_hit;
                        }
                    }
                };

                // Submit tasks to thread pool
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    futures.push_back(search_thread_pool->push(search_worker, i));
                }

                // Wait for all tasks to complete
                for (auto& future : futures) {
                    future.get();
                }

                // Find the first hit and use its results
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    if (hit_results[i]) {
                        // Copy the winning results to the output
                        std::copy(all_tags[i].begin(), all_tags[i].end(), query_result_tags_ptr);
                        std::copy(all_dists[i].begin(), all_dists[i].end(), query_result_dists_ptr);
                        res = std::move(all_res[i]);
                        return true;
                    }
                }

                return false; // No hits found
            }

        public:
            template <typename... Args>
            TieredIndex(const std::string& data_path,
                        const std::string& disk_index_prefix,
                        uint32_t R, uint32_t memory_L, uint32_t disk_L,
                        uint32_t B, uint32_t M,
                        float alpha,
                        uint32_t build_threads,
                        uint32_t search_threads,
                        int disk_index_already_built,
                        bool use_reconstructed_vectors,
                        double p,
                        double deviation_factor, 
                        size_t memory_index_max_points,
                        bool use_regional_theta = true,
                        size_t pca_dim = 16,
                        size_t buckets_per_dim = 4,
                        uint32_t n_async_insert_threads_ = 4,
                        bool lazy_theta_updates_ = true,
                        size_t number_of_mini_indexes_ = 2,
                        bool search_mini_indexes_in_parallel_ = false,
                        size_t max_search_threads_ = 32)
                        : data_path(data_path),
                        disk_index_prefix(disk_index_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor),
                        memory_L(memory_L),
                        disk_L(disk_L),
                        use_regional_theta(use_regional_theta),
                        number_of_mini_indexes(number_of_mini_indexes_),
                        memory_index_max_points_per_index(memory_index_max_points / number_of_mini_indexes_), // Equal capacity per index
                        n_async_insert_threads(n_async_insert_threads_),
                        lazy_theta_updates(lazy_theta_updates_),
                        search_mini_indexes_in_parallel(search_mini_indexes_in_parallel_),
                        max_search_threads(max_search_threads_)
            {                
                // Read metadata
                diskann::get_bin_metadata(data_path, num_points, dim);
                aligned_dim = ROUND_UP(dim, 8);

                // Build cycling shadow paging memory indices
                memory_indices.reserve(number_of_mini_indexes);
                for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                    memory_indices.push_back(create_memory_index(memory_index_max_points_per_index));
                }
                
                // Set index 0 as active initially for insertions
                active_insert_index_id.store(0);
            
                std::cout << "TieredIndex shadow paging memory indices built successfully!" << std::endl;
                std::cout << "Created " << number_of_mini_indexes << " indices, each can hold up to " << memory_index_max_points_per_index << " vectors" << std::endl;
                if (search_mini_indexes_in_parallel) {
                    std::cout << "Parallel search enabled with max " << max_search_threads << " threads" << std::endl;
                }

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
                // Get current active insert index ID (atomic read)
                size_t current_active_id = active_insert_index_id.load();
                
                // Search all memory indices (parallel or sequential based on configuration)
                bool is_hit = false;
                if (search_mini_indexes_in_parallel && number_of_mini_indexes > 1) {
                    is_hit = parallel_search_memory_indices(query_ptr, K, memory_L, query_result_tags_ptr, res, query_result_dists_ptr);
                } else {
                    // Sequential search (original logic)
                    // Try current active index first
                    if (memory_indices[current_active_id]->get_number_of_active_vectors() > 0) {
                        memory_indices[current_active_id]->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                        is_hit = this->isHit(query_ptr, K, memory_L, query_result_dists_ptr);
                    }
                    
                    // If miss in current active index, try all other indices
                    if (!is_hit) {
                        for (size_t i = 0; i < number_of_mini_indexes; ++i) {
                            if (i != current_active_id && memory_indices[i]->get_number_of_active_vectors() > 0) {
                                memory_indices[i]->search_with_tags(query_ptr, K, memory_L, query_result_tags_ptr, query_result_dists_ptr, res);
                                is_hit = this->isHit(query_ptr, K, memory_L, query_result_dists_ptr);
                                if (is_hit) break; // Found a hit, no need to search further
                            }
                        }
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
                        insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1], query_copy);
                    } else {
                        // Immediate theta update in main thread
                        update_theta(query_ptr, K, query_result_dists_ptr[K - 1]);
                        insert_pool->submit(memory_indices[current_active_id], tags_to_insert, data_path, this->dim, K, query_result_dists_ptr[K - 1]);
                    }
                    
                    for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                    return false; // Return false if the query is missed in the memory index
                }
            }

            size_t get_number_of_vectors_in_memory_index() const {
                size_t total = 0;
                for (const auto& index : memory_indices) {
                    total += index->get_number_of_active_vectors();
                }
                return total;
            }

            size_t get_number_of_max_points_in_memory_index() const {
                return memory_index_max_points_per_index * number_of_mini_indexes; // Total capacity across all indices
            }

            // New methods for cycling shadow paging status
            bool is_cycling_in_progress() const {
                return cycling_in_progress.load();
            }

            size_t get_index_vector_count(size_t index_id) const {
                if (index_id < number_of_mini_indexes) {
                    return memory_indices[index_id]->get_number_of_active_vectors();
                }
                return 0;
            }

            size_t get_active_index_id() const {
                return active_insert_index_id.load();
            }

            size_t get_number_of_mini_indexes() const {
                return number_of_mini_indexes;
            }

            bool is_parallel_search_enabled() const {
                return search_mini_indexes_in_parallel;
            }

            size_t get_max_search_threads() const {
                return max_search_threads;
            }

            size_t get_number_of_active_pca_regions() const {
                if (use_regional_theta && pca_utils) {
                    return pca_utils->get_number_of_active_regions();
                }
                return 0;
            }


    };
}