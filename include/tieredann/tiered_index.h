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
#include <chrono>
#include "tieredann/insert_thread_pool.h"

namespace tieredann {

    template <typename T, typename TagT = uint32_t>
    class TieredIndex {
        
        private:
            std::unique_ptr<greator::PQFlashIndex<float>> disk_index;
            std::unique_ptr<diskann::AbstractIndex> memory_index;
            std::string data_path;
            std::string disk_index_prefix;
            size_t dim, aligned_dim;
            size_t num_points;
            uint32_t search_threads;
            double hit_rate;
            bool use_reconstructed_vectors;
            std::unique_ptr<tieredann::InsertThreadPool<T, TagT>> insert_pool;

            void memory_index_insert_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim) {
                // Allocate buffer for all vectors
                std::vector<T*> vectors;
                vectors.reserve(to_be_inserted.size());
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    T* vector = nullptr;
                    diskann::alloc_aligned((void**)&vector, dim * sizeof(T), 8 * sizeof(T));
                    diskann::load_vector_by_index(data_path, vector, dim, to_be_inserted[i]);
                    vectors.push_back(vector);
                }
                // Insert all vectors
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    index->insert_point(vectors[i], 1 + to_be_inserted[i]);
                }
                // Free all allocated vectors
                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const size_t dim) {
                // Get reconstructed PQ vectors for the tags
                std::vector<std::vector<T>> reconstructed_vectors = this->disk_index->inflate_vectors_by_tags(to_be_inserted);
                
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                }
            }

            bool isHit(double hit_rate) {
                return static_cast<double>(rand()) / RAND_MAX < hit_rate;
            }

        
        public:
            TieredIndex(const std::string& data_path,
                        const std::string& disk_index_prefix,
                        uint32_t R, uint32_t L,
                        uint32_t B, uint32_t M,
                        float alpha,
                        double hit_rate,
                        uint32_t consolidate_threads,
                        uint32_t build_threads,
                        uint32_t search_threads,
                        int disk_index_already_built,
                        bool use_reconstructed_vectors):
                        data_path(data_path),
                        disk_index_prefix(disk_index_prefix),
                        search_threads(search_threads),
                        hit_rate(hit_rate),
                        use_reconstructed_vectors(use_reconstructed_vectors)
            {
                // Set random seed
                srand(42);
                
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
            }


            void search(const T* query_ptr, uint32_t K, uint32_t L, uint32_t* query_result_tags_ptr, std::vector<T *>& res, uint32_t beamwidth, float* query_result_dists_ptr, greator::QueryStats* stat) {
                if (this->isHit(hit_rate)) {
                    auto start = std::chrono::high_resolution_clock::now();
                    this->memory_index->search_with_tags(query_ptr, K, L, query_result_tags_ptr, query_result_dists_ptr, res);
                } 
                else {
                    this->disk_index->cached_beam_search(query_ptr, (uint64_t)K, (uint64_t)L, query_result_tags_ptr, query_result_dists_ptr, (uint64_t)beamwidth, stat);
                    std::vector<uint32_t> tags_to_insert(query_result_tags_ptr, query_result_tags_ptr + K);
                    insert_pool->submit(this->memory_index, tags_to_insert, data_path, this->dim);
                    for (size_t j = 0; j < K; j++) query_result_tags_ptr[j] += 1;
                }
            }

            // Get reconstructed PQ vectors for search results
            std::vector<std::vector<T>> get_reconstructed_pq_vectors(const uint32_t* query_result_tags_ptr, uint32_t K) {
                std::vector<TagT> tags(query_result_tags_ptr, query_result_tags_ptr + K);
                return this->disk_index->inflate_vectors_by_tags(tags);
            }

            // Manually insert reconstructed PQ vectors for specific tags
            void insert_reconstructed_vectors(const std::vector<TagT>& tags) {
                memory_index_insert_reconstructed_sync(memory_index, tags, dim);
            }

            // Check if using reconstructed vectors for insertion
            bool is_using_reconstructed_vectors() const {
                return use_reconstructed_vectors;
            }

            size_t get_number_of_vectors_in_memory_index() const {
                return this->memory_index->template get_number_of_active_vectors<TagT>();
            }
    };
}