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
#include <array>

// Headers for PCA and region-aware theta map
#include <Eigen/Dense>
#include <type_traits>
#include <fstream>
#include <filesystem>

namespace tieredann {

    // Hash for std::vector<uint8_t>
    struct ArrayHash {
        std::size_t operator()(const std::vector<uint8_t>& arr) const {
            std::size_t h = 0;
            for (auto v : arr) h = h * 31 + v;
            return h;
        }
    };

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
            std::atomic<size_t> num_points_in_memory_index{0};
            uint32_t memory_L;  // L value for memory index search
            uint32_t disk_L;    // L value for disk index search
            bool use_regional_theta = true;
            
            // --- PCA and region-aware theta map additions ---
            size_t PCA_DIM; // Project to PCA_DIM dimensions
            size_t BUCKETS_PER_DIM; // Number of buckets per dim
            using RegionKey = std::vector<uint8_t>; // Dynamic size
            // Map: region -> (K -> theta)
            std::unordered_map<RegionKey, std::unordered_map<uint32_t, double>, ArrayHash> region_theta_map;
            // PCA projection matrix and min/max for bucketing
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pca_components; // [dim, PCA_DIM]
            Eigen::Matrix<T, 1, Eigen::Dynamic> pca_mean; // [1, dim]
            std::vector<T> pca_min, pca_max; // min/max for each PCA dim
            // --- PCA float storage for int8/uint8 types ---
            // Only used if T is not floating point
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> pca_components_float;
            Eigen::Matrix<float, 1, Eigen::Dynamic> pca_mean_float;
            std::vector<float> pca_min_float, pca_max_float;
            
            // --- PCA persistence helpers ---
            std::string get_pca_filename() const {
                return disk_index_prefix + ".pca.bin";
            }
            bool file_exists(const std::string& filename) const {
                return std::filesystem::exists(filename);
            }
            // Save PCA data to file
            void save_pca_to_file(bool is_float) {
                std::ofstream ofs(get_pca_filename(), std::ios::binary);
                if (!ofs) return;
                ofs.write((char*)&dim, sizeof(dim));
                ofs.write((char*)&PCA_DIM, sizeof(PCA_DIM));
                ofs.write((char*)&BUCKETS_PER_DIM, sizeof(BUCKETS_PER_DIM));
                if (is_float) {
                    // Save pca_mean
                    ofs.write(reinterpret_cast<const char*>(pca_mean_float.data()), sizeof(float) * dim);
                    // Save pca_components (row-major)
                    ofs.write(reinterpret_cast<const char*>(pca_components_float.data()), sizeof(float) * dim * PCA_DIM);
                    // Save pca_min, pca_max
                    ofs.write(reinterpret_cast<const char*>(pca_min_float.data()), sizeof(float) * PCA_DIM);
                    ofs.write(reinterpret_cast<const char*>(pca_max_float.data()), sizeof(float) * PCA_DIM);
                } else {
                    // Save pca_mean
                    ofs.write(reinterpret_cast<const char*>(pca_mean.data()), sizeof(T) * dim);
                    // Save pca_components (row-major)
                    ofs.write(reinterpret_cast<const char*>(pca_components.data()), sizeof(T) * dim * PCA_DIM);
                    // Save pca_min, pca_max
                    ofs.write(reinterpret_cast<const char*>(pca_min.data()), sizeof(T) * PCA_DIM);
                    ofs.write(reinterpret_cast<const char*>(pca_max.data()), sizeof(T) * PCA_DIM);
                }
            }
            // Load PCA data from file
            bool load_pca_from_file(bool is_float) {
                std::ifstream ifs(get_pca_filename(), std::ios::binary);
                if (!ifs) return false;
                size_t file_dim, file_pca_dim, file_buckets_per_dim;
                ifs.read((char*)&file_dim, sizeof(file_dim));
                ifs.read((char*)&file_pca_dim, sizeof(file_pca_dim));
                ifs.read((char*)&file_buckets_per_dim, sizeof(file_buckets_per_dim));
                if (file_dim != dim || file_pca_dim != PCA_DIM || file_buckets_per_dim != BUCKETS_PER_DIM) return false;
                if (is_float) {
                    pca_mean_float.resize(dim);
                    ifs.read(reinterpret_cast<char*>(pca_mean_float.data()), sizeof(float) * dim);
                    pca_components_float.resize(dim, PCA_DIM);
                    ifs.read(reinterpret_cast<char*>(pca_components_float.data()), sizeof(float) * dim * PCA_DIM);
                    pca_min_float.resize(PCA_DIM);
                    pca_max_float.resize(PCA_DIM);
                    ifs.read(reinterpret_cast<char*>(pca_min_float.data()), sizeof(float) * PCA_DIM);
                    ifs.read(reinterpret_cast<char*>(pca_max_float.data()), sizeof(float) * PCA_DIM);
                } else {
                    pca_mean.resize(dim);
                    ifs.read(reinterpret_cast<char*>(pca_mean.data()), sizeof(T) * dim);
                    pca_components.resize(dim, PCA_DIM);
                    ifs.read(reinterpret_cast<char*>(pca_components.data()), sizeof(T) * dim * PCA_DIM);
                    pca_min.resize(PCA_DIM);
                    pca_max.resize(PCA_DIM);
                    ifs.read(reinterpret_cast<char*>(pca_min.data()), sizeof(T) * PCA_DIM);
                    ifs.read(reinterpret_cast<char*>(pca_max.data()), sizeof(T) * PCA_DIM);
                }
                return true;
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
                for (size_t i = 0; i < to_be_inserted.size(); ++i) {
                    int ret = index->insert_point(vectors[i], 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                num_points_in_memory_index.fetch_add(successful_inserts, std::memory_order_relaxed);
                for (auto v : vectors) {
                    diskann::aligned_free(v);
                }
                // No theta update here; always done in main thread (eager theta update)
            }

            void memory_index_insert_reconstructed_sync(std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const size_t dim, uint32_t K = 0, float query_distance = 0.0f) {
                std::vector<std::vector<T>> reconstructed_vectors = this->disk_index->inflate_vectors_by_tags(to_be_inserted);
                size_t successful_inserts = 0;
                for (size_t i = 0; i < to_be_inserted.size(); i++) {
                    const auto& reconstructed_vec = reconstructed_vectors[i];
                    int ret = index->insert_point(reconstructed_vec.data(), 1 + to_be_inserted[i]);
                    if (ret == 0) ++successful_inserts;
                }
                num_points_in_memory_index.fetch_add(successful_inserts, std::memory_order_relaxed);
                // No theta update here; always done in main thread (eager theta update)
            }

            // --- Helper: Project vector to PCA and compute region key ---
            RegionKey compute_region_key(const T* vec) {
                RegionKey key(PCA_DIM);
                if constexpr (std::is_floating_point<T>::value) {
                    Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> v(vec, dim);
                    Eigen::Matrix<T, 1, Eigen::Dynamic> proj = (v - pca_mean) * pca_components.leftCols(PCA_DIM);
                    for (size_t i = 0; i < PCA_DIM; ++i) {
                        T val = proj(0, i);
                        T minv = pca_min[i], maxv = pca_max[i];
                        if (maxv == minv) key[i] = 0;
                        else {
                            T norm = (val - minv) / (maxv - minv);
                            size_t bucket = std::min<size_t>(BUCKETS_PER_DIM - 1, static_cast<size_t>(norm * BUCKETS_PER_DIM));
                            key[i] = static_cast<uint8_t>(bucket);
                        }
                    }
                } else {
                    std::vector<float> float_vec(dim);
                    for (size_t j = 0; j < dim; ++j) float_vec[j] = static_cast<float>(vec[j]);
                    Eigen::Map<const Eigen::Matrix<float, 1, Eigen::Dynamic>> v(float_vec.data(), dim);
                    Eigen::Matrix<float, 1, Eigen::Dynamic> proj = (v - pca_mean_float) * pca_components_float.leftCols(PCA_DIM);
                    for (size_t i = 0; i < PCA_DIM; ++i) {
                        float val = proj(0, i);
                        float minv = pca_min_float[i], maxv = pca_max_float[i];
                        if (maxv == minv) key[i] = 0;
                        else {
                            float norm = (val - minv) / (maxv - minv);
                            size_t bucket = std::min<size_t>(BUCKETS_PER_DIM - 1, static_cast<size_t>(norm * BUCKETS_PER_DIM));
                            key[i] = static_cast<uint8_t>(bucket);
                        }
                    }
                }
                return key;
            }

            // --- Helper: Lazy initialize region theta map ---
            void lazy_init_region(const RegionKey& key) {
                if (use_regional_theta) {
                    if (region_theta_map.find(key) == region_theta_map.end()) {
                        // Default values as before
                        region_theta_map[key][1] = 0;
                        region_theta_map[key][5] = 0;
                        region_theta_map[key][10] = 0;
                        region_theta_map[key][100] = 0;
                    }
                }
            }

            // --- Modified isHit to use region-aware theta map ---
            bool isHit(const T* query_ptr, uint32_t K, uint32_t L, const float* distances) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (use_regional_theta) {
                    RegionKey region = compute_region_key(query_ptr);
                    lazy_init_region(region);
                    if (this->get_number_of_vectors_in_memory_index() < K){
                        return false;
                    }
                    else if (distances[K - 1] > (1 + deviation_factor)*region_theta_map[region][K]) {
                        return false;
                    }
                    return true;
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

            // --- Modified theta update logic to use region-aware map ---
            void update_theta(const T* query_ptr, uint32_t K, float query_distance) {
                std::lock_guard<std::mutex> lock(theta_map_mutex);
                if (use_regional_theta) {
                    RegionKey region = compute_region_key(query_ptr);
                    lazy_init_region(region);
                    region_theta_map[region][K] = p * query_distance + (1 - p) * region_theta_map[region][K];
                } else {
                    theta_map[K] = p * query_distance + (1 - p) * theta_map[K];
                }
            }

        
        public:
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
                        size_t buckets_per_dim = 4)
                        : data_path(data_path),
                        disk_index_prefix(disk_index_prefix),
                        search_threads(search_threads),
                        use_reconstructed_vectors(use_reconstructed_vectors),
                        p(p),
                        deviation_factor(deviation_factor),
                        memory_L(memory_L),
                        disk_L(disk_L),
                        use_regional_theta(use_regional_theta),
                        PCA_DIM(pca_dim),
                        BUCKETS_PER_DIM(buckets_per_dim),
                        memory_index_max_points(memory_index_max_points)
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
                    insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(4, task);
                } else {
                    auto task = [this](std::unique_ptr<diskann::AbstractIndex>& index, std::vector<TagT> to_be_inserted, const std::string& data_path, const size_t dim, uint32_t K, float query_distance) {
                        this->memory_index_insert_sync(index, to_be_inserted, data_path, dim, K, query_distance);
                    };
                    insert_pool = std::make_unique<tieredann::InsertThreadPool<T, TagT>>(4, task);
                }

                std::cout << "TieredIndex built successfully!" << std::endl;

                // --- PCA construction (Eigen required) ---
                // PCA is constructed at construction time using Eigen. Eigen is required.
                if (use_regional_theta) {
                    bool loaded = false;
                    if constexpr (std::is_floating_point<T>::value) {
                        loaded = load_pca_from_file(false);
                    } else {
                        loaded = load_pca_from_file(true);
                    }
                    if (loaded) {
                        std::cout << "[TieredIndex] Loaded PCA from file: " << get_pca_filename() << std::endl;
                    } else {
                        std::cout << "[TieredIndex] No PCA file found or mismatch, running PCA..." << std::endl;
                        T* data = nullptr;
                        diskann::alloc_aligned((void**)&data, num_points * aligned_dim * sizeof(T), 8 * sizeof(T));
                        diskann::load_aligned_bin<T>(data_path, data, num_points, dim, aligned_dim);
                        if constexpr (std::is_floating_point<T>::value) {
                            std::cout << "[TieredIndex] Starting PCA construction (float/double)..." << std::endl;
                            // Copy to Eigen matrix (only the first 'dim' of each vector)
                            std::cout << "[TieredIndex] Copying data to Eigen matrix..." << std::endl;
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_mat(num_points, dim);
                            for (size_t i = 0; i < num_points; ++i) {
                                for (size_t j = 0; j < dim; ++j) {
                                    data_mat(i, j) = data[i * aligned_dim + j];
                                }
                            }
                            diskann::aligned_free(data);
                            std::cout << "[TieredIndex] Mean centering..." << std::endl;
                            pca_mean = data_mat.colwise().mean();
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centered = data_mat.rowwise() - pca_mean;
                            std::cout << "[TieredIndex] Running SVD..." << std::endl;
                            Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                            pca_components = svd.matrixV().leftCols(PCA_DIM);
                            std::cout << "[TieredIndex] Projecting data and computing min/max for each PCA dim..." << std::endl;
                            // Project all data to PCA and compute min/max for each dim
                            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> projected = centered * pca_components.leftCols(PCA_DIM);
                            pca_min.resize(PCA_DIM);
                            pca_max.resize(PCA_DIM);
                            for (size_t i = 0; i < PCA_DIM; ++i) {
                                pca_min[i] = projected.col(i).minCoeff();
                                pca_max[i] = projected.col(i).maxCoeff();
                            }
                            std::cout << "[TieredIndex] PCA construction complete." << std::endl;
                            save_pca_to_file(false);
                        } else {
                            std::cout << "[TieredIndex] Starting PCA construction (int8/uint8 branch, using float)..." << std::endl;
                            // Convert to float for PCA
                            std::cout << "[TieredIndex] Converting data to float..." << std::endl;
                            std::vector<float> float_data(num_points * dim);
                            for (size_t i = 0; i < num_points; ++i) {
                                for (size_t j = 0; j < dim; ++j) {
                                    float_data[i * dim + j] = static_cast<float>(data[i * aligned_dim + j]);
                                }
                            }
                            diskann::aligned_free(data);
                            std::cout << "[TieredIndex] Copying float data to Eigen matrix..." << std::endl;
                            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_mat(num_points, dim);
                            for (size_t i = 0; i < num_points; ++i) {
                                for (size_t j = 0; j < dim; ++j) {
                                    data_mat(i, j) = float_data[i * dim + j];
                                }
                            }
                            std::cout << "[TieredIndex] Mean centering..." << std::endl;
                            pca_mean_float = data_mat.colwise().mean();
                            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> centered = data_mat.rowwise() - pca_mean_float;
                            std::cout << "[TieredIndex] Running SVD..." << std::endl;
                            Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
                            pca_components_float = svd.matrixV().leftCols(PCA_DIM);
                            std::cout << "[TieredIndex] Projecting data and computing min/max for each PCA dim..." << std::endl;
                            // Project all data to PCA and compute min/max for each dim
                            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> projected = centered * pca_components_float.leftCols(PCA_DIM);
                            pca_min_float.resize(PCA_DIM);
                            pca_max_float.resize(PCA_DIM);
                            for (size_t i = 0; i < PCA_DIM; ++i) {
                                pca_min_float[i] = projected.col(i).minCoeff();
                                pca_max_float[i] = projected.col(i).maxCoeff();
                            }
                            std::cout << "[TieredIndex] PCA construction complete." << std::endl;
                            save_pca_to_file(true);
                        }
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
                return num_points_in_memory_index.load(std::memory_order_relaxed);
            }
    };
}