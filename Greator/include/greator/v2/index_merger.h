#pragma once

#include "v2/graph_delta.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include "windows_customizations.h"
#include <omp.h>
#include "tbb/concurrent_vector.h"
#include "tbb/concurrent_hash_map.h"
#include "tbb/concurrent_unordered_map.h"
#include <thread>

namespace greator {

  struct Element_for_prune {
    float    value;
    uint32_t index;
  };

  bool compare_element_for_prune(Element_for_prune a, Element_for_prune b);
  template<typename T, typename TagT = uint32_t>
  class StreamingMerger {
   public:
    // constructor to read a constructed index, allocated IDs
    // disk_in : SSD-DiskANN index to merge into
    // mem_in : list of mem-DiskANN indices to merge into disk_in
    // disk_out : SSD-DiskANN index to write out
    // delete_list : list of IDs to delete from disk_in
    // ndims : dimensionality of full-prec vectors
    // dist : distance comparator -- WARNING :: assumed to be L2
    // beam_width : BW for search on disk_in
    // range : max out-degree
    // l_index : L param for indexing
    // maxc : max num of candidates to consider while pruning
    /*      StreamingMerger(const char* disk_in, const std::vector<std::string>
       &mem_in, const char* disk_out, const char* deleted_tags, const uint32_t
       ndims, Distance<T>* dist, const uint32_t beam_width, const uint32_t
       range, const uint32_t l_index, const float alpha, const uint32_t maxc,
       bool single_file_index);*/
    DISKANN_DLLEXPORT StreamingMerger(
        const uint32_t ndims, Distance<T> *dist, greator::Metric dist_metric,
        const uint32_t beam_width, const uint32_t range, const uint32_t l_index,
        const float alpha, const uint32_t maxc, bool single_file_index,
        uint32_t id_map = 0);

    DISKANN_DLLEXPORT ~StreamingMerger();

    DISKANN_DLLEXPORT void merge(
        const char *disk_in, const std::vector<std::string> &mem_in,
        const char                             *disk_out,
        std::vector<const std::vector<TagT> *> &deleted_tags,
        std::string &working_folder, uint32_t id_map = 0);
    // merge all memory indices into the disk index and write out new disk index
    void mergeImpl(uint32_t id_map = 0);

   private:
    /* insert related funcs */
    void process_inserts();
    void process_inserts_pq();
    void insert_mem_vec(const T *vec, const uint32_t offset_id);
    void offset_iterate_to_fixed_point(
        const T *vec, const uint32_t Lsize,
        std::vector<Neighbor>         &expanded_nodes_info,
        tsl::robin_map<uint32_t, T *> &coord_map);
    // used to prune insert() edges
    void prune_neighbors(const tsl::robin_map<uint32_t, T *> &coord_map,
                         std::vector<Neighbor>               &pool,
                         std::vector<uint32_t>               &pruned_list);
    // used to prune inter-insert() edges
    void prune_neighbors_pq(std::vector<Neighbor> &pool,
                            std::vector<uint32_t> &pruned_list,
                            uint8_t               *scratch = nullptr);
    void occlude_list(std::vector<Neighbor>               &pool,
                      const tsl::robin_map<uint32_t, T *> &coord_map,
                      std::vector<Neighbor>               &result,
                      std::vector<float>                  &occlude_factor);
    void occlude_list_pq(std::vector<Neighbor> &pool,
                         std::vector<Neighbor> &result,
                         std::vector<float>    &occlude_factor,
                         uint8_t               *scratch = nullptr);

    void dump_to_disk(const uint32_t start_id, const char *buf,
                      const uint32_t n_sector, std::ofstream &output_writer);

    /* delete related funcs */
    // converts tags into deleted IDs
    void compute_deleted_ids();
    // process all deletes
    void process_deletes();
    // reads nhoods of all deleted nods
    void populate_deleted_nhoods();
    // eliminates references to deleted nodes in id_nhoods
    uint32_t consolidate_deletes(DiskNode<T> &disk_node,
                                 uint8_t     *scratch = nullptr);
    // whether the specific node is deleted / node id not in use
    bool is_deleted(const DiskNode<T> &disk_node);

    /* rename related funcs */
    // assign smallest free IDs to new inserts
    void compute_rename_map();
    void rename(DiskNode<T> &node) const;
    void rename(std::vector<uint32_t> &nhood) const;
    // returns uint32_t::max() upon failure
    uint32_t rename(uint32_t id) const;
    // returns uint32_t::max() upon failure
    uint32_t rename_inverse(uint32_t renamed_id) const;
    // returns ID of mem index offset_id belongs to; uint32_t::max() otherwise
    uint32_t              get_index_id(const uint32_t offset_id) const;
    std::vector<uint32_t> get_edge_list(const uint32_t offset_id);
    const T              *get_mem_data(const uint32_t offset_id);

    /* merge related funcs */
    void write_tag_file(const std::string &tag_out_filename,
                        const uint32_t     npts);
    void process_merges();
    // read io info
    static inline void get_io_info(std::string info = "") {
      static std::map<std::string, long long> previousDataMap;
      std::string                             pid = std::to_string(getpid());
      std::string ioStatsFilePath = "/proc/" + pid + "/io";

      std::ifstream                    file(ioStatsFilePath);
      std::map<std::string, long long> currentDataMap;

      if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
          size_t delimiterPos = line.find(':');
          if (delimiterPos != std::string::npos) {
            std::string key = line.substr(0, delimiterPos);
            std::string valueString = line.substr(delimiterPos + 1);
            long long   value = std::stoull(valueString);
            currentDataMap[key] = value;
          }
        }

        file.close();

        // print diff value
        for (const auto &entry : currentDataMap) {
          printf(" %s%s(GB): %.2f\n", info.c_str(), entry.first.c_str(),
                 entry.second / 1024.0 / 1024 / 1024);
          if (previousDataMap.count(entry.first) > 0) {
            long long diff = entry.second - previousDataMap[entry.first];
            printf(" @%s%s(GB): %.2f\n", info.c_str(), entry.first.c_str(),
                   diff / 1024.0 / 1024 / 1024);
          } else {
            printf(" @%s%s(GB): %.2f\n", info.c_str(), entry.first.c_str(),
                   entry.second / 1024.0 / 1024 / 1024);
          }
        }

        previousDataMap = currentDataMap;
      } else {
        std::cerr << "Failed to open file: " << ioStatsFilePath << std::endl;
      }
    }
    // deletes
    tsl::robin_set<TagT>                            deleted_tags;
    tsl::robin_map<uint32_t, std::vector<uint32_t>> disk_deleted_nhoods;
    tsl::robin_set<uint32_t>                        disk_deleted_ids;
    std::vector<tsl::robin_set<uint32_t>>           mem_deleted_ids;
    char *delete_backing_buf = nullptr;

    // rename stuff
    tsl::robin_map<uint32_t, uint32_t>         rename_map;
    tsl::robin_map<uint32_t, uint32_t>         inverse_map;
    std::vector<std::pair<uint32_t, uint32_t>> rename_list;
    std::vector<std::pair<uint32_t, uint32_t>> inverse_list;

    // disk index
    GraphDelta            *disk_delta;
    PQFlashIndex<T, TagT> *disk_index;
    std::vector<uint32_t>  init_ids;
    uint8_t               *pq_data = nullptr;
    TagT                  *disk_tags = nullptr;
    uint32_t               pq_nchunks;
    uint32_t               max_node_len, nnodes_per_sector, disk_npts;
    std::string disk_index_out_path, disk_index_in_path, pq_coords_file;
    std::string temp_disk_index_path, temp_pq_coords_path, temp_tags_path;
    std::string final_index_file, final_pq_coords_file, final_tags_file;
    // std::fstream output_writer;
    std::vector<ThreadData<T>> disk_thread_data;

    // mem-index
    std::vector<GraphDelta *> mem_deltas;
    // std::vector<Index<T, TagT> *>                           mem_indices;
    // std::vector<const std::vector<std::vector<uint32_t>> *> mem_graphs;
    std::vector<const T *>               mem_data;
    std::vector<std::unique_ptr<TagT[]>> mem_tags;
    std::vector<uint32_t>                offset_ids;
    std::vector<uint32_t>                mem_npts;
    Distance<T>                         *dist_cmp;
    greator::Metric                      dist_metric;
    // T *                                                     _data_load;
    std::vector<tsl::robin_set<TagT>> latter_deleted_tags;

    // allocators
    // FixedSizeAlignedAllocator<T> *fp_alloc = nullptr;
    // FixedSizeAlignedAllocator<uint8_t> *pq_alloc = nullptr;

    // book keeping
    tbb::concurrent_vector<uint32_t> free_ids;
    char                            *thread_pq_scratch = nullptr;
    std::vector<char *>              thread_bufs;
    // vector info
    uint32_t ndims, aligned_ndims;
    // search + index params
    uint32_t beam_width;
    uint32_t l_index, range, maxc;
    float    alpha;

    uint64_t disk_index_num_frozen = 0;  // assuming that every merge
    // folds a single mem index into the PQFlashIndex
    uint64_t disk_index_frozen_loc = 0;
    // timing stuff
    std::vector<float> insert_times, delta_times, search_times, prune_times;
    std::mutex         _print_lock;

    bool _single_file_index = false;

    std::string TMP_FOLDER;

    // linsy
    uint32_t            id_map = 0;
    std::vector<double> prune_time;
    void                get_sector_with_deleted_nbrs_nodes(
                       const std::string &indir,
                       tbb::concurrent_unordered_map<uint32_t,
                                                     tbb::concurrent_vector<uint32_t>>
                           &sector_with_deleted_nbrs_nodes);

    /*
        update topology_graph after delete_phase or merge_phase
     */

    _u32 topological_size_per_node, topological_nnodes_per_sector,
        topological_page_num;

    // topological_delete();
    tbb::concurrent_vector<DiskNode<T>> delete_nodes;
    // tbb::concurrent_vector<std::pair<_u32, _u32>>
    //      topological_delete_map;  // key=page_id,value=in_nodes.location
    void process_topological_delete();
    // topological_merge();
    tbb::concurrent_vector<DiskNode<T>> merge_nodes;
    // tbb::concurrent_vector<std::pair<_u32, _u32>> topological_merge_map;
    void process_topological_merge();

    std::vector<uint32_t> findMinKIndexes(const std::vector<float> &dists,
                                          int                       k) {
      std::vector<Element_for_prune> elements;

      for (uint32_t i = 0; i < dists.size(); ++i) {
        elements.push_back({dists[i], i});
      }

      std::nth_element(elements.begin(), elements.begin() + k, elements.end(),
                       compare_element_for_prune);

      std::vector<uint32_t> result;
      for (int i = 0; i < k; ++i) {
        result.push_back(elements[i].index);
      }

      return result;
    }
    std::vector<std::vector<_u32>> topo;
    omp_lock_t                     prune_time_mutex;
    omp_lock_t                     in_nodes_mutex;
    omp_lock_t                     merge_nodes_mutex;
    omp_lock_t                     free_ids_mutex;
    omp_lock_t                     cout_mutex;
    omp_lock_t                     compute_pq_dists_mutex;
    uint32_t                       prune_nums = 0;
    uint32_t                       not_prune_nums = 0;
  };
};  // namespace diskann
