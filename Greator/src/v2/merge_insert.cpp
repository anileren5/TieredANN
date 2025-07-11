#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/merge_insert.h"
#include <csignal>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "tcmalloc/malloc_extension.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "aux_utils.h"
#include "exceptions.h"
#include "index.h"
#include "pq_flash_index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"

#include "logger.h"

#include "Neighbor_Tag.h"
#ifdef _WINDOWS
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#else
#include "linux_aligned_file_reader.h"
#endif

namespace greator {
  template<typename T, typename TagT>
  MergeInsert<T, TagT>::MergeInsert(
      Parameters& parameters, size_t dim, const std::string mem_prefix,
      const std::string disk_prefix_in, const std::string disk_prefix_out,
      Distance<T>* dist, greator::Metric dist_metric, bool single_file_index,
      std::string working_folder)
      : _dim(dim), _dist_metric(dist_metric), _active_0(true), _active_1(false),
        _active_del_0(true), _active_del_1(false), _clearing_index_0(false),
        _clearing_index_1(false), _switching_disk_prefixes(false),
        _check_switch_index(false), _check_switch_delete(false) {
    _merge_th = MERGE_TH;

    std::cout << "_merge_th in merge_insert.cpp:" << _merge_th
              << " MERGE_TH:" << MERGE_TH << std::endl;
    _single_file_index = single_file_index;
    this->_dist_metric = dist_metric;
    _mem_index_0 = std::make_shared<greator::Index<T, TagT>>(
        this->_dist_metric, dim, _merge_th * 2, 1, _single_file_index, 1);
    _mem_index_1 = std::make_shared<greator::Index<T, TagT>>(
        this->_dist_metric, dim, _merge_th * 2, 1, _single_file_index, 1);

    _paras_mem.Set<unsigned>("L", parameters.Get<unsigned>("L_mem"));
    _paras_mem.Set<unsigned>("R", parameters.Get<unsigned>("R_mem"));
    _paras_mem.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_mem.Set<float>("alpha", parameters.Get<float>("alpha_mem"));
    _paras_mem.Set<unsigned>("num_rnds", 2);
    _paras_mem.Set<bool>("saturate_graph", 0);

    _paras_disk.Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    _paras_disk.Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    _paras_disk.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_disk.Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    _paras_disk.Set<unsigned>("num_rnds", 2);
    _paras_disk.Set<bool>("saturate_graph", 0);

    _num_search_threads = parameters.Get<_u32>("num_search_threads");
    _beamwidth = parameters.Get<uint32_t>("beamwidth");
    _num_nodes_to_cache = parameters.Get<_u32>("nodes_to_cache");

    // _search_tpool = new ThreadPool(_num_search_threads);

    _mem_index_prefix = mem_prefix;
    _deleted_tags_file = mem_prefix + "_deleted.tags";
    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    _disk_index = new greator::PQFlashIndex<T, TagT>(this->_dist_metric, reader,
                                                     _single_file_index, true);
    std::cout << "Start_LOAD\n";
    std::string pq_prefix = _disk_index_prefix_in + "_pq";
    std::string disk_index_file = _disk_index_prefix_in + "_disk.index";

    int res =
        _disk_index->load(_disk_index_prefix_in.c_str(), _num_search_threads);
    if (res != 0) {
      greator::cout << "Failed to load disk index in MergeInsert constructor"
                    << std::endl;
      exit(-1);
    }

    TMP_FOLDER = working_folder;
    std::cout << "TMP_FOLDER inside MergeInsert : " << TMP_FOLDER << std::endl;
  }

  template<typename T, typename TagT>
  MergeInsert<T, TagT>::~MergeInsert() {
    // put in destructor code
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::construct_index_merger(uint32_t id_map) {
    uint32_t range = _paras_disk.Get<unsigned>("R");
    uint32_t l_index = _paras_disk.Get<unsigned>("L");
    uint32_t maxc = _paras_disk.Get<unsigned>("C");
    float    alpha = _paras_disk.Get<float>("alpha");
    if (id_map)
      _merger = new greator::StreamingMerger<T, TagT>(
          (uint32_t) _dim, _dist_comp, _dist_metric, (uint32_t) _beamwidth,
          range, l_index, alpha, maxc, _single_file_index, id_map);
    else
      _merger = new greator::StreamingMerger<T, TagT>(
          (uint32_t) _dim, _dist_comp, _dist_metric, (uint32_t) _beamwidth,
          range, l_index, alpha, maxc, _single_file_index);
    greator::cout << "Created index merger object" << std::endl;
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::destruct_index_merger() {
    delete (_merger);
    _merger = nullptr;
  }

  template<typename T, typename TagT>
  int MergeInsert<T, TagT>::insert(const T* point, const TagT& tag) {
    while (_check_switch_index.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::shared_lock<std::shared_timed_mutex> lock(_index_lock);
    if ((_active_index == 0) && (_active_0.load() == false)) {
      greator::cout << "Active index indicated as _mem_index_0 but it cannot "
                       "accept insertions"
                    << std::endl;
      return -1;
    }
    if ((_active_index == 1) && (_active_1.load() == false)) {
      greator::cout << "Active index indicated as _mem_index_1 but it cannot "
                       "accept insertions"
                    << std::endl;
      return -1;
    }

    if (_active_index == 0) {
      if (_mem_index_0->get_num_points() < _mem_index_0->return_max_points()) {
        if (_mem_index_0->insert_point(point, _paras_mem, tag) != 0) {
          greator::cout << "Could not insert point with tag " << tag
                        << std::endl;
          return -3;
        }
        {
          std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
          _mem_points++;
        }
        return 0;
      } else {
        greator::cout << "Capacity exceeded" << std::endl;
      }
    } else {
      if (_mem_index_1->get_num_points() < _mem_index_1->return_max_points()) {
        if (_mem_index_1->insert_point(point, _paras_mem, tag) != 0) {
          greator::cout << "Could not insert point with tag " << tag
                        << std::endl;
          return -3;
        }
        {
          std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
          _mem_points++;
        }
        return 0;
      } else {
        greator::cout << "Capacity exceeded in mem_index 1" << std::endl;
      }
    }

    return -2;
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::lazy_delete(const TagT& tag) {
    std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
    if ((_active_delete_set == 0) && (_active_del_0.load() == false)) {
      greator::cout << "Active deletion set indicated as _deletion_set_0 but "
                       "it cannot accept deletions"
                    << std::endl;
    }
    if ((_active_delete_set == 1) && (_active_del_1.load() == false)) {
      greator::cout << "Active deletion set indicated as _deletion_set_1 but "
                       "it cannot accept deletions"
                    << std::endl;
    }

    if (_active_delete_set == 0) {
      _deletion_set_0.insert(tag);
      _mem_index_0->lazy_delete(tag);
    } else {
      _deletion_set_1.insert(tag);
      _mem_index_1->lazy_delete(tag);
    }
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::search_sync(const T* query, const uint64_t K,
                                         const uint64_t search_L, TagT* tags,
                                         float* distances, QueryStats* stats) {
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();
    bool reachLimit = false;
    std::set<Neighbor_Tag<TagT>> best;
    // search disk index and get top K tags
    // check each memory index - if non empty and not being currently cleared -
    // search and get top K active tags
    {
      std::shared_lock<std::shared_timed_mutex> lock(_disk_lock);
      assert(_switching_disk_prefixes == false);
      std::vector<uint64_t> disk_result_ids_64(search_L);
      std::vector<float>    disk_result_dists(search_L);
      std::vector<TagT>     disk_result_tags(search_L);
      int                   searchNum = _disk_index->cached_beam_search(
          query, search_L, search_L, disk_result_tags.data(),
          disk_result_dists.data(), _beamwidth, stats);
      // std::cout << "searchNUM: " << searchNum << std::endl;
      auto   diskSearchEnd = std::chrono::high_resolution_clock::now();
      double elapsedSeconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd -
                                                                diskSearchBegin)
              .count();
      // std::cout << "Disk_search Time: " << elapsedSeconds << std ::endl;
      // std::cout << "SearchNum:" << searchNum << std::endl;
      for (unsigned i = 0; i < searchNum; i++) {
        Neighbor_Tag<TagT> n;
        n = Neighbor_Tag<TagT>(disk_result_tags[i], disk_result_dists[i]);
        //                    best.insert(Neighbor_Tag<TagT>(disk_result_tags[i],
        //                    disk_result_dists[i]));
        best.insert(n);
      }
    }

    if (stats != nullptr && !reachLimit) {
      auto   diskSearchEnd = std::chrono::high_resolution_clock::now();
      double elapsedSeconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd -
                                                                diskSearchBegin)
              .count();
      if (elapsedSeconds >= stats->n_current_used)
        reachLimit = true;
    }

    {
      if (_clearing_index_0.load() == false && !reachLimit) {
        std::shared_lock<std::shared_timed_mutex> lock(_clear_lock_0);
        if (_mem_index_0->get_num_points() > 0) {
          std::vector<Neighbor_Tag<TagT>> best_mem_index_0;
          auto memSearchBegin = std::chrono::high_resolution_clock::now();
          _mem_index_0->search(query, (uint32_t) search_L, (uint32_t) search_L,
                               best_mem_index_0);
          auto   memSearchEnd = std::chrono::high_resolution_clock::now();
          double elapsedSeconds =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  memSearchEnd - memSearchBegin)
                  .count();
          // std::cout << "Mem1_search Time: " << elapsedSeconds << std ::endl;
          // std::cout << "mem_search Num: " << best_mem_index_0.size()
          //           << std::endl;
          for (auto iter : best_mem_index_0)
            best.insert(iter);
        }
      }

      if (stats != nullptr) {
        auto   diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                diskSearchEnd - diskSearchBegin)
                .count();
        if (elapsedSeconds >= stats->n_current_used)
          reachLimit = true;
      }

      if (_clearing_index_1.load() == false && !reachLimit) {
        std::shared_lock<std::shared_timed_mutex> lock(_clear_lock_1);
        if (_mem_index_1->get_num_points() > 0) {
          std::vector<Neighbor_Tag<TagT>> best_mem_index_1;
          _mem_index_1->search(query, (uint32_t) search_L, (uint32_t) search_L,
                               best_mem_index_1);
          for (auto iter : best_mem_index_1)
            best.insert(iter);
        }
        // auto   diskSearchEnd = std::chrono::high_resolution_clock::now();
        // double elapsedSeconds =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(
        //         diskSearchEnd - diskSearchBegin)
        //         .count();
        // std::cout << "Mem2_search Time: " << elapsedSeconds << std ::endl;
      }
    }

    std::vector<Neighbor_Tag<TagT>> best_vec;
    for (auto iter : best)
      best_vec.emplace_back(iter);
    std::sort(best_vec.begin(), best_vec.end());
    if (best_vec.size() > K)
      best_vec.erase(best_vec.begin() + K, best_vec.end());
    // aggregate results, sort and pick top K candidates
    {
      std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
      size_t                                    pos = 0;
      for (auto iter : best_vec) {
        if ((_deletion_set_0.find(iter.tag) == _deletion_set_0.end()) &&
            (_deletion_set_1.find(iter.tag) == _deletion_set_1.end())) {
          tags[pos] = iter.tag;
          distances[pos] = iter.dist;
          pos++;
        }
        if (pos == K)
          break;
      }
    }
  }

  template<typename T, typename TagT>
  int MergeInsert<T, TagT>::trigger_merge() {
    if (_mem_points >= _merge_th) {
      save_del_set();
      switch_index();
      return 1;
    }
    return 0;
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::final_merge(uint32_t id_map) {
    greator::cout << "Inside final_merge()." << std::endl;
    greator::cout << _mem_index_0->get_num_points() << "  "
                  << _mem_index_1->get_num_points() << std::endl;
    if (_mem_points > 0) {
      save_del_set();
      switch_index(id_map);
    }
    greator::cout << _mem_index_0->get_num_points() << "  "
                  << _mem_index_1->get_num_points() << std::endl;
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::merge(uint32_t id_map) {
    std::vector<std::string> mem_in;
    if (_active_index == 0)
      mem_in.push_back(_mem_index_prefix + "_1");
    else
      mem_in.push_back(_mem_index_prefix + "_0");

    std::cout << "merge_start: _disk_index_prefix_in: " << _disk_index_prefix_in
              << " _disk_index_prefix_out: " << _disk_index_prefix_out
              << std::endl;
    _merger->merge(_disk_index_prefix_in.c_str(), mem_in,
                   _disk_index_prefix_out.c_str(), _deleted_tags_vector,
                   TMP_FOLDER, id_map);

    greator::cout << "Merge done" << std::endl;
    {
      std::unique_lock<std::shared_timed_mutex> lock(_disk_lock);
      bool                                      expected_value = false;
      if (_switching_disk_prefixes.compare_exchange_strong(expected_value,
                                                           true)) {
        greator::cout << "Switching to latest merged disk index " << std::endl;
      } else {
        greator::cout << "Failed to switch" << std::endl;
        //              return -1;
      }

      std::string temp = _disk_index_prefix_out;
      _disk_index_prefix_out = _disk_index_prefix_in;
      _disk_index_prefix_in = temp;
      std::cout << "merge_end: _disk_index_prefix_in: " << _disk_index_prefix_in
                << " _disk_index_prefix_out: " << _disk_index_prefix_out
                << std::endl;
      delete (_disk_index);
      _disk_index = new greator::PQFlashIndex<T, TagT>(
          this->_dist_metric, reader, _single_file_index, true);

      std::string pq_prefix = _disk_index_prefix_in + "_pq";
      std::string disk_index_file = _disk_index_prefix_in + "_disk.index";
      int         res =
          _disk_index->load(_disk_index_prefix_in.c_str(), _num_search_threads);
      if (res != 0) {
        greator::cout << "Failed to load new disk index after merge"
                      << std::endl;
        exit(-1);
      }
      expected_value = true;
      _switching_disk_prefixes.compare_exchange_strong(expected_value, false);
    }
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::switch_index(uint32_t id_map) {
    // unique lock throughout the function to ensure another thread does not
    // flip the value of _active_index after it has been saved by one thread,
    // and multiple threads do not save the same index
    // unique lock is acquired when no other thread holds any shared lock over
    // it, so this function will wait till any on-going insertions are completed
    // and then change the value of all related flags
    std::cout << "Begin_switch_index" << std::endl;
    {
      bool expected_value = false;
      _check_switch_index.compare_exchange_strong(expected_value, true);

      std::unique_lock<std::shared_timed_mutex> lock(_index_lock);

      // make new index active
      if (_active_index == 0) {
        _mem_index_1 = std::make_shared<greator::Index<T, TagT>>(
            this->_dist_metric, _dim, _merge_th * 2, 1, _single_file_index, 1);
        bool expected_active = false;

        if (_active_1.compare_exchange_strong(expected_active, true)) {
          greator::cout << "Initialised new index for _mem_index_1 "
                        << std::endl;
        } else {
          greator::cout << "Failed to initialise new _mem_index_1" << std::endl;
          //              return -1;
        }

      } else {
        _mem_index_0 = std::make_shared<greator::Index<T, TagT>>(
            this->_dist_metric, _dim, _merge_th * 2, 1, _single_file_index, 1);
        bool expected_active = false;

        if (_active_0.compare_exchange_strong(expected_active, true)) {
          greator::cout << "Initialised new index for _mem_index_0 "
                        << std::endl;
        } else {
          greator::cout << "Failed to initialise new _mem_index_0" << std::endl;
          //            return -1;
        }
      }

      _active_index = 1 - _active_index;
      _mem_points = 0;
      expected_value = true;
      _check_switch_index.compare_exchange_strong(expected_value, false);
    }
    save();
    construct_index_merger(id_map);
    merge(id_map);
    destruct_index_merger();

    {
      std::shared_lock<std::shared_timed_mutex> lock(_index_lock);
      // make older index inactive after merge is complete or before ?
      if (_active_index == 0) {
        bool expected_clearing = false;
        bool expected_active = true;
        _clearing_index_1.compare_exchange_strong(expected_clearing, true);
        {
          std::unique_lock<std::shared_timed_mutex> lock(_clear_lock_1);
          _mem_index_1.reset();
          _mem_index_1 = nullptr;
          _mem_index_1 = std::make_shared<greator::Index<T, TagT>>(
              _dist_metric, _dim, _merge_th * 2, 1, _single_file_index, 1);
        }
        expected_clearing = true;
        assert(expected_clearing == true);
        _clearing_index_1.compare_exchange_strong(expected_clearing, false);
        assert(expected_active == true);
        _active_1.compare_exchange_strong(expected_active, false);
      } else {
        bool expected_clearing = false;
        bool expected_active = true;
        _clearing_index_0.compare_exchange_strong(expected_clearing, true);
        std::unique_lock<std::shared_timed_mutex> lock(_clear_lock_0);
        {
          _mem_index_0.reset();
          _mem_index_0 = nullptr;
          _mem_index_0 = std::make_shared<greator::Index<T, TagT>>(
              _dist_metric, _dim, _merge_th * 2, 1, _single_file_index, 1);
        }
        expected_clearing = true;
        assert(expected_clearing == true);
        _clearing_index_0.compare_exchange_strong(expected_clearing, false);
        assert(expected_active == true);
        _active_0.compare_exchange_strong(expected_active, false);
      }
      // if merge() has returned, clear older active index
    }
  }

  template<typename T, typename TagT>
  int MergeInsert<T, TagT>::save() {
    // only switch_index will call this function
    bool expected_active = true;
    if (_active_index == 1) {
      if (_active_0.compare_exchange_strong(expected_active, false)) {
        greator::cout << "Saving mem index 0 to merge it into disk index"
                      << std::endl;
        std::string save_path = _mem_index_prefix + "_0";
        _mem_index_0->save(save_path.c_str());
      } else {
        greator::cout << "Index 0 is already inactive" << std::endl;
        return -1;
      }
    } else {
      if (_active_1.compare_exchange_strong(expected_active, false)) {
        greator::cout << "Saving mem index 1 to merge it into disk index"
                      << std::endl;
        std::string save_path = _mem_index_prefix + "_1";
        _mem_index_1->save(save_path.c_str());
      } else {
        greator::cout << "Index 1 is already inactive" << std::endl;
        return -1;
      }
    }
    greator::cout << "Saved mem index" << std::endl;
    return 0;
  }

  template<typename T, typename TagT>
  void MergeInsert<T, TagT>::save_del_set() {
    greator::Timer timer;
    {
      bool expected_value = false;
      _check_switch_delete.compare_exchange_strong(expected_value, true);
      std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
      if (_active_delete_set == 0) {
        _deletion_set_1.clear();
        bool expected_active = false;
        if (_active_del_1.compare_exchange_strong(expected_active, true)) {
          greator::cout
              << "Cleared _deletion_set_1 - ready to accept new points"
              << std::endl;
        } else {
          greator::cout << "Failed to clear _deletion_set_1" << std::endl;
        }
      } else {
        _deletion_set_0.clear();
        bool expected_active = false;
        if (_active_del_0.compare_exchange_strong(expected_active, true)) {
          greator::cout
              << "Cleared _deletion_set_0 - ready to accept new points"
              << std::endl;
        } else {
          greator::cout << "Failed to clear _deletion_set_0" << std::endl;
        }
      }
      _active_delete_set = 1 - _active_delete_set;
      bool expected_active = true;
      if (_active_delete_set == 0)
        _active_del_1.compare_exchange_strong(expected_active, false);
      else
        _active_del_0.compare_exchange_strong(expected_active, false);
      expected_value = true;
      _check_switch_delete.compare_exchange_strong(expected_value, false);
    }

    if (_active_delete_set == 0) {
      std::vector<TagT>* del_vec =
          new std::vector<TagT>(_deletion_set_1.size());

      size_t i = 0;
      for (auto iter : _deletion_set_1) {
        (*del_vec)[i] = iter;
        i++;
      }
      _deleted_tags_vector.clear();
      _deleted_tags_vector.push_back(del_vec);
    } else {
      std::vector<TagT>* del_vec =
          new std::vector<TagT>(_deletion_set_0.size());

      size_t i = 0;
      for (auto iter : _deletion_set_0) {
        (*del_vec)[i] = iter;
        i++;
      }
      _deleted_tags_vector.clear();
      _deleted_tags_vector.push_back(del_vec);
    }
    double time = (double) timer.elapsed() / 1000000.0;
    std::cout << "_deleted_tags_vector push over! use " << time << " s!"
              << std::endl;
  }

  template<typename T, typename TagT>
  std::string MergeInsert<T, TagT>::ret_merge_prefix() {
    return _disk_index_prefix_in;
  }
  // template class instantiations
  template class MergeInsert<float, uint32_t>;
  template class MergeInsert<uint8_t, uint32_t>;
  template class MergeInsert<int8_t, uint32_t>;
  template class MergeInsert<float, int64_t>;
  template class MergeInsert<uint8_t, int64_t>;
  template class MergeInsert<int8_t, int64_t>;
  template class MergeInsert<float, uint64_t>;
  template class MergeInsert<uint8_t, uint64_t>;
  template class MergeInsert<int8_t, uint64_t>;
}  // namespace diskann
