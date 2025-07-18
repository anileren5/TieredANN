// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iomanip>
#include <omp.h>

#include "greator/pq_flash_index.h"
#include "greator/aux_utils.h"
#include "greator/linux_aligned_file_reader.h"
#include <boost/program_options.hpp>

#define WARMUP true

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  greator::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    greator::cout << std::setw(8) << percentiles[s] << "%";
  }
  greator::cout << std::endl;
  greator::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    greator::cout << std::setw(9) << results[s];
  }
  greator::cout << std::endl;
}

template <typename T> int search_disk_index(int argc, char **argv) {
  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  // Parse command line arguments using boost::program_options
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  
  std::string index_prefix_path, query_bin, truthset_bin, result_output_prefix, dist_metric;
  bool single_file_index, tags_flag;
  _u64 num_nodes_to_cache, recall_at;
  _u32 num_threads, beamwidth;
  uint32_t sector_len = 4096; // Default value
  
  desc.add_options()
    ("index_prefix_path", po::value<std::string>(&index_prefix_path)->required(), "Index prefix path")
    ("single_file_index", po::value<bool>(&single_file_index)->required(), "Single file index (0/1)")
    ("tags", po::value<bool>(&tags_flag)->required(), "Tags flag (0/1)")
    ("num_nodes_to_cache", po::value<_u64>(&num_nodes_to_cache)->required(), "Number of nodes to cache")
    ("num_threads", po::value<_u32>(&num_threads)->required(), "Number of threads")
    ("beamwidth", po::value<_u32>(&beamwidth)->required(), "Beamwidth")
    ("query_bin", po::value<std::string>(&query_bin)->required(), "Query file")
    ("truthset_bin", po::value<std::string>(&truthset_bin)->required(), "Truthset file")
    ("recall_at", po::value<_u64>(&recall_at)->required(), "Recall at K")
    ("result_output_prefix", po::value<std::string>(&result_output_prefix)->required(), "Result output prefix")
    ("dist_metric", po::value<std::string>(&dist_metric)->required(), "Distance metric")
    ("sector_len", po::value<uint32_t>(&sector_len)->default_value(4096), "Sector length in bytes")
    ("L_values", po::value<std::vector<_u64>>()->multitoken(), "L values for search");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Set the global SECTOR_LEN variable
  set_sector_len(sector_len);

  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";

  greator::Metric m =
      dist_metric == "cosine" ? greator::Metric::COSINE : greator::Metric::L2;
  if (dist_metric != "l2" && m == greator::Metric::L2) {
    greator::cout << "Unknown distance metric: " << dist_metric
                  << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  // Parse L values from command line arguments
  if (vm.count("L_values")) {
    std::vector<_u64> l_values = vm["L_values"].as<std::vector<_u64>>();
    for (_u64 curL : l_values) {
      if (curL >= recall_at)
        Lvec.push_back(curL);
    }
  }

  if (Lvec.size() == 0) {
    greator::cout
        << "No valid Lsearch found. Lsearch must be at least recall_at"
        << std::endl;
    return -1;
  }

  greator::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    greator::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    greator::cout << " beamwidth: " << beamwidth << std::endl;

  greator::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);

  if (greator::file_exists(truthset_bin)) {
    greator::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &tags);
    if (gt_num != query_num) {
      greator::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<greator::AlignedFileReader> reader = nullptr;
  reader.reset(new greator::LinuxAlignedFileReader());

  std::unique_ptr<greator::PQFlashIndex<T>> _pFlashIndex(
      new greator::PQFlashIndex<T>(m, reader, single_file_index,
                                   tags_flag)); // no tags support yet.

  int res = _pFlashIndex->load(index_prefix_path.c_str(), num_threads);
  if (res != 0) {
    return res;
  }

  if (tags_flag) {
    tsl::robin_set<_u32> active_tags;
    _pFlashIndex->get_active_tags(active_tags);

    greator::cout << "Loaded " << active_tags.size()
                  << " tags from index for recall measurement." << std::endl;
  } else {
    greator::cout << "Not loading tags since they are disabled." << std::endl;
  }

  // cache bfs levels
  std::vector<uint32_t> node_list;
  greator::cout << "Caching " << num_nodes_to_cache
                << " BFS nodes around medoid(s)" << std::endl;
  _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  //_pFlashIndex->generate_cache_list_from_sample_queries(
  //   warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
  _pFlashIndex->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  omp_set_num_threads(num_threads);

  uint64_t warmup_L;
  warmup_L = 20;
  uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
  T *warmup = nullptr;
  if (WARMUP) {
    if (greator::file_exists(warmup_query_file)) {
      greator::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                   warmup_dim, warmup_aligned_dim);
    } else {
      warmup_num = (std::min)((_u32)150000, (_u32)15000 * num_threads);
      warmup_dim = query_dim;
      warmup_aligned_dim = query_aligned_dim;
      greator::alloc_aligned(((void **)&warmup),
                             warmup_num * warmup_aligned_dim * sizeof(T),
                             8 * sizeof(T));
      std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(-128, 127);
      for (uint32_t i = 0; i < warmup_num; i++) {
        for (uint32_t d = 0; d < warmup_dim; d++) {
          warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
        }
      }
    }
    greator::cout << "Warming up index... " << std::flush;
    std::vector<uint64_t> warmup_result_tags_64(warmup_num, 0);
    std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t)warmup_num; i++) {
      _pFlashIndex->cached_beam_search_ids(
          warmup + (i * warmup_aligned_dim), 1, warmup_L,
          warmup_result_tags_64.data() + (i * 1),
          warmup_result_dists.data() + (i * 1), (uint64_t)4);
    }
    greator::cout << "..done" << std::endl;
  }

  greator::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  greator::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  greator::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
                << std::setw(16) << "99.9 Latency" << std::setw(16)
                << "Mean IOs" << std::setw(16) << "CPU (s)";
  if (calc_recall_flag) {
    greator::cout << std::setw(16) << recall_string << std::endl;
  } else
    greator::cout << std::endl;
  greator::cout
      << "==============================================================="
         "==========================================="
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  uint32_t optimized_beamwidth = 2;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (beamwidth <= 0) {
      // diskann::cout<<"Tuning beamwidth.." << std::endl;
      optimized_beamwidth =
          optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                             warmup_aligned_dim, (_u32)L, optimized_beamwidth);
    } else
      optimized_beamwidth = beamwidth;

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    greator::QueryStats *stats = new greator::QueryStats[query_num];

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t)query_num; i++) {
      _pFlashIndex->cached_beam_search(
          query + (i * query_aligned_dim), (uint64_t)recall_at, (uint64_t)L,
          query_result_tags_32.data() + (i * recall_at),
          query_result_dists[test_id].data() + (i * recall_at),
          (uint64_t)optimized_beamwidth, stats + i);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps =
        (float)((1.0 * (double)query_num) / (1.0 * (double)diff.count()));

    greator::convert_types<uint32_t, uint32_t>(
        query_result_tags_32.data(), query_result_tags[test_id].data(),
        (size_t)query_num, (size_t)recall_at);

    float mean_latency = (float)greator::get_mean_stats(
        stats, query_num,
        [](const greator::QueryStats &stats) { return stats.total_us; });

    /*    float latency_90 = (float) diskann::get_percentile_stats(
            stats, query_num, 0.900,
            [](const diskann::QueryStats& stats) { return stats.total_us; });

        float latency_95 = (float) diskann::get_percentile_stats(
            stats, query_num, 0.950,
            [](const diskann::QueryStats& stats) { return stats.total_us; });
    */
    float latency_999 = (float)greator::get_percentile_stats(
        stats, query_num, 0.999f,
        [](const greator::QueryStats &stats) { return stats.total_us; });

    float mean_ios = (float)greator::get_mean_stats(
        stats, query_num,
        [](const greator::QueryStats &stats) { return stats.n_ios; });

    float mean_cpuus = (float)greator::get_mean_stats(
        stats, query_num,
        [](const greator::QueryStats &stats) { return stats.cpu_us; });
    delete[] stats;

    float recall = 0;
    if (calc_recall_flag) {
      recall = (float)greator::calculate_recall(
          (_u32)query_num, gt_ids, gt_dists, (_u32)gt_dim,
          query_result_tags[test_id].data(), (_u32)recall_at, (_u32)recall_at);
    }

    greator::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth
                  << std::setw(16) << qps << std::setw(16) << mean_latency
                  << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                  << std::setw(16) << mean_cpuus;
    if (calc_recall_flag) {
      greator::cout << std::setw(16) << recall << std::endl;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(10));

  greator::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    greator::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);

    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_tags_uint32.bin";
    greator::save_bin<_u32>(cur_result_path, query_result_tags[test_id].data(),
                            query_num, recall_at);
    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    greator::save_bin<float>(cur_result_path,
                             query_result_dists[test_id++].data(), query_num,
                             recall_at);
  }

  greator::aligned_free(query);
  if (warmup != nullptr)
    greator::aligned_free(warmup);
  delete[] gt_ids;
  delete[] gt_dists;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    greator::cout
        << "Usage: " << argv[0]
        << " <index_type (float/int8/uint8)> --index_prefix_path <path>"
           " --single_file_index <0/1> --tags <0/1> "
           " --num_nodes_to_cache <num> --num_threads <num> --beamwidth <num> "
           " --query_bin <file> --truthset_bin <file> "
           " --recall_at <K> --result_output_prefix <prefix> --dist_metric <cosine/l2> "
           " --sector_len <bytes> --L_values <L1> <L2> ..."
        << std::endl;
    exit(-1);
  }

  greator::cout << "Attach debugger and press a key" << std::endl;
  /*  char x;
    std::cin >> x;
    */

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    greator::cout << "Unsupported index type. Use float or int8 or uint8"
                  << std::endl;
}
