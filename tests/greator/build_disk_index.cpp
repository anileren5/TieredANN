// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "greator/aux_utils.h"
#include "greator/utils.h"

template <typename T>
bool build_index(const char *dataFilePath, const char *indexFilePath,
                 const char *indexBuildParameters, greator::Metric m,
                 bool singleFile) {
  return greator::build_disk_index<T>(dataFilePath, indexFilePath,
                                      indexBuildParameters, m, singleFile);
}

int main(int argc, char **argv) {
  if (argc != 11) {
    greator::cout << "Usage: " << argv[0]
                  << " <data_type (float/int8/uint8)>  <data_file.bin>"
                     " <index_prefix_path> <R>  <L>  <B>  <M>  <T>"
                     " <similarity metric (cosine/l2) case sensitive>."
                     " <single_file_index (0/1)>"
                     " See README for more information on parameters."
                  << std::endl;
  } else {
    std::string params = std::string(argv[4]) + " " + std::string(argv[5]) +
                         " " + std::string(argv[6]) + " " +
                         std::string(argv[7]) + " " + std::string(argv[8]);
    std::string dist_metric(argv[9]);
    bool single_file_index = std::atoi(argv[10]) != 0;

    greator::Metric m =
        dist_metric == "cosine" ? greator::Metric::COSINE : greator::Metric::L2;
    if (dist_metric != "l2" && m == greator::Metric::L2) {
      greator::cout << "Metric " << dist_metric << " is not supported. Using L2"
                    << std::endl;
    }
    if (std::string(argv[1]) == std::string("float"))
      build_index<float>(argv[2], argv[3], params.c_str(), m,
                         single_file_index);
    else if (std::string(argv[1]) == std::string("int8"))
      build_index<int8_t>(argv[2], argv[3], params.c_str(), m,
                          single_file_index);
    else if (std::string(argv[1]) == std::string("uint8"))
      build_index<uint8_t>(argv[2], argv[3], params.c_str(), m,
                           single_file_index);
    else
      greator::cout << "Error. wrong file type" << std::endl;
  }
}
