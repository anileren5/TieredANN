#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <spdlog/spdlog.h>
#include <string>
#include "diskann/utils.h"

template <typename T, typename TagT = uint32_t>
void calculate_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    for (int32_t i = 0; i < query_num; i++) {
        std::set<uint32_t> groundtruth_closest_neighbors;
        std::set<uint32_t> calculated_closest_neighbors;
        for (int32_t j = 0; j < K; j++) {
            calculated_closest_neighbors.insert(*(query_result_tags.data() + i * K + j));
            groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
        }
        uint32_t matching_neighbors = 0;
        for (uint32_t x : calculated_closest_neighbors) if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
        double recall = matching_neighbors / (double)K;
        total_recall += recall;
    }
    double average_recall = total_recall / (query_num);
    spdlog::info("{{\"event\": \"recall\", \"K\": {}, \"recall\": {}, \"type\": \"all\"}}", K, average_recall);
}

template <typename T, typename TagT = uint32_t>
void calculate_hit_recall(size_t K, TagT* groundtruth_ids, std::vector<TagT>& query_result_tags, 
                         const std::vector<bool>& hit_results, size_t query_num, size_t groundtruth_dim) {
    double total_recall = 0.0;
    size_t hit_count = 0;
    for (int32_t i = 0; i < query_num; i++) {
        if (hit_results[i]) {
            std::set<uint32_t> groundtruth_closest_neighbors;
            std::set<uint32_t> calculated_closest_neighbors;
            for (int32_t j = 0; j < K; j++) {
                calculated_closest_neighbors.insert(*(query_result_tags.data() + i * K + j));
                groundtruth_closest_neighbors.insert(*(groundtruth_ids + i * groundtruth_dim + j));
            }
            uint32_t matching_neighbors = 0;
            for (uint32_t x : calculated_closest_neighbors) if (groundtruth_closest_neighbors.count(x - 1)) matching_neighbors++;
            double recall = matching_neighbors / (double)K;
            total_recall += recall;
            hit_count++;
        }
    }
    if (hit_count > 0) {
        double average_recall = total_recall / hit_count;
        spdlog::info("{{\"event\": \"recall\", \"K\": {}, \"recall\": {}, \"type\": \"cache_hits\", \"hit_count\": {}}}", K, average_recall, hit_count);
    } else {
        spdlog::info("{{\"event\": \"recall\", \"K\": {}, \"recall\": null, \"type\": \"cache_hits\", \"hit_count\": 0}}", K);
    }
}

template <typename T>
void load_aligned_binary_data(const std::string& file_path, T*& data, size_t& num, size_t& dim, size_t& aligned_dim) {
    diskann::load_aligned_bin<T>(file_path, data, num, dim, aligned_dim);
}

template <typename TagT>
void load_ground_truth_data(const std::string& file_path, TagT*& ids, float*& dists, size_t& num, size_t& dim) {
    diskann::load_truthset(file_path, ids, dists, num, dim);
}

#endif 