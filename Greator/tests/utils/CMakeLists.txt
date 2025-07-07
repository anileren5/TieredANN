# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(greator_uint8_to_float uint8_to_float.cpp)
target_link_libraries(greator_uint8_to_float ${PROJECT_NAME})

add_executable(greator_fvecs_to_bin fvecs_to_bin.cpp)
target_link_libraries(greator_fvecs_to_bin ${PROJECT_NAME})

add_executable(greator_bin_to_fvecs bin_to_fvecs.cpp)
target_link_libraries(greator_bin_to_fvecs ${PROJECT_NAME})

add_executable(greator_bin_to_tsv bin_to_tsv.cpp)
target_link_libraries(greator_bin_to_tsv ${PROJECT_NAME})

add_executable(greator_normalize_data normalize_data.cpp)
target_link_libraries(greator_normalize_data ${PROJECT_NAME})

add_executable(greator_tsv_to_bin tsv_to_bin.cpp)
target_link_libraries(greator_tsv_to_bin ${PROJECT_NAME})

add_executable(greator_ivecs_to_bin ivecs_to_bin.cpp)
target_link_libraries(greator_ivecs_to_bin ${PROJECT_NAME})

add_executable(greator_int8_to_float int8_to_float.cpp)
target_link_libraries(greator_int8_to_float ${PROJECT_NAME})

add_executable(greator_uint32_to_uint8 uint32_to_uint8.cpp)
target_link_libraries(greator_uint32_to_uint8 ${PROJECT_NAME})

add_executable(greator_calculate_recall calculate_recall.cpp)
target_link_libraries(greator_calculate_recall ${PROJECT_NAME} aio -ltcmalloc)

add_executable(greator_compute_groundtruth compute_groundtruth.cpp)
target_link_libraries(greator_compute_groundtruth ${PROJECT_NAME} aio)

add_executable(greator_seed_index_merger seed_index_merger.cpp)
target_link_libraries(greator_seed_index_merger ${PROJECT_NAME} -ltcmalloc)

add_executable(greator_generate_pq generate_pq.cpp)
target_link_libraries(greator_generate_pq ${PROJECT_NAME} -ltcmalloc)

add_executable(greator_partition_data partition_data.cpp)
target_link_libraries(greator_partition_data ${PROJECT_NAME} -ltcmalloc)

add_executable(greator_partition_with_ram_budget partition_with_ram_budget.cpp)
target_link_libraries(greator_partition_with_ram_budget ${PROJECT_NAME} -ltcmalloc)

add_executable(greator_merge_shards merge_shards.cpp)
target_link_libraries(greator_merge_shards ${PROJECT_NAME} -ltcmalloc aio)

add_executable(greator_create_disk_layout create_disk_layout.cpp)
target_link_libraries(greator_create_disk_layout ${PROJECT_NAME} aio -ltcmalloc)
