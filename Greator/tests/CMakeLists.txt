# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(greator_lazy_reinsert_batch lazy_reinsert_batch.cpp )
target_link_libraries(greator_lazy_reinsert_batch ${PROJECT_NAME} -ltcmalloc)

add_executable(greator_build_memory_index build_memory_index.cpp )
target_link_libraries(greator_build_memory_index ${PROJECT_NAME} -ltcmalloc)

add_executable(greator_search_memory_index search_memory_index.cpp )
target_link_libraries(greator_search_memory_index ${PROJECT_NAME} aio -ltcmalloc)

add_executable(greator_build_disk_index build_disk_index.cpp )
target_link_libraries(greator_build_disk_index ${PROJECT_NAME} -ltcmalloc aio)

add_executable(greator_search_disk_index search_disk_index.cpp ${PROJECT_SOURCE_DIR}/src/aux_utils.cpp )
target_link_libraries(greator_search_disk_index ${PROJECT_NAME} aio -ltcmalloc)

add_executable(greator_test_concurr_merge_insert test_concurr_merge_insert.cpp)
target_link_libraries(greator_test_concurr_merge_insert ${PROJECT_NAME} aio -ltbb)

add_executable(greator_test_incremental_index test_incremental_index.cpp)
target_link_libraries(greator_test_incremental_index ${PROJECT_NAME} -ltcmalloc aio)

add_executable(greator_build_stream_merger_disk_index build_stream_merger_disk_index.cpp)
target_link_libraries(greator_build_stream_merger_disk_index ${PROJECT_NAME} -ltcmalloc aio)

add_executable(greator_overall_performance overall_performance.cpp)
target_link_libraries(greator_overall_performance ${PROJECT_NAME} aio -ltbb)