# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_stitched_index build_stitched_index.cpp)
target_link_libraries(build_stitched_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(search_memory_index search_memory_index.cpp)
target_link_libraries(search_memory_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(range_search_disk_index range_search_disk_index.cpp)
target_link_libraries(range_search_disk_index ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(test_streaming_scenario test_streaming_scenario.cpp)
target_link_libraries(test_streaming_scenario ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(test_insert_deletes_consolidate test_insert_deletes_consolidate.cpp)
target_link_libraries(test_insert_deletes_consolidate ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(batch_delete_reinsert batch_delete_reinsert.cpp)
target_link_libraries(batch_delete_reinsert ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_insert_delete concurrent_insert_delete.cpp)
target_link_libraries(concurrent_insert_delete ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_insert_delete_non_uniform_chunks concurrent_insert_delete_non_uniform_chunks.cpp)
target_link_libraries(concurrent_insert_delete_non_uniform_chunks ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search concurrent_search.cpp)
target_link_libraries(concurrent_search ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search_only_delete concurrent_search_only_delete.cpp)
target_link_libraries(concurrent_search_only_delete ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search_only_insert concurrent_search_only_insert.cpp)
target_link_libraries(concurrent_search_only_insert ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search_only_delete_fixed_size concurrent_search_only_delete_fixed_size.cpp)
target_link_libraries(concurrent_search_only_delete_fixed_size ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search_only_insert_fixed_size concurrent_search_only_insert_fixed_size.cpp)
target_link_libraries(concurrent_search_only_insert_fixed_size ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search_only_delete_minibatch_consolidate concurrent_search_only_delete_minibatch_consolidate.cpp)
target_link_libraries(concurrent_search_only_delete_minibatch_consolidate ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(concurrent_search_windowed_insert_delete concurrent_search_windowed_insert_delete.cpp)
target_link_libraries(concurrent_search_windowed_insert_delete ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(only_search only_search.cpp)
target_link_libraries(only_search ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

if (NOT MSVC)
    install(TARGETS build_memory_index
            build_stitched_index
            search_memory_index
            build_disk_index
            search_disk_index
            range_search_disk_index
            test_streaming_scenario
            test_insert_deletes_consolidate
            batch_delete_reinsert
            concurrent_insert_delete
            concurrent_search
            concurrent_search_only_delete
            concurrent_search_only_insert
            concurrent_search_only_delete_fixed_size
            concurrent_search_only_insert_fixed_size
            concurrent_search_only_delete_minibatch_consolidate
            concurrent_search_windowed_insert_delete
            only_search
            RUNTIME
    )
endif()
