# List of test executable names and their source files
set(TEST_EXECUTABLES
    build_disk_index greator/build_disk_index.cpp
    search_disk_index greator/search_disk_index.cpp
    qvcache_search qvcache/qvcache_search.cpp
    qvcache_split_search qvcache/qvcache_split_search.cpp
    qvcache_split_search_noisy qvcache/qvcache_split_search_noisy.cpp
    qvcache_split_search_bruteforce_backend qvcache/qvcache_split_search_bruteforce_backend.cpp
    qvcache_benchmark qvcache/qvcache_benchmark.cpp
    cache_pressure qvcache/cache_pressure.cpp
    backend_cache_pressure qvcache/backend_cache_pressure.cpp
    backend_benchmark qvcache/backend_benchmark.cpp
    diskann_split_search diskann/diskann_split_search.cpp
)

# Compute length of the list
list(LENGTH TEST_EXECUTABLES len)
math(EXPR max_index "${len} - 1")

# Loop through the list, two items at a time
foreach(i RANGE 0 ${max_index} 2)
    list(GET TEST_EXECUTABLES ${i} exec_name)
    math(EXPR src_index "${i} + 1")
    list(GET TEST_EXECUTABLES ${src_index} src_file)

    add_executable(${exec_name} ${src_file})

    target_link_libraries(${exec_name}
        PRIVATE
            greator
            diskann
            qvcache
            TBB::tbb
            OpenMP::OpenMP_CXX
            Boost::program_options
            spdlog::spdlog
)

    target_include_directories(${exec_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/backends
            ${PROJECT_SOURCE_DIR}/backends/Greator/include
            ${PROJECT_SOURCE_DIR}/external/DiskANN/include
    )
endforeach()

# Special handling for compute_groundtruth (simpler, only needs OpenMP)
add_executable(compute_groundtruth compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth
    PRIVATE
        OpenMP::OpenMP_CXX
)
target_include_directories(compute_groundtruth
    PRIVATE
        ${PROJECT_SOURCE_DIR}/external/DiskANN/include
)
