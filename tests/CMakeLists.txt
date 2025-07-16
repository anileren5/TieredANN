# List of test executable names and their source files
set(TEST_EXECUTABLES
    build_disk_index greator/build_disk_index.cpp
    search_disk_index greator/search_disk_index.cpp
    search_memory_index diskann/search_memory_index.cpp
    batch_delete_reinsert_memory_index diskann/batch_delete_reinsert_memory_index.cpp
    concurrent_insert_delete_memory_index diskann/concurrent_insert_delete_memory_index.cpp
    tiered_index_search tieredann/tiered_index_search.cpp
    disk_index_search tieredann/disk_index_search.cpp
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
            tieredann
            TBB::tbb
            OpenMP::OpenMP_CXX
            Boost::program_options
)

    target_include_directories(${exec_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/Greator/include
            ${PROJECT_SOURCE_DIR}/DiskANN/include
    )
endforeach()
