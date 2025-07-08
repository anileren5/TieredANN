set(TEST_EXECUTABLES
    build_disk_index build_disk_index.cpp
    search_disk_index search_disk_index.cpp
)

function(add_tests)
    set(exec_list ${ARGV})
    list(LENGTH exec_list len)
    math(EXPR max_index "${len} - 1")

    foreach(i RANGE 0 ${max_index} 2)
        list(GET exec_list ${i} exec_name)
        math(EXPR src_index "${i} + 1")
        list(GET exec_list ${src_index} src_file)

        add_executable(${exec_name} ${src_file})
        target_link_libraries(${exec_name}
            PRIVATE
                greator
                TBB::tbb
                OpenMP::OpenMP_CXX
        )
        target_include_directories(${exec_name}
            PRIVATE
                ${PROJECT_SOURCE_DIR}/Greator/include
        )
    endforeach()
endfunction()

add_tests(${TEST_EXECUTABLES})

