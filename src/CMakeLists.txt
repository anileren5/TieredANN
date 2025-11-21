find_package(TBB REQUIRED)

set(CPP_SOURCES delete_consolidate_thread_pool.cpp
    insert_thread_pool.cpp
)

add_library(tieredann STATIC ${CPP_SOURCES})

# Add -fPIC for shared library linking (needed for Python module)
set_target_properties(tieredann PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Include dirs
target_include_directories(tieredann
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${BOOST_ROOT}
        ${PROJECT_SOURCE_DIR}/external/DiskANN/include
)
