cmake_minimum_required(VERSION 3.12)
project(tieredann LANGUAGES CXX)

set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler options first!
add_compile_options(
    -w
    -march=native
    -Wall
    -mavx
    -fno-builtin-malloc
    -fno-builtin-calloc
    -fno-builtin-realloc
    -fno-builtin-free
    -fopenmp
    -fopenmp-simd
    -funroll-loops
    -Wfatal-errors
    -DUSE_ACCELERATED_PQ
    -DUSE_AVX2
    -Ofast
    -march=native 
    -mtune=native
    -DNDEBUG
)

# Then find dependencies
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost COMPONENTS program_options)
find_package(spdlog REQUIRED)
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense PATHS /usr/include/eigen3 /usr/local/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Eigen3. Please install libeigen3-dev or set EIGEN3_INCLUDE_DIR.")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# Then add subdirectories
add_subdirectory(backends/Greator)
add_subdirectory(external/DiskANN)
add_subdirectory(tests)
add_subdirectory(src)
