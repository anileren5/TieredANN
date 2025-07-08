cmake_minimum_required(VERSION 3.12)
project(outer_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
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
)

# Then find dependencies
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

# Then add subdirectories
add_subdirectory(Greator)
add_subdirectory(tests)
