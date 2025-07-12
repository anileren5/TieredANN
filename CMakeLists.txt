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

# Then add subdirectories
add_subdirectory(Greator)
add_subdirectory(DiskANN)
add_subdirectory(tests)
add_subdirectory(src)
