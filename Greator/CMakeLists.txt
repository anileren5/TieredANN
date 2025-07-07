cmake_minimum_required(VERSION 2.8)

set(CMAKE_STANDARD 14)

project(greator)
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/tsl/include ${PROJECT_SOURCE_DIR}/include/dll ${PROJECT_SOURCE_DIR}/include/ols)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

#TBB
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
message(STATUS "TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB_LIBRARIES: ${TBB_LIBRARIES}")
function(checkEnvAndSetLocalVar env_var msg local_var)
	if (NOT EXISTS "$ENV{${env_var}}" )
		message (FATAL_ERROR ${msg})
	else()
		if ($ENV{${env_var}} MATCHES "\\$"  OR $ENV{${env_var}} MATCHES "/$" )
			set(${local_var} $ENV{${env_var}} PARENT_SCOPE)
		else()
			message(STATUS "Appending trailing backslash to ${env_var}")
			set(${local_var} "$ENV{${env_var}}\\" PARENT_SCOPE)
		endif()
	endif()
endfunction()

	
#MKL Config
add_compile_options(-m64 -Wl,--no-as-needed)
include_directories(/usr/include/mkl)
link_libraries(
	mkl_intel_lp64
	mkl_core
	mkl_sequential
	pthread
	m
	dl
)
add_definitions(-DMKL_ILP64)
include_directories(include /usr/include/mkl ${BOOST_ROOT})


#Main compiler/linker settings 
set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")
add_compile_options(-w -march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ -DUSE_AVX2)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)



