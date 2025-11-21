cmake_minimum_required(VERSION 3.12)

# Find pybind11
find_package(pybind11 REQUIRED)

# Python module - use a different target name to avoid conflict with static library
# but the Python import name will still be "tieredann"
pybind11_add_module(tieredann_python_module 
    tieredann_python.cpp
)

# Set the Python module name to "tieredann" for import
set_target_properties(tieredann_python_module PROPERTIES
    OUTPUT_NAME "tieredann"
    PREFIX ""
)

# Link libraries - link to the static library tieredann
target_link_libraries(tieredann_python_module PRIVATE
    tieredann
    diskann
    spdlog::spdlog
    ${TBB_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    pthread
    m
    dl
)

# Find numpy include directory
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find numpy via Python
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE NUMPY_FOUND_RESULT
)

# Include directories
target_include_directories(tieredann_python_module PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/backends
    ${PROJECT_SOURCE_DIR}/python
    ${PROJECT_SOURCE_DIR}/external/DiskANN/include
    ${BOOST_ROOT}
    ${EIGEN3_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

if(NUMPY_INCLUDE_DIR AND NUMPY_FOUND_RESULT EQUAL 0 AND EXISTS "${NUMPY_INCLUDE_DIR}")
    target_include_directories(tieredann_python_module PRIVATE ${NUMPY_INCLUDE_DIR})
    message(STATUS "Found numpy include directory: ${NUMPY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Could not find numpy include directory. Please install numpy: pip install numpy")
endif()

# Compile definitions
target_compile_definitions(tieredann_python_module PRIVATE
    VERSION_INFO=${PROJECT_VERSION}
)

# Set output directory
set_target_properties(tieredann_python_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python
)

