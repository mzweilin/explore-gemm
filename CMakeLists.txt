cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set CUDA compiler before project() if not already set
# INSTALL CUDA 12.8 on Debian 12 with:
# wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-8
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS /usr/local/cuda-12.8/bin/nvcc)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
        message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    endif()
endif()

project(explore_gemm CUDA CXX)

# Set CUDA architecture if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -Xcompiler=-fno-gnu-unique")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")

# Find CUDA kernel source files
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/cuda/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/cuda/*.cuh")

# Set up paths for dependencies
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /usr/local/cuda-12.8)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third-party)

# Tests
add_executable(test_gemm_naive ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_naive.cpp ${gpu_source_files})
target_link_libraries(test_gemm_naive "${TORCH_LIBRARIES}")

add_executable(test_gemm_coalesce ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_coalesce.cpp ${gpu_source_files})
target_link_libraries(test_gemm_coalesce "${TORCH_LIBRARIES}")

add_executable(test_gemm_shared_mem ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_shared_mem.cpp ${gpu_source_files})
target_link_libraries(test_gemm_shared_mem "${TORCH_LIBRARIES}")

add_executable(test_gemm_blocktiling1d ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_blocktiling1d.cpp ${gpu_source_files})
target_link_libraries(test_gemm_blocktiling1d "${TORCH_LIBRARIES}")

add_executable(test_gemm_blocktiling2d ${CMAKE_SOURCE_DIR}/cuda/tests/test_gemm_blocktiling2d.cpp ${gpu_source_files})
target_link_libraries(test_gemm_blocktiling2d "${TORCH_LIBRARIES}")

enable_testing()

add_test(NAME test_gemm_naive COMMAND test_gemm_naive)
add_test(NAME test_gemm_coalesce COMMAND test_gemm_coalesce)
add_test(NAME test_gemm_shared_mem COMMAND test_gemm_shared_mem)
add_test(NAME test_gemm_blocktiling1d COMMAND test_gemm_blocktiling1d)
add_test(NAME test_gemm_blocktiling2d COMMAND test_gemm_blocktiling2d)

# NOTES
# Build with: cmake -B build
# Or override defaults: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/path/to/nvcc
# Then: cmake --build build
# Run tests: ctest --test-dir build