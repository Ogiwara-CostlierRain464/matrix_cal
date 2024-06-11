cmake_minimum_required(VERSION 3.11)

project(matrix-cal CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(test test.cu)

#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_75")