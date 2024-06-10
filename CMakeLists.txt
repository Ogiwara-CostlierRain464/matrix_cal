cmake_minimum_required(VERSION 3.11)

project(matrix-cal)

find_package(CUDA REQUIRED)


cuda_add_executable(hello-cuda hello-cuda.cu)