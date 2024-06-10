cmake_minimum_required(VERSION 3.11)

project(matrix-cal CUDA)

#find_package(CUDA REQUIRED)

#add_compile_options(-arch=sm_70 -DUSE_TENSORCORE -std=c++11)


set(CMAKE_CUDA_ARCHITECTURES 86)

#cuda_add_executable(hello-cuda hello-cuda.cu)
add_executable(tensor_core tensor_core.cu)


#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_70 -DUSE_TENSORCORE -std=c++11")


list(APPEND CUDA_NVCC_FLAGS "-std=c++11")