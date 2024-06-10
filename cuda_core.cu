#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matMul(const half* const a_ptr,
                       const half* const b_ptr,
                       half* const c_ptr){
    // cの一行目はbの1行…となるように計算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t _i = 0; _i < 1'000'000'0; _i++){
        for(size_t i = 0; i < 16; i++){
            c_ptr[i * 16 + idx] = b_ptr[i * 16 + idx] + half(4);
        }
    }
}

int main(int argc, char** argv){
    half *a;
    cudaMalloc((void**)  &a, 16 * 16 );
    half *b;
    cudaMalloc((void**)  &b, 16 * 16 );
    half *c;
    cudaMalloc((void**)  &c, 16 * 16 );

    dim3 grid(4);
    dim3 block(4); // 16スレッド並列

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matMul<<<grid, block>>>(a, b, c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    return 0;
}