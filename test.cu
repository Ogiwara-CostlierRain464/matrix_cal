#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>

__global__ void tcMatMul(const half* const a_ptr,
                       const half* const b_ptr,
                       half* const c_ptr){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> c_frag;

    for(size_t _i = 0; _i < 1'000'000'0; _i++){
        nvcuda::wmma::fill_fragment(c_frag, __float2half(.0f));

        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, 16);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, 16);

        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, 16, nvcuda::wmma::mem_col_major);
    }
}

__global__ void cuMatMul(const half* const a_ptr,
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

float measureKernel(std::function<void(void)> fn){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    fn();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main(int argc, char** argv){
    half *a;
    cudaMalloc((void**)  &a, 16 * 16 );
    half *b;
    cudaMalloc((void**)  &b, 16 * 16 );
    half *c;
    cudaMalloc((void**)  &c, 16 * 16 );

    float ms = measureKernel([a, b, c](){
        tcMatMul<<<4, 4>>>(a, b, c);
    });
    std::cout << "TensorCore Time: " << ms << "ms" << std::endl;
    ms = measureKernel([a, b, c](){
        cuMatMul<<<1, 16>>>(a, b, c);
    });
    std::cout << "CudaCore Time: " << ms << "ms" << std::endl;

    return 0;
}