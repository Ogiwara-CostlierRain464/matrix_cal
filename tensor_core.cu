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

int main(int argc, char** argv){
    half *a;
    cudaMalloc((void**)  &a, 16 * 16 );
    half *b;
    cudaMalloc((void**)  &b, 16 * 16 );
    half *c;
    cudaMalloc((void**)  &c, 16 * 16 );

    dim3 grid(1);
    dim3 block(1);

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