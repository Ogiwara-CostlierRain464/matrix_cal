#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <array>
#include <cassert>

__global__ void tcMatMul(const half* const a_ptr,
                       const half* const b_ptr,
                       half* const c_ptr){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> c_frag;


    nvcuda::wmma::fill_fragment(c_frag, __float2half(.0f));

    nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, 16);

    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, 16, nvcuda::wmma::mem_col_major);
}

__global__ void cuMatMul(const half* const a_ptr,
                       const half* const b_ptr,
                       half* const c_ptr){
    // cの一行目はbの1行…となるように計算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t i = 0; i < 16; i++){
        c_ptr[i * 16 + idx] = b_ptr[i * 16 + idx];
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

void make_I(std::array<half, 256> &b){
    b.fill(0);
    for(size_t i = 0; i < 16; i++){
        b.at(i + i * 16) = 1;
    }
}

int main(int argc, char** argv){
    half *a;
    cudaMalloc((void**)  &a, sizeof(half) * 16 * 16 );
    std::array<half, 256> a_ar;
    a_ar.fill(1);

    half *b;
    cudaMalloc((void**)  &b, sizeof(half) * 16 * 16 );
    std::array<half, 256> b_ar;
    make_I(b_ar);

    half *c;
    cudaMalloc((void**)  &c, sizeof(half) * 16 * 16 );
    cudaMemset(c, 0, sizeof(half) * 16 * 16);

    std::array<half, 256> c_ar;

    cudaMemcpy(a, a_ar.data(), 256 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_ar.data(), 256 * sizeof(half), cudaMemcpyHostToDevice);

    float ms = measureKernel([a, b, c](){
        // 32でないとだめ
        tcMatMul<<<1, 32>>>(a, b, c);
    });
    std::cout << "TensorCore Time: " << ms << "ms" << std::endl;

    cudaMemcpy(c_ar.data(), c, 256 * sizeof(half), cudaMemcpyDeviceToHost);
    assert(c_ar.at(34) == half(1) && "what");

    c_ar.fill(0); // clear

    ms = measureKernel([a, b, c](){
        cuMatMul<<<1, 16>>>(a, b, c);
    });
    std::cout << "CudaCore Time: " << ms << "ms" << std::endl;
    cudaMemcpy(c_ar.data(), c, 256 * sizeof(half), cudaMemcpyDeviceToHost);
    assert(c_ar.at(0) == half(1) && "what");
    assert(c_ar.at(17) == half(1) && "what");

    return 0;
}