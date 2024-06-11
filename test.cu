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

__global__ void tcMatMul(const signed char* const a,
                       const signed char* const b,
                       int* const c){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, b, 16);
    nvcuda::wmma::fill_fragment(c_frag, 0);

    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_col_major);
}

__global__ void cuMatMul(const char* const a,
                       const char* const b,
                       int* const c){
    // assert elements in a must be {-1, 0, 1}

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t row = 0; row < 16; row++){
        int accum = 0;
        for(size_t i = 0; i < 16; i++){
            if(a[row * 16 + i] == 1) {
                accum += b[i * 16 + col];
            }
        }
        c[row * 16 + col] = accum;
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

void make_binary(std::array<char, 256> &a){
    a.fill(0);

    // 1+1が右端と左端にある
    for(size_t r = 0; r < 16; r++){
        a[0 + r * 16] = 1;
        a[1 + r * 16] = 1;
        a[14 + r * 16] = 1;
        a[15 + r * 16] = 1;
    }
}

void make_I(std::array<char, 256> &b){
    b.fill(0);
    for(size_t i = 0; i < 16; i++){
        b.at(i + i * 16) = 1;
    }
}

int main(int argc, char** argv){
    char *a; cudaMalloc((void**)  &a, sizeof(char) * 16 * 16 );
    std::array<char, 256> a_ar; make_binary(a_ar);
    cudaMemcpy(a, a_ar.data(), 256 * sizeof(char), cudaMemcpyHostToDevice);

    char *b; cudaMalloc((void**)  &b, sizeof(char) * 16 * 16 );
    std::array<char, 256> b_ar; make_I(b_ar);
    cudaMemcpy(b, b_ar.data(), 256 * sizeof(char), cudaMemcpyHostToDevice);

    int *c; cudaMalloc((void**)  &c, sizeof(int) * 16 * 16 ); cudaMemset(c, 0, sizeof(int) * 16 * 16);
    std::array<int, 256> c_ar;

    float ms = measureKernel([a, b, c](){
        // 32でないとだめ
        for(size_t i = 0; i < 1000; i++){
            tcMatMul<<<1, 32>>>(( signed char * ) a, ( signed char * )  b, c);
        }
    });
    std::cout << "TensorCore Time: " << ms << "ms" << std::endl;

    cudaMemcpy(c_ar.data(), c, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar.at(0) == 1 && "what");

    ms = measureKernel([a, b, c](){
        for(size_t i = 0; i < 1000; i++){
            cuMatMul<<<1, 16>>>( a, b, c);
        }
    });
    std::cout << "CudaCore Time: " << ms << "ms" << std::endl;
    cudaMemcpy(c_ar.data(), c, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar.at(0) == 1 && "what");
    assert(c_ar.at(17) == 1 && "what");

    return 0;
}