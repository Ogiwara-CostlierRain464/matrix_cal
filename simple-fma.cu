#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleFusedMultiplyAdd(
        const int8_t* const A,
        const int8_t* const B,
        int* const C){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, int8_t, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, int8_t, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::load_matrix_sync(a_frag, A, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, B, 16);
    nvcuda::wmma::load_matrix_sync(c_frag, C, 16);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    nvcuda::wmma::store_matrix_sync(C, c_frag, 16, nvcuda::wmma::mem_col_major);
}

int main(int argc, char** argv){
    int8_t *A_d;
    int8_t *B_d;
    int *C_d;
    cudaMalloc((void**) &A_d, 16 * 16 * sizeof(int8_t));
    cudaMalloc((void**) &B_d, 16 * 16 * sizeof(int8_t));
    cudaMalloc((void**) &C_d, 16 * 16 * sizeof(int));

    simpleFusedMultiplyAdd<<<1, 32>>>(A_d, B_d, C_d);
    cudaDeviceSynchronize();
}