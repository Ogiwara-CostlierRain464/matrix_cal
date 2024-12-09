#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>


#define M 160
#define K 160
#define N 160

__global__ void naiveTC(
        const signed char* const X,
        const signed char* const W_mat){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::col_major> X_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> W_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0);

    for(size_t k = 0; k < K; k += 16){
        nvcuda::wmma::load_matrix_sync(X_frag, X + (k * M + blockIdx.y * 16), M);
        nvcuda::wmma::load_matrix_sync(W_frag, W_mat + (k + blockIdx.x * 16 * K), K);
        nvcuda::wmma::mma_sync(c_frag, X_frag, W_frag, c_frag);
    }
    nvcuda::wmma::store_matrix_sync(c + (blockIdx.x * 16 * M + blockIdx.y * 16), c_frag, M, nvcuda::wmma::mem_col_major);
}


int main(int argc, char** argv){
    int8_t *A_d;
    int8_t *B_d;
    int *C_d;
    cudaMalloc((void**) &A_d, 160 * 160 * sizeof(int8_t));
    cudaMalloc((void**) &B_d, 160 * 160 * sizeof(int8_t));
    cudaMalloc((void**) &C_d, 160 * 160 * sizeof(int));

    simpleFusedMultiplyAdd<<<dim3(N / 16, M / 16), 32>>>(A_d, B_d, C_d);
    cudaDeviceSynchronize();
}