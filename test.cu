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
#include <set>


#define M 12288L
#define K 12288L
#define N (M * 4L)
#define ITER_NUM 1000

#define W_MAP_LENGTH (K / 10)

#define CALC_M_LENGTH (8L)

__device__ signed char W_mat[M * K];
// TODO X map should support dynamic length
// I just fill this matrix with index num
__device__ short W_map[W_MAP_LENGTH * M];
__device__ short W_map_negative[W_MAP_LENGTH * M];

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

/**
 * Prepare both W_mat and W_map before the measurement.
 */
__global__ void prepareW(){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= M){
        // this thread won't work for init
        return;
    }

    int row = tid;

    for(size_t col = 0; col < W_MAP_LENGTH; col++){
        W_map[row * W_MAP_LENGTH + col] = col;
    }
    for(size_t col = 0; col < W_MAP_LENGTH; col++){
        W_map_negative[row * W_MAP_LENGTH + col] = col;
    }

    for(size_t col = 0; col < K; col++){
        if(col < W_MAP_LENGTH){
            W_mat[row * K + col] = 1;
        }else if(W_MAP_LENGTH <= col && col < W_MAP_LENGTH * 2){
            W_mat[row * K + col] = -1;
        }else{
            W_mat[row * K + col] = 0;
        }
    }
}

__global__ void tcMatMul(const signed char* const X,
                       int* const c){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> W_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> X_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0);

    for(size_t k = 0; k < K; k += 16){
        nvcuda::wmma::load_matrix_sync(W_frag, W_mat + (blockIdx.y * K * 16 + k), K);
        nvcuda::wmma::load_matrix_sync(X_frag, X + ( k * N + blockIdx.x * 16) , N);
        nvcuda::wmma::mma_sync(c_frag, W_frag, X_frag, c_frag);
    }

    nvcuda::wmma::store_matrix_sync(c + (blockIdx.y * N * 16 + blockIdx.x * 16), c_frag, N, nvcuda::wmma::mem_row_major);
}

__device__ __forceinline__ short make_sign(short x){
    return  (2 * (short)(x > 0) - 1);
}

// <<< N * M /  CALC_M_LENGTH / 32, 32  >>>
__global__ void cuMatMul2(const char* const X, int* const c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_row = (tid / N) * CALC_M_LENGTH;
    int col = tid % N;

#pragma unroll
    for(size_t row = start_row; row < start_row + CALC_M_LENGTH; row++){
        int accum = 0;
#pragma unroll
        for(size_t i = 0; i < W_MAP_LENGTH; i++){
            accum += X[W_map[row * W_MAP_LENGTH + i]];
        }
        // indexを負の値にする方法では、なぜかパフォーマンスが劣化した
        // このため、別のmapとし作成することにより、パフォーマンスの劣化を抑える。
#pragma unroll
        for(size_t i = 0; i < W_MAP_LENGTH; i++){
            accum += -X[W_map_negative[row * W_MAP_LENGTH + i]];
        }
        c[row * N + col] = accum;
    }
}


float measureKernel(std::function<void(void)> fn){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    fn();

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

void make_J(std::array<char, K * N> *X){
    X->fill(1);
}

void make_I(std::array<char, K * N> *X){
    for(size_t row = 0; row < K; row++){
        for(size_t col = 0; col < N; col++){
            if(row == col) {
                X->at(row * N + col) = 1;
            }
        }
    }
}

/**
 * Calc matmul of W(MxK) and X(KxN), where W is ternary matrix and X is 8-bit matrix.
 * Since X is 8-bit, we need to implement W as a 8-bit matrix due to restriction of wmma.
 * W is prepared before the performance measure.
 */
int main(int argc, char** argv){

    static_assert(M % 16 == 0 && "mod 16 should be 0");
    static_assert(K % 16 == 0 && "mod 16 should be 0");
    static_assert(N % 16 == 0 && "mod 16 should be 0");
    static_assert(K < (65536 / 2) && "K should be fit in the maximum of short");

    char *X_d;
    cudaMalloc((void**)  &X_d, sizeof(char) * K * N );
    auto *X_ar = new std::array<char, K * N>(); make_J(X_ar);
    cudaMemcpy(X_d, X_ar->data(), K * N * sizeof(char), cudaMemcpyHostToDevice);

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(int) * M * N ); cudaMemset(c_d, 0, sizeof(int) * M * N);
    auto c_ar = new std::array<int, N * 1>(); // store only first row

    prepareW<<< M / 16, 16>>>();
    cudaDeviceSynchronize();

    std::cout << "Start: " << "M=" << M << " K=" << K << " N=" << N << " ITER=" << ITER_NUM << " W_MAP_LENGTH=" << W_MAP_LENGTH << " CALC_M_LENGTH=" << CALC_M_LENGTH << std::endl;

    float ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((tcMatMul<<< dim3(N / 16, M / 16) , 32>>>(( signed char * )  X_d, c_d)));
        }
    });
    std::cout << "TensorCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");

    ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((cuMatMul2<<< N * M / (CALC_M_LENGTH * 32), 32 >>>(X_d, c_d)));
        }
    });
    std::cout << "CudaCore2 Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");


    return 0;
}