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
#include <type_traits>

#include "submodule/wmma_extension/include/wmma_extension/wmma_extension.hpp"

//#define RUN_TC
//#define RUN_CUDA
#define RUN_NEW

// X: MxK  W: KxN  C: MxN
#define D_MODEL 4096L
#define M (D_MODEL)
#define K D_MODEL
#define N (D_MODEL * 4)
#define ITER_NUM 1000

#define W_MAP_LENGTH (K / 22)

#define CALC_N_LENGTH (8L)

#define MAJOR_ROW 0
#define MAJOR_COL 1
#define X_MAJOR MAJOR_ROW
#define W_MAJOR MAJOR_COL
#define C_MAJOR MAJOR_COL

#define CAT(x, y) x ## y

#define BT_0(mat, row_dim, col_dim, row, col) mat[row * col_dim + col]
#define BT_1(mat, row_dim, col_dim, row, col) mat[col * row_dim + row]
#define BT(major) CAT(BT_, major)

__device__ signed char W_mat[K * N];
__device__ unsigned short W_map[W_MAP_LENGTH * N];
__device__ unsigned short W_map_negative[W_MAP_LENGTH * N];

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

    if(tid >= N){
        // this thread won't work for init
        return;
    }

    int col = tid;

    for(size_t row = 0; row < W_MAP_LENGTH; row++){
        BT(W_MAJOR) (W_map, W_MAP_LENGTH , N, row, col) = row;
    }
    for(size_t row = 0; row < W_MAP_LENGTH; row++){
        BT(W_MAJOR) (W_map_negative ,W_MAP_LENGTH ,N, row, col) = row;
    }

    for(size_t row = 0; row < K; row++){
        if(row < W_MAP_LENGTH){
            BT(W_MAJOR) (W_mat, K, N, row, col) = 1;
        }else if(W_MAP_LENGTH <= row && row < W_MAP_LENGTH * 2){
            BT(W_MAJOR) (W_mat, K, N, row, col) = -1;
        }else{
            BT(W_MAJOR) (W_mat, K, N, row, col) = 0;
        }
    }
}

/**
 * ここはroとcol orderで固定にする良さそう
 */
__global__ void tcMatMul(const signed char* const X,
                       int* const c){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, std::conditional_t<X_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> X_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, std::conditional_t<W_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> W_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0);

#pragma unroll
    for(size_t k = 0; k < K; k += 16){
        if constexpr(X_MAJOR == MAJOR_ROW){
            nvcuda::wmma::load_matrix_sync(X_frag, X + (blockIdx.y * K * 16 + k), K);
        }else{
            nvcuda::wmma::load_matrix_sync(X_frag, X + (k * M + blockIdx.y * 16), M);
        }

        if constexpr(W_MAJOR == MAJOR_ROW){
            nvcuda::wmma::load_matrix_sync(W_frag, W_mat + ( k * N + blockIdx.x * 16), N);
        }else{
            nvcuda::wmma::load_matrix_sync(W_frag, W_mat + ( k + blockIdx.x * 16 * K), K);
        }
        nvcuda::wmma::mma_sync(c_frag, X_frag, W_frag, c_frag);
    }

    if constexpr(C_MAJOR == MAJOR_ROW){
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.y * 16 * N + blockIdx.x * 16), c_frag, N, nvcuda::wmma::mem_row_major);
    }else{
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.x * 16 * M + blockIdx.y * 16), c_frag, M, nvcuda::wmma::mem_col_major);
    }
}

__global__ void cuMatMul(const char* const X , int* const C){
    // CUDA内では2配列として使うことはできない。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_col = (tid / M) * CALC_N_LENGTH;
    int row = tid % M;

#pragma unroll
    for(size_t col = start_col; col < start_col + CALC_N_LENGTH; col++){
        int accum = 0;
#pragma unroll
        for(size_t i = 0; i < W_MAP_LENGTH; i++){
            auto idx = BT(W_MAJOR) (W_map, W_MAP_LENGTH, N, i, col);
            accum += BT(X_MAJOR) (X, M, K, row, idx);
        }
        // indexを負の値にする方法では、なぜかパフォーマンスが劣化した
        // このため、別のmapとし作成することにより、パフォーマンスの劣化を抑える。
#pragma unroll
        for(size_t i = 0; i < W_MAP_LENGTH; i++){
            auto idx = BT(W_MAJOR) (W_map_negative, W_MAP_LENGTH, N, i, col);
            accum += -BT(X_MAJOR) (X, M, K, row, idx);
        }
        BT(C_MAJOR) (C, M, N, row, col) = accum;
    }
}

__global__ void newMatMul(const signed char* const X, int* const c){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, std::conditional_t<X_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> X_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, std::conditional_t<W_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> W_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0);

    int land_id = mtk::wmma::detail::common::get_lane_id();

    for(size_t k = 0; k < W_MAP_LENGTH; k++){
        int col_idx = BT(W_MAJOR) (W_map, W_MAP_LENGTH, N, k, blockIdx.x * 16 + (land_id % 16));

        mtk::wmma::foreach_ij<decltype(X_frag)>([&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned i, const unsigned j){
            for(unsigned f = 0; f < fragment_index_count; f++){
                X_frag.x[frag_index_list[f]] = BT(X_MAJOR) (X, M, K,  blockIdx.y * 16 + f , col_idx);
            }
        });

        mtk::wmma::detail::sm_75::make_identity_matrix(W_frag);
        __syncwarp();

        nvcuda::wmma::mma_sync(c_frag, X_frag, W_frag, c_frag);


        col_idx = BT(W_MAJOR) (W_map_negative, W_MAP_LENGTH, N, k, blockIdx.x * 16 + (land_id % 16));

        mtk::wmma::foreach_ij<decltype(X_frag)>([&](const unsigned* frag_index_list, const unsigned fragment_index_count, const unsigned i, const unsigned j){
            for(unsigned f = 0; f < fragment_index_count; f++){
                X_frag.x[frag_index_list[f]] = -BT(X_MAJOR) (X, M, K,  blockIdx.y * 16 + f , col_idx);
            }
        });
        __syncwarp();

        nvcuda::wmma::mma_sync(c_frag, X_frag, W_frag, c_frag);
    }

    if constexpr(C_MAJOR == MAJOR_ROW){
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.y * 16 * N + blockIdx.x * 16), c_frag, N, nvcuda::wmma::mem_row_major);
    }else{
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.x * 16 * M + blockIdx.y * 16), c_frag, M, nvcuda::wmma::mem_col_major);
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

void make_J(std::array<char, M * K> *X){
    X->fill(1);
}

int main(int argc, char** argv){

    static_assert(M % 16 == 0 && "mod 16 should be 0");
    static_assert(K % 16 == 0 && "mod 16 should be 0");
    static_assert(N % 16 == 0 && "mod 16 should be 0");
    static_assert(K < 65536 && "K should be fit in the maximum of short");

    char *X_d;
    cudaMalloc((void**) &X_d, sizeof(char) * M * K);
    auto *X_ar = new std::array<char, M * K>(); make_J(X_ar);
    cudaMemcpy(X_d, X_ar->data(), sizeof(char) * M * K, cudaMemcpyHostToDevice);

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(int) * M * N ); cudaMemset(c_d, 0, sizeof(int) * M * N);
    auto c_ar = new std::array<int, N * 1>(); // store only first row

    prepareW<<< N / 16, 16>>>();
    cudaDeviceSynchronize();

    std::cout << "Start: " << "M=" << M << " K=" << K << " N=" << N << " ITER=" << ITER_NUM << " W_MAP_LENGTH=" << W_MAP_LENGTH << " CALC_N_LENGTH=" << CALC_N_LENGTH << std::endl;

    float ms = 0;

#ifdef RUN_TC
    ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((tcMatMul<<< dim3(N / 16, M / 16) , 32>>>((signed char *) X_d, c_d)));
        }
    });
    std::cout << "TensorCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");
#endif

#ifdef RUN_CUDA
    ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((cuMatMul<<< N * M / (CALC_N_LENGTH * 32), 32 >>>(X_d, c_d)));
        }
    });
    std::cout << "CudaCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");
#endif

#ifdef RUN_NEW
    ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((newMatMul<<< dim3(N / 16, M / 16) , 32>>>((signed char *) X_d, c_d)));
        }
    });
    std::cout << "New Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");
#endif

    return 0;
}