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

//#define NDEBUG

//#define TRANSFER

//#define RUN_TC
//#define RUN_CUDA
#define RUN_NEW_2

// X: MxK  W: KxN  C: MxN
#define D_MODEL 768L
#define BATCH_SIZE 500000L // for real-time inference
#define M BATCH_SIZE
#define K (D_MODEL * 4)
#define N (D_MODEL)
#define ITER_NUM 100

#define NZ_RATIO 10
#define W_MAP_LENGTH (K / NZ_RATIO)

#define CALC_N_LENGTH (768L)

#define MAJOR_ROW 0
#define MAJOR_COL 1
#define X_MAJOR MAJOR_COL
#define W_MAJOR MAJOR_COL
#define C_MAJOR MAJOR_COL

#define CAT(x, y) x ## y

#define BT_0(mat, row_dim, col_dim, row, col) mat[row * col_dim + col]
#define BT_1(mat, row_dim, col_dim, row, col) mat[col * row_dim + row]
#define BT(major) CAT(BT_, major)

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

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

__global__ void prepareW_mat(char* const W_mat){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N){
        // this thread won't work for init
        return;
    }

    int col = tid;

    for(size_t row = 0; row < K; row++){
        if(row % NZ_RATIO == 0){
            int sign = (row / NZ_RATIO) % 2 == 0 ? 1 : -1;
                    BT(W_MAJOR) (W_mat, K, N, row, col) = sign;
        }else{
                    BT(W_MAJOR) (W_mat, K, N, row, col) = 0;
        }
    }
}

__global__ void prepareW_map(char* const W_map){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N){
        // this thread won't work for init
        return;
    }

    int col = tid;

    for(int row = 0; row < W_MAP_LENGTH; row++){
        int sign = row % 2 == 0 ? 1 : -1;
        BT(W_MAJOR) (W_map, W_MAP_LENGTH , N, row, col) = sign * NZ_RATIO; // delta encoding
    }
}

/**
 * ここはroとcol orderで固定にする良さそう
 */
__global__ void tcMatMul(
        const signed char* const X,
        const signed char* const W_mat,
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

// non divergence
__device__ __forceinline__ char sign(char x){
    //return (x >> 7) | (x != 0); こっちは0の時に0を返してしまう
    //return (x > 0) - (x < 0); こっちは遅い
    return 1 | (x >> 7);
}

__device__ __forceinline__ char abs(char x){
    int mask = x >> 7; // get the sign bit: 0 for positive, -1 for negative
    return (x + mask) ^ mask;
}

__global__ void cuMatMul(
        const char* const X,
        const char* const W_map,
        int* const C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_col = (tid / M) * CALC_N_LENGTH;
    int row = tid % M;

#pragma unroll
    for(int col = start_col; col < start_col + CALC_N_LENGTH; col++){ // NOTE: don't use size_t
        int accum = 0;
        int idx = 0;
#pragma unroll
        for(int i = 0; i < W_MAP_LENGTH; i++){
            int idx_delta = BT(W_MAJOR) (W_map, W_MAP_LENGTH, N, i, col);
            idx += abs(idx_delta);
            accum += sign(idx_delta) *  BT(X_MAJOR) (X, M, K, row, idx); // delta encoding
        }
        BT(C_MAJOR) (C, M, N, row, col) = accum;
    }
}

// assert uint8_t, col major, sm80
__device__ void make_map_a(unsigned tid, unsigned *i_map, unsigned *j_map){
    auto div_4 = tid / 4;
    auto mod_4 = tid % 4;

    for(unsigned i = 0; i < 4; i++){
        i_map[i] = div_4;
        j_map[i] = mod_4 * 4 + i;
    }
    for(unsigned i = 0; i < 4; i++){
        i_map[i + 4] = div_4 + 8;
        j_map[i + 4] = mod_4 * 4 + i;
    }
}

// assert uint8_t, col major, sm80
__device__ void make_map_b(unsigned tid, unsigned *i_map, unsigned *j_map){
    auto div_4 = tid / 4; // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
    auto mod_4 = tid % 4; // 0 1 2 3 0 1 2 3 0 1 2 3

    for(unsigned i = 0; i < 4; i++){
        i_map[i] = mod_4 * 4 + i;
        j_map[i] = div_4;
    }
    for(unsigned i = 0; i < 4; i++){
        i_map[i + 4] = mod_4 * 4 + i;
        j_map[i + 4] = div_4 + 8;
    }
}


// no use shared memory
__global__ void newMatMul2(
        const signed char* const X,
        const signed char* const W_map,
        int* const c){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, std::conditional_t<X_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> M_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, std::conditional_t<W_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> I_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0);
    nvcuda::wmma::fill_fragment(I_frag, 0);

    int lane_id = mtk::wmma::detail::common::get_lane_id();

    unsigned a_i_map[8];
    unsigned a_j_map[8];
    make_map_a(lane_id, a_i_map, a_j_map);

    unsigned b_i_map[8];
    unsigned b_j_map[8];
    make_map_b(lane_id, b_i_map, b_j_map);

#pragma unroll
    for(unsigned f = 0; f < 8; f++){
        if constexpr(W_MAJOR == MAJOR_COL){
            if(b_i_map[f] == b_j_map[f]){
                I_frag.x[f] = 1;
            }
        }else{
            if(a_i_map[f] == a_j_map[f]){
                I_frag.x[f] = 1;
            }
        }
    }

    int idx[8] = {0};

#pragma unroll
    for(unsigned k = 0; k < W_MAP_LENGTH; k++){
#pragma unroll
        for(unsigned f = 0; f < 8; f++){
            unsigned i, j;
            if constexpr(X_MAJOR == MAJOR_COL){
                i = a_i_map[f];
                j = a_j_map[f];
            }else{
                i = b_i_map[f];
                j = b_j_map[f];
            }
            assert(0 <= k && k < W_MAP_LENGTH);
            assert(0 <= blockIdx.x * 16 + j && blockIdx.x * 16 + j < N);
            auto col_idx_delta = BT(W_MAJOR) (W_map, W_MAP_LENGTH, N, k, blockIdx.x * 16 + j);
            idx[f] += abs(col_idx_delta);

            assert(0 <= blockIdx.y * 16 + i && blockIdx.y * 16 + i < M);
            assert(0 <= idx[f] && idx[f] < K);
            M_frag.x[f] = sign(col_idx_delta) * BT(X_MAJOR) (X, M, K, blockIdx.y * 16 + i, idx[f]); // delta encoding
        }
        __syncwarp();

        nvcuda::wmma::mma_sync(c_frag, M_frag, I_frag, c_frag);
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

    static_assert(NZ_RATIO < 128 && "NZ_RATIO must be smaller than int8 indexing size");

    char *X_d;
    cudaMalloc((void**) &X_d, sizeof(char) * M * K);
    auto *X_ar = new std::array<char, M * K>(); make_J(X_ar);
    cudaMemcpy(X_d, X_ar->data(), sizeof(char) * M * K, cudaMemcpyHostToDevice);

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(int) * M * N ); cudaMemset(c_d, 0, sizeof(int) * M * N);
    auto c_ar = new std::array<int, 1 * N>(); // store only first row

    std::cout << "Start: " << "M=" << M << " K=" << K << " N=" << N << " ITER=" << ITER_NUM << " W_MAP_LENGTH=" << W_MAP_LENGTH << " CALC_N_LENGTH=" << CALC_N_LENGTH << " NZ_RATIO=" << NZ_RATIO << " X:" << X_MAJOR << " W: " << W_MAJOR << " Y: " << C_MAJOR << std::endl;

    float ms = 0;

#ifdef RUN_TC
    // prepare W
    char *W = (char*) malloc(sizeof(char) * K * N);
    char *W_d;
    checkCudaErrors(cudaMalloc((void**) &W_d, sizeof(char) * K * N));
    prepareW_mat<<<N/16, 16>>>(W_d);
    cudaDeviceSynchronize(); // wait for prepareW
    checkCudaErrors(cudaMemcpy(W, W_d, sizeof(char) * K * N, cudaMemcpyDeviceToHost));

#ifndef TRANSFER
    checkCudaErrors(cudaMemcpy(W_d, W, sizeof(char) * K * N, cudaMemcpyHostToDevice)); // transfer W before hand
#endif

    ms = measureKernel([&](){
        for(size_t i = 0; i < ITER_NUM; i++){
#ifdef TRANSFER
            checkCudaErrors(cudaMemcpy(W_d, W, sizeof(char) * K * N, cudaMemcpyHostToDevice)); // transfer W every time
#endif
            checkKernelErrors((tcMatMul<<< dim3(N / 16, M / 16) , 32>>>((signed char *) X_d, (signed char *) W_d,c_d)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "TensorCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar->at(0) == -1 || c_ar->at(0) == 0 || c_ar->at(0) == 1);
#endif

#if defined(RUN_CUDA) || defined(RUN_NEW_2)
    // prepare W_map
    char *W_map = (char*) malloc(sizeof(char) * W_MAP_LENGTH * N);
    char *W_map_d;
    checkCudaErrors(cudaMalloc((void**) &W_map_d, sizeof(char) * W_MAP_LENGTH * N));
    prepareW_map<<<N/16, 16>>>(W_map_d);
    cudaDeviceSynchronize(); // wait for prepareW
    checkCudaErrors(cudaMemcpy(W_map, W_map_d, sizeof(char) * W_MAP_LENGTH * N, cudaMemcpyDeviceToHost));
#endif

#ifdef RUN_CUDA

#ifndef TRANSFER
    checkCudaErrors(cudaMemcpy(W_map_d, W_map, sizeof(char) * W_MAP_LENGTH * N, cudaMemcpyHostToDevice)); // before hand
#endif
    ms = measureKernel([&](){
        for(size_t i = 0; i < ITER_NUM; i++){
#ifdef TRANSFER
            checkCudaErrors(cudaMemcpy(W_map_d, W_map, sizeof(char) * W_MAP_LENGTH * N, cudaMemcpyHostToDevice)); // every time
#endif
            checkKernelErrors((cuMatMul<<< N * M / (CALC_N_LENGTH * 32), 32 >>>(X_d, W_map_d, c_d)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "CudaCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar->at(0) == -1 || c_ar->at(0) == 0 || c_ar->at(0) == 1);
#endif

#ifdef RUN_NEW_2

#ifndef TRANSFER
    checkCudaErrors(cudaMemcpy(W_map_d, W_map, sizeof(char) * W_MAP_LENGTH * N, cudaMemcpyHostToDevice)); // before hand
#endif
    ms = measureKernel([=](){
        for(size_t i = 0; i < ITER_NUM; i++){
#ifdef TRANSFER
            checkCudaErrors(cudaMemcpy(W_map_d, W_map, sizeof(char) * W_MAP_LENGTH * N, cudaMemcpyHostToDevice)); // every time
#endif
            checkKernelErrors((newMatMul2<<< dim3(N / 16, M / 16) , 32>>>((signed char *) X_d, (signed char *)  W_map_d, c_d)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "New Time 2: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar->at(0) == -1 || c_ar->at(0) == 0 || c_ar->at(0) == 1);
#endif

    return 0;
}