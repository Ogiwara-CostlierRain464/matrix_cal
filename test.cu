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

// X: MxK  W: KxN  C: MxN
#define D_MODEL 2048L
#define M D_MODEL
#define K D_MODEL
#define N (D_MODEL * 4)
#define ITER_NUM 1000

#define W_MAP_LENGTH (K / 10)

#define CALC_N_LENGTH (8L)

#define MAJOR_ROW 0
#define MAJOR_COL 1
#define X_MAJOR MAJOR_ROW
#define W_MAJOR MAJOR_COL
#define C_MAJOR MAJOR_COL

// when x major is row major
#if !(X_MAJOR)
#define X_TYPE char[M][K]
#define _X_DIM(M, K) [M][K]
#define X_DIM _X_DIM(M, K)
#else // col major
#define X_TYPE char[K][M]
#define _X_DIM(K, M) [K][M]
#define X_DIM _X_DIM(K, M)
#endif

#define MAKE_GPU_MATRIX_ROW_MAJOR(name, type, row_size, col_size) __device__ type name[row_size][col_size];
#define MAKE_GPU_MATRIX_COL_MAJOR(name, type, row_size, col_size) __device__ type name[col_size][row_size];

// when c major is row major
#if !(C_MAJOR)
#define C_TYPE int[M][N]
#define _C_DIM(M, N) [M][N]
#define C_DIM _C_DIM(M, N)
#else // col major
#define C_TYPE int[N][M]
#define _C_DIM(N, M) [N][M]
#define C_DIM _C_DIM(N, M)
#endif


#define CAT(x, y) x ## y

// row major
#define AT_0(mat, row, col) mat[row][col]
// col major
#define AT_1(mat, row, col) mat[col][row]
#define AT(major) CAT(AT_, major)

#define BT_0(mat, row_dim, col_dim, row, col) mat[row * col_dim + col]
#define BT_1(mat, row_dim, col_dim, row, col) mat[col * row_dim + row]
#define BT(major) CAT(BT_, major)

__device__ signed char W_mat[K * N];
//MAKE_GPU_MATRIX_COL_MAJOR(W_mat, signed char, K, N)
// W map should support dynamic length
// I just fill this matrix with index num
__device__ unsigned short W_map[W_MAP_LENGTH * N];
//MAKE_GPU_MATRIX_COL_MAJOR(W_map, unsigned short, W_MAP_LENGTH, N)
__device__ unsigned short W_map_negative[W_MAP_LENGTH * N];
//MAKE_GPU_MATRIX_COL_MAJOR(W_map_negative, unsigned short, W_MAP_LENGTH, N)

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


__global__ void tcMatMul(const signed char* const X,
                       int* const c){

#if X_MAJOR == MAJOR_ROW
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> X_frag;
#else
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::col_major> X_frag;
#endif
#if W_MAJOR == MAJOR_ROW
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> W_frag;
#else
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::col_major> W_frag;
#endif
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0);

#pragma unroll
    for(size_t k = 0; k < K; k += 16){
        nvcuda::wmma::load_matrix_sync(X_frag, X + (blockIdx.y * K * 16 + k), X_MAJOR == MAJOR_ROW ? K : M);
        nvcuda::wmma::load_matrix_sync(W_frag, X + ( k * N + blockIdx.x * 16), W_MAJOR == MAJOR_ROW ? N : K);
        nvcuda::wmma::mma_sync(c_frag, X_frag, W_frag, c_frag);
    }

    // ここ、変わる？
    nvcuda::wmma::store_matrix_sync(c + (blockIdx.y * N * 16 + blockIdx.x * 16), c_frag, N, C_MAJOR == MAJOR_ROW ? nvcuda::wmma::mem_row_major : nvcuda::wmma::mem_col_major );
}

__global__ void cuMatMul(char *X , int *C){
    // CUDA内は2配列として使うことはできない。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_col = (tid / M) * CALC_N_LENGTH;
    int row = tid % M;

#pragma unroll
    for(size_t col = start_col; col < start_col + CALC_N_LENGTH; col++){
        int accum = 0;
#pragma unroll
        for(size_t i = 0; i < W_MAP_LENGTH; i++){
            unsigned short idx = BT(W_MAJOR) (W_map, W_MAP_LENGTH, N, i, col);
            accum += BT(X_MAJOR) (X, M, K, row, idx);
        }
        // indexを負の値にする方法では、なぜかパフォーマンスが劣化した
        // このため、別のmapとし作成することにより、パフォーマンスの劣化を抑える。
#pragma unroll
        for(size_t i = 0; i < W_MAP_LENGTH; i++){
            unsigned short idx = BT(W_MAJOR) (W_map, W_MAP_LENGTH, N, i, col);
            accum += BT(X_MAJOR) (X, M, K, row, idx);
        }
        BT(C_MAJOR) (C, M, N, row, col) = accum;
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

/**
 * Calc matmul of W(MxK) and X(KxN), where W is ternary matrix and X is 8-bit matrix.
 * Since X is 8-bit, we need to implement W as a 8-bit matrix due to restriction of wmma.
 * W is prepared before the performance measure.
 */
int main(int argc, char** argv){

    static_assert(M % 16 == 0 && "mod 16 should be 0");
    static_assert(K % 16 == 0 && "mod 16 should be 0");
    static_assert(N % 16 == 0 && "mod 16 should be 0");
    static_assert(K < 65536 && "K should be fit in the maximum of short");

    char *X_d;
    cudaMalloc((void**) &X_d, sizeof(X_TYPE));
    auto *X_ar = new std::array<char, M * K>(); make_J(X_ar);
    cudaMemcpy(X_d, X_ar->data(), sizeof(X_TYPE), cudaMemcpyHostToDevice);

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(C_TYPE) ); cudaMemset(c_d, 0, sizeof(C_TYPE));
    auto c_ar = new std::array<int, N * 1>(); // store only first row

    prepareW<<< N / 16, 16>>>();
    cudaDeviceSynchronize();

    std::cout << "Start: " << "M=" << M << " K=" << K << " N=" << N << " ITER=" << ITER_NUM << " W_MAP_LENGTH=" << W_MAP_LENGTH << " CALC_N_LENGTH=" << CALC_N_LENGTH << std::endl;

    float ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((tcMatMul<<< dim3(N / 16, M / 16) , 32>>>((signed char *) X_d, c_d)));
        }
    });
    std::cout << "TensorCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");

    ms = measureKernel([X_d, c_d](){
        for(size_t i = 0; i < ITER_NUM; i++){
            checkKernelErrors((cuMatMul<<< N * M / (CALC_N_LENGTH * 32), 32 >>>(X_d, c_d)));
        }
    });
    std::cout << "CudaCore Time: " << ms / ((float) ITER_NUM) << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == 0 && "what");


    return 0;
}