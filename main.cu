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
#include <gflags/gflags.h>

#include "submodule/wmma_extension/include/wmma_extension/wmma_extension.hpp"

//#define RUN_TC

DEFINE_bool(run_naive_tc, false, "Run naive TC method when true");
DEFINE_bool(run_row, false, "Run Row-wise method when true");
DEFINE_bool(run_tile, false, "Run Tile-wise method when true");

// X: MxK  W: KxN  C: MxN
DEFINE_uint64(d_model, 12288L, "d_model");
DEFINE_uint64(batch_size, 32L, "batch size");

#define M FLAGS_batch_size
#define K (FLAGS_d_model * 4)
#define N (FLAGS_d_model)

DEFINE_uint32(iter_num, 10, "Number of launching kernels");

DEFINE_uint32(sparse_ratio, 12, "(100 - 100/this)% of sparsity");

#define W_MAP_LENGTH (K / (FLAGS_sparse_ratio * 2))

DEFINE_uint64(L, 16L, "Number of how each CUDA thread calculates in row-wise method");

#define CALC_N_LENGTH (FLAGS_L)

#define MAJOR_ROW 0
#define MAJOR_COL 1
#define X_MAJOR MAJOR_COL
#define W_MAJOR MAJOR_COL
#define C_MAJOR MAJOR_COL

#define CAT(x, y) x ## y

#define BT_0(mat, row_dim, col_dim, row, col) mat[row * col_dim + col]
#define BT_1(mat, row_dim, col_dim, row, col) mat[col * row_dim + row]
#define BT(major) CAT(BT_, major)

struct ctx{
    uint64_t m = M;
    uint64_t k = K;
    uint64_t n = N;
    uint32_t sparse_ratio = FLAGS_sparse_ratio;
    uint64_t l = FLAGS_L;
    uint32_t w_map_length_pos = W_MAP_LENGTH;
} ctx_v;

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


__global__ void prepareW_mat(char* const W_mat, ctx ctx){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= ctx.n){
        // this thread won't work for init
        return;
    }

    int col = tid;

    for(size_t row = 0; row < ctx.k; row++){
        if(row % ctx.sparse_ratio == 0){
            int sign = (row / ctx.sparse_ratio) % 2 == 0 ? 1 : -1;
                    BT(W_MAJOR) (W_mat, ctx.k, ctx.n, row, col) = sign;
        }else{
                    BT(W_MAJOR) (W_mat, ctx.k, ctx.n, row, col) = 0;
        }
    }
}

/**
 * Prepare both W_mat and W_map before the measurement.
 */
__global__ void prepareW_map(char* const W_map, char* const W_map_negative, ctx ctx){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= ctx.n){
        // this thread won't work for init
        return;
    }

    int col = tid;

    // todo diff from prepareW_mat
    for(size_t row = 0; row < ctx.w_map_length_pos; row++){
        BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, row, col) = row;
    }
    for(size_t row = 0; row < ctx.w_map_length_pos; row++){
        BT(W_MAJOR) (W_map_negative ,ctx.w_map_length_pos , ctx.n, row, col) = row + ctx.w_map_length_pos;
    }
}

/**
 * ここはroとcol orderで固定にする良さそう
 */
__global__ void tcMatMul(
        const signed char* const X,
        const signed char* const W_mat,
        int* const c, ctx ctx){
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, std::conditional_t<X_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> X_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, std::conditional_t<W_MAJOR == MAJOR_ROW, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> W_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0);

#pragma unroll
    for(size_t k = 0; k < ctx.k; k += 16){
        if constexpr(X_MAJOR == MAJOR_ROW){
            nvcuda::wmma::load_matrix_sync(X_frag, X + (blockIdx.y * ctx.k * 16 + k), ctx.k);
        }else{
            nvcuda::wmma::load_matrix_sync(X_frag, X + (k * ctx.m + blockIdx.y * 16), ctx.m);
        }

        if constexpr(W_MAJOR == MAJOR_ROW){
            nvcuda::wmma::load_matrix_sync(W_frag, W_mat + ( k * ctx.n + blockIdx.x * 16), ctx.n);
        }else{
            nvcuda::wmma::load_matrix_sync(W_frag, W_mat + ( k + blockIdx.x * 16 * ctx.k), ctx.k);
        }
        nvcuda::wmma::mma_sync(c_frag, X_frag, W_frag, c_frag);
    }

    if constexpr(C_MAJOR == MAJOR_ROW){
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.y * 16 * ctx.n + blockIdx.x * 16), c_frag, ctx.n, nvcuda::wmma::mem_row_major);
    }else{
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.x * 16 * ctx.m + blockIdx.y * 16), c_frag, ctx.m, nvcuda::wmma::mem_col_major);
    }
}

__global__ void cuMatMul(
        const char* const X,
        const char* const W_map,
        const char* const W_map_negative,
        int* const C,
        ctx ctx){
    // CUDA内では2配列として使うことはできない。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_col = (tid / ctx.m) * ctx.l;
    int row = tid % ctx.m;

#pragma unroll
    for(size_t col = start_col; col < start_col + ctx.l; col++){
        int accum = 0;
#pragma unroll
        for(size_t i = 0; i < ctx.w_map_length_pos; i++){
            auto idx = BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, i, col);
            accum += BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx);
        }
        // indexを負の値にする方法では、なぜかパフォーマンスが劣化した
        // このため、別のmapとし作成することにより、パフォーマンスの劣化を抑える。
#pragma unroll
        for(size_t i = 0; i < ctx.w_map_length_pos; i++){
            auto idx = BT(W_MAJOR) (W_map_negative, ctx.w_map_length_pos, ctx.n, i, col);
            accum += -BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx);
        }
        BT(C_MAJOR) (C, ctx.m, ctx.n, row, col) = accum;
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
        const signed char* const W_map_negative,
        int* const c,
        ctx ctx){
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

#pragma unroll
    for(unsigned k = 0; k < ctx.w_map_length_pos; k++){

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
            auto col_idx = BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, k, blockIdx.x * 16 + j);
            M_frag.x[f] = BT(X_MAJOR) (X, ctx.m, ctx.k, blockIdx.y * 16 + i, col_idx);
        }
        __syncwarp();

        nvcuda::wmma::mma_sync(c_frag, M_frag, I_frag, c_frag);

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
            auto col_idx = BT(W_MAJOR) (W_map_negative, ctx.w_map_length_pos, ctx.n, k, blockIdx.x * 16 + j);
            M_frag.x[f] = -BT(X_MAJOR) (X, ctx.m, ctx.k, blockIdx.y * 16 + i, col_idx);
        }
        __syncwarp();

        nvcuda::wmma::mma_sync(c_frag, M_frag, I_frag, c_frag);
    }

    if constexpr(C_MAJOR == MAJOR_ROW){
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.y * 16 * ctx.n + blockIdx.x * 16), c_frag, ctx.n, nvcuda::wmma::mem_row_major);
    }else{
        nvcuda::wmma::store_matrix_sync(c + (blockIdx.x * 16 * ctx.m + blockIdx.y * 16), c_frag, ctx.m, nvcuda::wmma::mem_col_major);
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

void make_J(char *X){
    for(unsigned i = 0; i < M * K; i++) {
        X[i] = 1;
    }
}

int main(int argc, char** argv){
    gflags::SetUsageMessage("matrix multiply speed check");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    assert(M % 16 == 0 && "mod 16 should be 0");
    assert(K % 16 == 0 && "mod 16 should be 0");
    assert(N % 16 == 0 && "mod 16 should be 0");
    assert(K < 65536 && "K should be fit in the maximum of short");

    char *X_d;
    cudaMalloc((void**) &X_d, sizeof(char) * M * K);
    char *X_ar = (char*) malloc(sizeof(char) * K * N); make_J(X_ar);
    cudaMemcpy(X_d, X_ar, sizeof(char) * M * K, cudaMemcpyHostToDevice);

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(int) * M * N ); cudaMemset(c_d, 0, sizeof(int) * M * N);
    int *c_ar = (int*) malloc(sizeof(int) * N * 1); // store only first row

    std::cout << "Start: " << "M=" << M << " K=" << K << " N=" << N << " ITER=" << FLAGS_iter_num << " W_MAP_LENGTH=" << W_MAP_LENGTH << " CALC_N_LENGTH=" << CALC_N_LENGTH << std::endl;

    float ms = 0;

if(FLAGS_run_naive_tc) {
    char *W_d;
    checkCudaErrors(cudaMalloc((void **) &W_d, sizeof(char) * K * N));
    prepareW_mat<<<N / 16, 16>>>(W_d, ctx_v);
    cudaDeviceSynchronize();

    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((tcMatMul<<< dim3(N / 16, M / 16), 32>>>((signed char *) X_d, (signed char *) W_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "TensorCore Time: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar[0] == -1 || c_ar[0] == 0 || c_ar[0] == 1);
}

char *W_map_d;
char *W_map_negative_d;

if(FLAGS_run_row || FLAGS_run_tile){
    checkCudaErrors(cudaMalloc((void**) &W_map_d, sizeof(char) * W_MAP_LENGTH * N));
    checkCudaErrors(cudaMalloc((void**) &W_map_negative_d, sizeof(char) * W_MAP_LENGTH * N));
    prepareW_map<<<N/16, 16>>>(W_map_d, W_map_negative_d, ctx_v);
    cudaDeviceSynchronize();
}

if(FLAGS_run_row) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((cuMatMul<<< N * M / (CALC_N_LENGTH * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "CudaCore Time: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar[0] == 0 && "what");
    assert(c_ar[N / 2] == 0 && "what");
    assert(c_ar[N - 1] == 0 && "what");
}

if(FLAGS_run_tile) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((newMatMul2<<< dim3(N / 16, M / 16), 32>>>((signed char *) X_d, (signed char *)  W_map_d, (signed char *)  W_map_negative_d,  c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "New Time 2: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar[0] == 0 && "what");
    assert(c_ar[N / 2] == 0 && "what");
    assert(c_ar[N - 1] == 0 && "what");
}

    return 0;
}