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

using namespace nvcuda;

#ifndef CPU_DEBUG
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// A100などのデバイスではすでにこの制限はないが、簡単のため
#define SHARED_MEMORY_LIMIT_64K 1
#endif

#define WARP_SIZE 32

#define M 16
#define K 16
#define N 16

#define M_TILES 256
#define K_TILES 256
#define N_TILES 256

#define M_GLOBAL (M * M_TILES)
#define K_GLOBAL (K * K_TILES)
#define N_GLOBAL (N * N_TILES)

// とりあえず、256 thread per blocで比較する
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
#define CHUNK_K 8
#else
// ここはもっとでかくできる
#define CHUNK_K 16
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(uint8_t))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

#define SKEW_UINT8 32

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

__host__ void init_host_matrices(int8_t *X, int8_t *W) {
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            X[i * K_GLOBAL + j] = (int8_t)(rand() % 100);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            int percentage = 20;
            int r = rand() % percentage;
            if(r == 0) {
                W[i * K_GLOBAL + j] = 1; // 5% of 1
            }
            if(r == 1){
                W[i * K_GLOBAL + j] = -1;  // 5% of -1
            }
        }
    }
}


#define THREAD_BLOCK_SIZE 32

#define ITER_NUM 1000

__device__ int8_t W_mat[K_GLOBAL * N_GLOBAL]; // col major
// TODO X map should support dynamic length
// I just fill this matrix with index num
#define W_MAP_WIDTH (K_GLOBAL / 4)

__device__ unsigned short W_map[W_MAP_WIDTH * M]; // row major


/**
 * WとW_mapあらかじめ初期化しておく。
 *
 * <<<M / 32, 32 >>>
 */
__global__ void prepareW(){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= M){
        // this thread won't work for init
        return;
    }

    int row = tid;

    for(size_t col = 0; col < (K / 4); col++){
        W_map[row * (K / 4) + col] = col;
    }

    for(size_t col = 0; col < K; col++){
        if(col < (K / 4)){
            W_mat[row * K + col] = 1;
        }else{
            W_mat[row * K + col] = 0;
        }
    }
}


__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B,
                                  const int *C, int *D, int alpha, int beta) {
    extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i =
                ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from G to shared
        // memory.
        const size_t gmem_idx =
                (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const int *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            typedef int4 copy_t;

            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                    *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                      laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES]
        [WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const int *tile_ptr =
                        shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const uint8_t *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                                  M * K_GLOBAL * (warpId % 4) * 2)
                                               : (&B[block_tile_j * N * K_GLOBAL] +
                                                  N * K_GLOBAL * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx =
                    warpId < (WARPS_PER_BLOCK / 2)
                    ? (M * (warpId % 4) * 2)
                    : (N * (warpId % 4) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                      (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                             (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
                 i++) { // 8
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                        *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (int4 *)((uint8_t *)lane_ptr +
                                    K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major>
                        a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major>
                        b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off +
                                                 (WARP_ROW_TILES * N) * (warpId % 2) +
                                                 (j * N);
                            const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

                int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                    *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
}



__device__ char (* copyToShared(const char* const X_g))[THREAD_BLOCK_SIZE]
{
    /**
     * A100のshared memoryは164KB
     * H100のshared memoryは256KB
     * 0番目のスレッドがアクセスするのは、X行列の0列
     *
     */
     __shared__ char X_s[K][THREAD_BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(size_t row = 0; row < K; row++){
        for(size_t col = 0; col < THREAD_BLOCK_SIZE; col++){
            X_s[row][col] = X_g[row * N + tid];
        }
    }
    return X_s;
}

__global__ void cuMatMul(const char* const X, int* const c){


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    char (*X_s)[THREAD_BLOCK_SIZE] = copyToShared(X);
    __shared__ short W_map_s[M][W_MAP_WIDTH];

    for(size_t row = 0; row < M; row++){
        for(size_t col = 0; col < W_MAP_WIDTH; col++){
            // Just copy
            W_map_s[row][col] = W_map[row * W_MAP_WIDTH + col];
        }
    }

    __shared__ char c_s[M][THREAD_BLOCK_SIZE];

    static_assert(K * THREAD_BLOCK_SIZE + M * W_MAP_WIDTH + M * THREAD_BLOCK_SIZE <= SHARED_MEM_SIZE);

    BEGIN_ITER

    for(size_t row = 0; row < M; row++){
        int accum = 0;
        for(size_t i = 0; i < (K/4); i++){
            accum += X_s[W_map_s[row][i]][local_tid];
        }
        c_s[row][local_tid] = accum;
        //c[row * N + col] = accum;

        /**
         * col = 0, row = 1の時: c[1 * N + 0] => c[N] , c[2N], c[3N], c[4N] …と、飛び飛び？　
         * col = 1, row = 1の時: c[1 * N + 1] => c[N+1], c[2N+1], …と、飛び飛び？
         * だが、メモリの性質はうまく利用している？
         */
    }

    END_ITER

    for(size_t row = 0; row < M; row++){
        c[row * N + tid] = c_s[row][local_tid];
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


int main(int argc, char** argv){
    int dev = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.major < 7 || (deviceProp.major <= 7 && deviceProp.minor < 2)) {
        printf(
                "Requires SM 7.2 or higher to use Tensor Cores.  "
                "Exiting...\n");
        exit(EXIT_WAIVED);
    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    int8_t *X_h = nullptr, *W_h = nullptr;
    int *C_h = nullptr;

#if CPU_DEBUG
    int *result_h = nullptr;
#endif

    X_h = (int8_t *)calloc(M_GLOBAL * K_GLOBAL, sizeof(uint8_t));
    W_h = (int8_t *)calloc(K_GLOBAL * N_GLOBAL, sizeof(uint8_t));
    C_h = (int *)calloc(M_GLOBAL * N_GLOBAL, sizeof(int));

#if CPU_DEBUG
    result_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#endif

    int8_t *X = nullptr, *W = nullptr;
    int *C = nullptr;

    checkCudaErrors(
            cudaMalloc(reinterpret_cast<void **>(&X), sizeof(int8_t) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(
            cudaMalloc(reinterpret_cast<void **>(&W), sizeof(int8_t) * K_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), sizeof(int) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)X) % 128 == 0);
    assert(((unsigned long long)W) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);

    init_host_matrices(X_h, W_h);

    checkCudaErrors(cudaMemcpy(X, X_h, sizeof(int8_t) * M_GLOBAL * K_GLOBAL,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(W, W_h, sizeof(int8_t) * N_GLOBAL * K_GLOBAL,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(C, 0, sizeof(int) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)X) % 128 == 0);
    assert(((unsigned long long)W) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);

    // WとW_mapをGPU上にコピーしておく

    prepareW<<< M / 16, 16>>>();
    cudaDeviceSynchronize();

    std::cout << "Start: " << "M=" << M << " K=" << K << " N=" << N << " ITER=" << ITER_NUM << std::endl;

    float ms = measureKernel([X_d, c_d](){
        tcMatMul<<< dim3(N / 16, M / 16) , 32>>>(( signed char * )  X_d, c_d);
    });
    std::cout << "TensorCore Time: " << ms << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == K / 4 && "what");

    ms = measureKernel([X_d, c_d](){
        cuMatMul<<<(N / 32) , 32>>>(X_d, c_d);
    });
    std::cout << "CudaCore Time: " << ms << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == K / 4 && "what");

    ms = measureKernel([X_d, c_with_bank_d](){
        cuMatMulCol<<<(N / 32) , 32>>>(X_d, c_with_bank_d);
    });
    std::cout << "CU Column Time: " << ms << "ms" << std::endl;
    cudaMemcpy(c_ar->data(), c_with_bank_d, N * sizeof(int), cudaMemcpyDeviceToHost);
    assert(c_ar->at(0) == K / 4 && "what");

    return 0;
}