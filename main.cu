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
#include <nvml.h>

#include "submodule/wmma_extension/include/wmma_extension/wmma_extension.hpp"

DEFINE_bool(run_naive_tc, false, "Run naive TC method when true");
DEFINE_bool(run_naive_cu, false, "Run naive CU method when true");
DEFINE_bool(run_sparse_cu, false, "Run sparse CU method (CSR-format) when true");
DEFINE_bool(run_row, false, "Run Row-wise method when true");
DEFINE_bool(run_row_simd, false, "Run Row-wise SIMD method when true");
DEFINE_bool(run_tile, false, "Run Tile-wise method when true");
DEFINE_uint64(d_model, 12288L, "d_model");
DEFINE_uint64(batch_size, 32L, "batch size");
DEFINE_uint32(iter_num, 10, "Number of launching kernels");
DEFINE_uint32(sparse_ratio, 12, "(100 - 100/this)% of sparsity");
DEFINE_uint64(L, 16L, "Number of how each CUDA thread calculates in row-wise method");

// X: MxK  W: KxN  C: MxN
#define M FLAGS_batch_size
#define K (FLAGS_d_model * 4)
#define N (FLAGS_d_model)
#define W_MAP_LENGTH (K / (FLAGS_sparse_ratio * 2))
#define CALC_N_LENGTH (FLAGS_L)
#define MAJOR_ROW 0
#define MAJOR_COL 1
#define X_MAJOR MAJOR_COL
#define W_MAJOR MAJOR_COL
#define C_MAJOR MAJOR_COL
#define MAJOR_STR(m) (m == MAJOR_ROW ? "ROW" : "COL")
#define CAT(x, y) x ## y
#define BT_0(mat, row_dim, col_dim, row, col) mat[row * col_dim + col]
#define BT_1(mat, row_dim, col_dim, row, col) mat[col * row_dim + row]
#define BT(major) CAT(BT_, major)
//#define POWER

void nvmlAPIRun();
void nvmlAPIEnd();
void *powerPollingFunc(void *ptr);
int getNVMLError(nvmlReturn_t resultToCheck);

/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might prefer this approach.
*/
bool pollThreadStatus = false;
unsigned int deviceCount = 0;
char deviceNameStr[64];

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
nvmlPciInfo_t nvmPCIInfo;
nvmlEnableState_t pmmode;
nvmlComputeMode_t computeMode;

pthread_t powerPollThread;

/*
Poll the GPU using nvml APIs.
*/
void *powerPollingFunc(void *ptr)
{

    unsigned int powerLevel = 0;
    FILE *fp = fopen("Power_data.txt", "w+");

    while (pollThreadStatus)
    {
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

        // Get the power management mode of the GPU.
        nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);

        // The following function may be utilized to handle errors as needed.
        getNVMLError(nvmlResult);

        // Check if power management mode is enabled.
        if (pmmode == NVML_FEATURE_ENABLED)
        {
            // Get the power usage in milliWatts.
            nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
        }

        // The output file stores power in Watts.
        fprintf(fp, "%.3lf\n", (powerLevel)/1000.0);
        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    }

    fclose(fp);
    pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void nvmlAPIRun()
{
    int i;

    // Initialize nvml.
    nvmlResult = nvmlInit();
    if (NVML_SUCCESS != nvmlResult)
    {
        printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }

    // Count the number of GPUs available.
    nvmlResult = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != nvmlResult)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }

    for (i = 0; i < deviceCount; i++)
    {
        // Get the device ID.
        nvmlResult = nvmlDeviceGetHandleByIndex(i, &nvmlDeviceID);
        if (NVML_SUCCESS != nvmlResult)
        {
            printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }

        // Get the name of the device.
        nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr)/sizeof(deviceNameStr[0]));
        if (NVML_SUCCESS != nvmlResult)
        {
            printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }

        // Get PCI information of the device.
        nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmPCIInfo);
        if (NVML_SUCCESS != nvmlResult)
        {
            printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }

        // Get the compute mode of the device which indicates CUDA capabilities.
        nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID, &computeMode);
        if (NVML_ERROR_NOT_SUPPORTED == nvmlResult)
        {
            printf("This is not a CUDA-capable device.\n");
        }
        else if (NVML_SUCCESS != nvmlResult)
        {
            printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
            exit(0);
        }
    }

    // This statement assumes that the first indexed GPU will be used.
    // If there are multiple GPUs that can be used by the system, this needs to be done with care.
    // Test thoroughly and ensure the correct device ID is being used.
    nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);

    pollThreadStatus = true;

    const char *message = "Test";
    int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, (void*) message);
    if (iret)
    {
        fprintf(stderr,"Error - pthread_create() return code: %d\n",iret);
        exit(0);
    }
}

/*
End power measurement. This ends the polling thread.
*/
void nvmlAPIEnd()
{
    pollThreadStatus = false;
    pthread_join(powerPollThread, NULL);

    nvmlResult = nvmlShutdown();
    if (NVML_SUCCESS != nvmlResult)
    {
        printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
        exit(0);
    }
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int getNVMLError(nvmlReturn_t resultToCheck)
{
    if (resultToCheck == NVML_ERROR_UNINITIALIZED)
        return 1;
    if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
        return 2;
    if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
        return 3;
    if (resultToCheck == NVML_ERROR_NO_PERMISSION)
        return 4;
    if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
        return 5;
    if (resultToCheck == NVML_ERROR_NOT_FOUND)
        return 6;
    if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
        return 7;
    if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
        return 8;
    if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
        return 9;
    if (resultToCheck == NVML_ERROR_TIMEOUT)
        return 10;
    if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
        return 11;
    if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
        return 12;
    if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
        return 13;
    if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
        return 14;
    if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
        return 15;
    if (resultToCheck == NVML_ERROR_UNKNOWN)
        return 16;

    return 0;
}






struct ctx{
    uint64_t m;
    uint64_t k;
    uint64_t n;
    uint64_t sparse_ratio;
    uint64_t l;
    uint64_t w_map_length_pos; // S / 2
} ctx_v;

void init_ctx(){
    ctx_v = {
      .m = M,
      .k = K,
      .n = N,
      .sparse_ratio = FLAGS_sparse_ratio,
      .l = FLAGS_L,
      .w_map_length_pos = W_MAP_LENGTH
    };
}

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
__global__ void prepareW_map(unsigned short* const W_map, unsigned short* const W_map_negative, ctx ctx){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= ctx.n){
        // this thread won't work for init
        return;
    }

    int col = tid;

    // todo diff from prepareW_mat
    for(unsigned short row = 0; row < ctx.w_map_length_pos; row++){
        BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, row, col) = row;
    }
    for(unsigned short row = 0; row < ctx.w_map_length_pos; row++){
        BT(W_MAJOR) (W_map_negative ,ctx.w_map_length_pos , ctx.n, row, col) = row + ctx.w_map_length_pos;
    }
}

__global__ void prepareW_CSC(int8_t* values, int* row_indices, int* col_offsets, ctx ctx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= ctx.n) {
        return;
    }

    int col = tid;

    col_offsets[0] = 0;
    col_offsets[col+1] = ctx.w_map_length_pos * 2 * (col+1);
    for(int i = 0; i < ctx.w_map_length_pos * 2; i++){
        row_indices[ctx.w_map_length_pos * 2 * col + i] = i;
    }
    for(int i = 0; i < ctx.w_map_length_pos; i++){
        values[ctx.w_map_length_pos * 2 * col + i] = 1;
    }
    for(int i = ctx.w_map_length_pos; i < ctx.w_map_length_pos * 2; i++){
        values[ctx.w_map_length_pos * 2 * col + i] = -1;
    }
}

__global__ void naiveTC(
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

__global__ void naiveCU(
        const signed char* const X,
        const signed char* const W_mat,
        int* const c, ctx ctx){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_col = (tid / ctx.m) * ctx.l;
    int row = tid % ctx.m;

#pragma unroll
    for(int col = start_col; col < start_col + ctx.l; col++){
        int accum = 0;

#pragma unroll
        for(int k = 0; k < ctx.k; k++){
            accum += BT(X_MAJOR)(X, ctx.m, ctx.k, row, k) * BT(W_MAJOR)(W_mat, ctx.k, ctx.n, k, col);
        }
        BT(C_MAJOR)(c, ctx.m, ctx.n, row, col) = accum;
    }
}

__global__ void sparseCU( // with CSC format because CSC is suitable for X . W
        const signed char* const X,
        const int8_t* values,
        const int* row_indices,
        const int* col_offsets,
        int* const c, ctx ctx){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_col = (tid / ctx.m) * ctx.l;
    int row = tid % ctx.m;

#pragma unroll
    for(int col = start_col; col < start_col + ctx.l; col++){
        int accum = 0;

        for(int j = col_offsets[col]; j < col_offsets[col + 1]; ++j){
            int w_row = row_indices[j];
            accum += BT(X_MAJOR)(X, ctx.m, ctx.k, row, w_row) * values[j];
        }
        BT(C_MAJOR)(c, ctx.m, ctx.n, row, col) = accum;
    }
}

__global__ void rowWise(
        const char* const X,
        const unsigned short* const W_map,
        const unsigned short* const W_map_negative,
        int* const C,
        ctx ctx){
    // CUDA内では2配列として使うことはできない。
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_col = (tid / ctx.m) * ctx.l;
    int row = tid % ctx.m;

#pragma unroll
    for(int col = start_col; col < start_col + ctx.l; col++){
        int accum = 0;
#pragma unroll
        for(int i = 0; i < ctx.w_map_length_pos; i++){
            auto idx = BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, i, col);
            accum += BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx);
        }
        // indexを負の値にする方法では、なぜかパフォーマンスが劣化した
        // このため、別のmapとし作成することにより、パフォーマンスの劣化を抑える。
#pragma unroll
        for(int i = 0; i < ctx.w_map_length_pos; i++){
            auto idx = BT(W_MAJOR) (W_map_negative, ctx.w_map_length_pos, ctx.n, i, col);
            accum += -BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx);
        }
        BT(C_MAJOR) (C, ctx.m, ctx.n, row, col) = accum;
    }
}

__global__ void rowWiseSIMD(
        const char* const X,
        const unsigned short* const W_map,
        const unsigned short* const W_map_negative,
        int* const C,
        ctx ctx){
    assert(ctx.l % 2 == 0 && "L should be multiple of 2");

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_col = (tid / ctx.m) * ctx.l;
    int row = tid % ctx.m;

#pragma unroll
    for(int col = start_col; col < start_col + ctx.l; col+=2){
        unsigned int accum = 0;
#pragma unroll
        for(int i = 0; i < ctx.w_map_length_pos; i++){
            short idx_a = BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, i, col);
            short idx_b = BT(W_MAJOR) (W_map, ctx.w_map_length_pos, ctx.n, i, col+1);

            short v_a = (short) BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx_a);
            short v_b = (short) BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx_b);

            unsigned int tmp = v_a << 16 | v_b;
            accum = __vaddss2(tmp, accum);
        }
        // indexを負の値にする方法では、なぜかパフォーマンスが劣化した
        // このため、別のmapとし作成することにより、パフォーマンスの劣化を抑える。
#pragma unroll
        for(int i = 0; i < ctx.w_map_length_pos; i++){
            short idx_a = BT(W_MAJOR) (W_map_negative, ctx.w_map_length_pos, ctx.n, i, col);
            short idx_b = BT(W_MAJOR) (W_map_negative, ctx.w_map_length_pos, ctx.n, i, col+1);

            short v_a = (short) -BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx_a);
            short v_b = (short) -BT(X_MAJOR) (X, ctx.m, ctx.k, row, idx_b);

            unsigned int tmp = v_a << 16 | v_b;
            accum = __vaddss2(tmp, accum);
        }
        BT(C_MAJOR) (C, ctx.m, ctx.n, row, col) = accum >> 16;
        BT(C_MAJOR) (C, ctx.m, ctx.n, row, col+1) = ((accum << 16 ) >> 16);
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
__global__ void tileWise(
        const signed char* const X,
        const unsigned short* const W_map,
        const unsigned short* const W_map_negative,
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

#ifdef POWER
    nvmlAPIRun();
    fn();
    nvmlAPIEnd();
#else
    fn();
#endif

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
    init_ctx();

    assert(M % 16 == 0 && "mod 16 should be 0");
    assert(K % 16 == 0 && "mod 16 should be 0");
    assert(N % 16 == 0 && "mod 16 should be 0");
    assert(K < 65536 && "K should be fit in the maximum of short");

    char *X_d;
    cudaMalloc((void**) &X_d, sizeof(char) * M * K);
    char *X_ar = (char*) malloc(sizeof(char) * M * K); make_J(X_ar);
    cudaMemcpy(X_d, X_ar, sizeof(char) * M * K, cudaMemcpyHostToDevice);

    int *c_d; cudaMalloc((void**)  &c_d, sizeof(int) * M * N ); cudaMemset(c_d, 0, sizeof(int) * M * N);
    int *c_ar = (int*) malloc(sizeof(int) * N * 1); // store only first row

    std::cout
        << "Start: "
        << "M=" << M
        << " K=" << K
        << " N=" << N
        << " ITER=" << FLAGS_iter_num
        << " W_MAP_LENGTH=" << ctx_v.w_map_length_pos
        << " (" << (100.0 - 100.0 / (float) FLAGS_sparse_ratio) << "% Sparse)"
        << " CALC_N_LENGTH=" << CALC_N_LENGTH
        << " X_MAJOR=" << MAJOR_STR(X_MAJOR)
        << " W_MAJOR=" << MAJOR_STR(W_MAJOR)
        << " C_MAJOR=" << MAJOR_STR(C_MAJOR)
        << std::endl;

#ifdef POWER
    std::cout << "!!!!!!!!!! POWER MEASURE ON !!!!!!!!!!!!!" << std::endl;
#endif

    float ms = 0;

char *W_d;

if(FLAGS_run_naive_tc || FLAGS_run_naive_cu){
    checkCudaErrors(cudaMalloc((void **) &W_d, sizeof(char) * K * N));
    prepareW_mat<<<N / 16, 16>>>(W_d, ctx_v);
    cudaDeviceSynchronize();
}

if(FLAGS_run_naive_tc) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((naiveTC<<< dim3(N / 16, M / 16), 32>>>((signed char *) X_d, (signed char *) W_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "Naive TC: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar[0] == -1 || c_ar[0] == 0 || c_ar[0] == 1);
}

if(FLAGS_run_naive_cu) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((naiveCU<<< N * M / (CALC_N_LENGTH * 32), 32>>>((signed char *) X_d, (signed char *) W_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "Naive CU: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar[0] == -1 || c_ar[0] == 0 || c_ar[0] == 1);
}

if(FLAGS_run_sparse_cu){
    int8_t* values_d; // nnz
    int* row_indices_d; // nnz
    int* col_offsets_d; // column (of W) + 1

    checkCudaErrors(cudaMalloc((void **) &values_d, sizeof(char) * W_MAP_LENGTH * 2 * N));
    checkCudaErrors(cudaMalloc((void **) &row_indices_d, sizeof(int) * W_MAP_LENGTH * 2 * N));
    checkCudaErrors(cudaMalloc((void **) &col_offsets_d, sizeof(int) * (K+1)));
    prepareW_CSC<<<N / 16, 16>>>(values_d, row_indices_d, col_offsets_d, ctx_v);
    cudaDeviceSynchronize();

    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((sparseCU<<< N * M / (CALC_N_LENGTH * 32), 32>>>((signed char *) X_d, values_d, row_indices_d, col_offsets_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "Sparse CU: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    assert(c_ar[0] == -1 || c_ar[0] == 0 || c_ar[0] == 1);
}

unsigned short *W_map_d;
unsigned short *W_map_negative_d;

if(FLAGS_run_row || FLAGS_run_tile || FLAGS_run_row_simd){
    checkCudaErrors(cudaMalloc((void**) &W_map_d, sizeof(unsigned short) * W_MAP_LENGTH * N));
    checkCudaErrors(cudaMalloc((void**) &W_map_negative_d, sizeof(unsigned short) * W_MAP_LENGTH * N));
    prepareW_map<<<N/16, 16>>>(W_map_d, W_map_negative_d, ctx_v);
    cudaDeviceSynchronize();
}

if(FLAGS_run_row) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((rowWise<<< N * M / (CALC_N_LENGTH * 32), 32 >>>(X_d, W_map_d,W_map_negative_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "Row-wise: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    //assert(c_ar[0] == 0 && "what");
    //assert(c_ar[N / 2] == 0 && "what");
    //assert(c_ar[N - 1] == 0 && "what");
}

if(FLAGS_run_tile) {
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((tileWise<<< dim3(N / 16, M / 16), 32>>>((signed char *) X_d, (unsigned short *)  W_map_d, (unsigned short *)  W_map_negative_d,  c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "Tile-wise: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    //assert(c_ar[0] == 0 && "what");
    //assert(c_ar[N / 2] == 0 && "what");
    //assert(c_ar[N - 1] == 0 && "what");
}

if(FLAGS_run_row_simd){
    ms = measureKernel([&]() {
        for (size_t i = 0; i < FLAGS_iter_num; i++) {
            checkKernelErrors((rowWiseSIMD<<< N * M / (CALC_N_LENGTH * 32) / 2, 32 >>>(X_d, W_map_d,W_map_negative_d, c_d, ctx_v)));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    });
    std::cout << "Row-wise SIMD: " << ms / ((float) FLAGS_iter_num) << "ms" << std::endl;
    checkCudaErrors(cudaMemcpy(c_ar, c_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    //assert(c_ar[0] == 0 && "what");
    //assert(c_ar[N / 2] == 0 && "what");
    //assert(c_ar[N - 1] == 0 && "what");
}

    return 0;
}