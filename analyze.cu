// 大友ら、「Tensor コアの API の構造解析を用いた 拡張ライブラリの開発」より引用。
#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <string.h>
#include <utility>

#include "submodule/wmma_extension/include/wmma_extension/wmma_extension.hpp"

constexpr unsigned WARP_SIZE = 32u;

template <class T>
__device__ void init_matrix(T* mat, size_t len){
    for(size_t i = 0; i < len; i++){
        mat[i] = (T) i;
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

__global__ void analyze_fragment(){
    //using T = signed char;
    using T = uint8_t;
    //using T = half;

    __shared__ T matrix[16 * 16];

    unsigned i_map[8];
    unsigned j_map[8];

    if(0 == threadIdx.x){
        init_matrix(matrix, 16 * 16);

//        make_map_b(31, i_map, j_map);
//
//        for(size_t i = 0; i < 8; i++){
//            printf("(%u  %u),", i_map[i], j_map[i]);
//        }
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major> b_frag;

    nvcuda::wmma::load_matrix_sync(a_frag, matrix, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, matrix, 16);

    for(size_t tid = 0; tid < WARP_SIZE; tid++){ // こうしないとぐちゃぐちゃ
        if(tid == threadIdx.x){
            printf("tid:%d ", threadIdx.x);

            printf("matrix a %d elements ", a_frag.num_elements);
            for(size_t i = 0; i < a_frag.num_elements; i++){
                printf("%u,", static_cast<unsigned>(__half2float(a_frag.x[i])));
            }

            printf("\n");
        }
    }

    for(size_t tid = 0; tid < WARP_SIZE; tid++){
        if(tid == threadIdx.x){
            printf("tid:%d ", threadIdx.x);

            printf("matrix b %d elements ", b_frag.num_elements);
            for(size_t i = 0; i < b_frag.num_elements; i++){
                printf("%u,", static_cast<unsigned>(__half2float(b_frag.x[i])));
            }

            printf("\n");
        }
    }
}

int main(){
    analyze_fragment<<<1, WARP_SIZE>>>();
    cudaDeviceSynchronize();
}