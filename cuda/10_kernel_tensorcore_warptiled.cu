// Tensor Core GEMM: C = alpha * A * B + beta * C
// A, B are FP16/BF16; C is FP32. Uses WMMA API with block and warp tiling.

#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

// Block tiling: BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M (default 256)
//               BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N (default 128)
// Warp tiling: Each warp computes WARP_ROW_TILES x WARP_COL_TILES WMMA tiles (default 4x2)
template <typename InputType,
          const int BLOCK_ROW_WARPS = 4,
          const int BLOCK_COL_WARPS = 4,
          const int WARP_ROW_TILES = 4,
          const int WARP_COL_TILES = 2,
          const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16>
__global__ void
sgemm_tensorcore_warptiled_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                  float alpha, const InputType *matrix_a,
                                  const InputType *matrix_b, float beta,
                                  float *matrix_c)
{
    const uint warp_id = threadIdx.x / 32;
    const uint warp_row = warp_id / BLOCK_COL_WARPS;
    const uint warp_col = warp_id % BLOCK_COL_WARPS;

    constexpr int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
    constexpr int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;

    constexpr int BM = BLOCK_ROW_TILES * WMMA_M;
    constexpr int BN = BLOCK_COL_TILES * WMMA_N;
    constexpr int BK = WMMA_K;

    // Shared memory: tile_a (BM x BK, row-major), tile_b (BK x BN, column-major)
    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

    const InputType *global_a = matrix_a;
    const InputType *global_b = matrix_b;
    float *global_c = matrix_c;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::col_major> b_frag;

    // Accumulator fragments (FP32): each warp maintains WARP_ROW_TILES x WARP_COL_TILES tiles
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j)
        {
            nvcuda::wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    constexpr int NUM_THREADS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32;

    // K-loop: iterate by BK, load A and B tiles, compute WMMA operations
    for (int block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        // Load A tile (BM x BK, row-major)
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS)
        {
            int row = idx / BK;
            int col = idx % BK;
            int global_row = blockIdx.y * BM + row;
            int global_col = block_k_idx + col;
            tile_a[row * BK + col] = global_a[global_row * num_cols_a + global_col];
        }

        // Load B tile (BK x BN, column-major for WMMA)
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS)
        {
            int row = idx / BN;
            int col = idx % BN;
            int global_row = block_k_idx + row;
            int global_col = blockIdx.x * BN + col;
            tile_b[col * BK + row] = global_b[global_row * num_cols_b + global_col];
        }

        __syncthreads();

        // Warp-level tiling: each warp computes WARP_ROW_TILES x WARP_COL_TILES WMMA tiles
#pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i)
        {
#pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j)
            {
                int a_tile_row = warp_row * WARP_ROW_TILES + i;
                int b_tile_col = warp_col * WARP_COL_TILES + j;

                InputType const *a_tile_ptr = tile_a + (a_tile_row * WMMA_M) * BK;
                InputType const *b_tile_ptr = tile_b + (b_tile_col * WMMA_N) * BK;

                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);
                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // Store results: C = alpha * (A * B) + beta * C
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j)
        {
            int c_tile_row = warp_row * WARP_ROW_TILES + i;
            int c_tile_col = warp_col * WARP_COL_TILES + j;

            int global_row = blockIdx.y * BM + c_tile_row * WMMA_M;
            int global_col = blockIdx.x * BN + c_tile_col * WMMA_N;

            float *c_ptr = global_c + global_row * num_cols_b + global_col;

            nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

#pragma unroll
            for (int t = 0; t < c_frag.num_elements; ++t)
            {
                c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
            }

            nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
        }
    }
}

template <typename InputType>
void sgemm_tensorcore_warptiled_impl(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                     torch::Tensor &output_matrix, float alpha, float beta,
                                     torch::ScalarType expected_dtype)
{
    TORCH_CHECK(matrix_a.device().is_cuda() && matrix_b.device().is_cuda(), "Matrices must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == expected_dtype && matrix_b.dtype() == expected_dtype, "Input dtype mismatch");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "Matrices must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a && output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix dimensions must match");

    const InputType *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr());
    const InputType *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Block tiling: 256x128 (4 warps x 4 tiles per warp)
    constexpr int BLOCK_ROW_WARPS = 4, BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4, WARP_COL_TILES = 2;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M;
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N;

    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);

    sgemm_tensorcore_warptiled_kernel<InputType, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, WMMA_M, WMMA_N, WMMA_K>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

void sgemm_tensorcore_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_warptiled_impl<half>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kFloat16);
}

void sgemm_tensorcore_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_warptiled_impl<nv_bfloat16>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kBFloat16);
}
