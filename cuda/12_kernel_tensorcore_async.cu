#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

// Async/Multi-stage pipeline version of tensor core kernel
// This version uses multi-buffering (3 stages) to overlap memory loads with computation
// Similar to double buffering but with an extra stage for better overlap

template <typename InputType,
          const int BLOCK_ROW_WARPS = 4,
          const int BLOCK_COL_WARPS = 4,
          const int WARP_ROW_TILES = 4,
          const int WARP_COL_TILES = 2,
          const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int NUM_STAGES = 3>  // Multi-stage pipeline (3-stage buffering)
__global__ void
sgemm_tensorcore_async_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                               float alpha, const InputType *matrix_a,
                               const InputType *matrix_b, float beta,
                               float *matrix_c)
{
    // Thread and warp identification
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / BLOCK_COL_WARPS;
    const int warp_col = warp_id % BLOCK_COL_WARPS;

    // Compute block tile dimensions
    constexpr int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
    constexpr int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;
    constexpr int BM = BLOCK_ROW_TILES * WMMA_M; // 256
    constexpr int BN = BLOCK_COL_TILES * WMMA_N; // 128
    constexpr int BK = WMMA_K;                   // 16

    // Multi-stage shared memory
    __shared__ InputType tile_a[NUM_STAGES][BM * BK];
    __shared__ InputType tile_b[NUM_STAGES][BK * BN];

    const InputType *global_a = matrix_a;
    const InputType *global_b = matrix_b;
    float *global_c = matrix_c;

    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulators
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

    // Multi-stage buffering control
    int read_stage = 0;

    // ===== Prologue: Load first stage (stage 0) =====
    {
        // Load A tile (no bounds check - assumes aligned dimensions)
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS)
        {
            int row = idx / BK;
            int col = idx % BK;
            int global_row = blockIdx.y * BM + row;
            int global_col = col;

            tile_a[0][row * BK + col] = global_a[global_row * num_cols_a + global_col];
        }

        // Load B tile (column-major storage for WMMA, no bounds check)
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS)
        {
            int row = idx / BN;
            int col = idx % BN;
            int global_row = row;
            int global_col = blockIdx.x * BN + col;

            tile_b[0][col * BK + row] = global_b[global_row * num_cols_b + global_col];
        }
    }

    __syncthreads();

    // ===== Main K-loop with multi-stage buffering =====
    for (int block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        // Determine which stage to write next tile into (circular buffer)
        int write_stage = (read_stage + 1) % NUM_STAGES;

        // ===== Prefetch next tile (if not last iteration) =====
        if (block_k_idx + BK < num_cols_a)
        {
            // Load next A tile (no bounds check - assumes aligned dimensions)
            for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS)
            {
                int row = idx / BK;
                int col = idx % BK;
                int global_row = blockIdx.y * BM + row;
                int global_col = block_k_idx + BK + col;

                tile_a[write_stage][row * BK + col] = global_a[global_row * num_cols_a + global_col];
            }

            // Load next B tile (no bounds check - assumes aligned dimensions)
            for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS)
            {
                int row = idx / BN;
                int col = idx % BN;
                int global_row = block_k_idx + BK + row;
                int global_col = blockIdx.x * BN + col;

                tile_b[write_stage][col * BK + row] = global_b[global_row * num_cols_b + global_col];
            }
        }

        // ===== Compute using current read_stage =====
#pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i)
        {
#pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j)
            {
                int a_tile_row = warp_row * WARP_ROW_TILES + i;
                int b_tile_col = warp_col * WARP_COL_TILES + j;

                InputType const *a_tile_ptr = tile_a[read_stage] + (a_tile_row * WMMA_M) * BK;
                InputType const *b_tile_ptr = tile_b[read_stage] + (b_tile_col * WMMA_N) * BK;

                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);

                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        // Synchronize before switching buffers
        __syncthreads();

        // Switch to the next stage for next iteration
        read_stage = write_stage;
    }

    // ===== Write results to global memory =====
    // No bounds checking - assumes aligned dimensions
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

            // Load existing C and apply alpha/beta scaling
            nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

#pragma unroll
            for (int t = 0; t < c_frag.num_elements; ++t)
            {
                c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
            }

            // Write result back
            nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
        }
    }
}

// ============================================================================
// Launcher Functions: FP16 and BF16 variants
// ============================================================================

template <typename InputType, typename TorchType>
void sgemm_tensorcore_async_launcher(
    const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix, float alpha, float beta,
    torch::ScalarType expected_dtype, const char* dtype_name)
{
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == expected_dtype,
                std::string("Matrix A must be ") + dtype_name + " for Tensor Core async kernel");
    TORCH_CHECK(matrix_b.dtype() == expected_dtype,
                std::string("Matrix B must be ") + dtype_name + " for Tensor Core async kernel");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    const auto *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr<TorchType>());
    const auto *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr<TorchType>());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4;
    constexpr int WARP_COL_TILES = 2;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int NUM_STAGES = 3;

    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M; // 256
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N; // 128

    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);

    sgemm_tensorcore_async_kernel<InputType, BLOCK_ROW_WARPS, BLOCK_COL_WARPS,
                                   WARP_ROW_TILES, WARP_COL_TILES,
                                   WMMA_M, WMMA_N, WMMA_K, NUM_STAGES>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void sgemm_tensorcore_async_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                  torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_async_launcher<half, at::Half>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        torch::kFloat16, "float16");
}

void sgemm_tensorcore_async_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                  torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_async_launcher<nv_bfloat16, at::BFloat16>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        torch::kBFloat16, "bfloat16");
}
