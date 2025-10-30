#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

template <typename InputType,
          const int BLOCK_ROW_WARPS = 4,
          const int BLOCK_COL_WARPS = 4,
          const int WARP_ROW_TILES = 4,
          const int WARP_COL_TILES = 2,
          const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16>
__global__ void
sgemm_tensorcore_double_buffered_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                        float alpha, const InputType *matrix_a,
                                        const InputType *matrix_b, float beta,
                                        float *matrix_c)
{
    // Thread and warp identification
    const int warp_id = threadIdx.x / 32; // Warp ID within block (0 to BLOCK_ROW_WARPS*BLOCK_COL_WARPS-1)

    // Warp position in 2D block layout (row-major ordering)
    // With 4x4 warp layout: warp_id 0-3 are row 0, warp_id 4-7 are row 1, etc.
    const int warp_row = warp_id / BLOCK_COL_WARPS; // Which warp row (0 to BLOCK_ROW_WARPS-1)
    const int warp_col = warp_id % BLOCK_COL_WARPS; // Which warp column (0 to BLOCK_COL_WARPS-1)

    // Compute block tile dimensions in WMMA tiles
    constexpr int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS; // Total 16x16 tiles along M
    constexpr int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS; // Total 16x16 tiles along N

    // Compute block tile dimensions in elements
    constexpr int BM = BLOCK_ROW_TILES * WMMA_M; // 256: rows of A/C per block
    constexpr int BN = BLOCK_COL_TILES * WMMA_N; // 128: cols of B/C per block
    constexpr int BK = WMMA_K;                   // 16: inner dimension per iteration

    // Double-buffered shared memory layout:
    // - tile_a[2]: two BM x BK buffers (2 * 256x16), stored row-major for coalesced A loads
    // - tile_b[2]: two BK x BN buffers (2 * 16x128), stored COLUMN-major to match WMMA fragment expectation
    __shared__ InputType tile_a[2][BM * BK];
    __shared__ InputType tile_b[2][BK * BN];

    // Base pointers to global memory (block-level, not offset yet)
    const InputType *global_a = matrix_a;
    const InputType *global_b = matrix_b;
    float *global_c = matrix_c;

    // WMMA fragments (register-level storage for matrix tiles)
    // Fragment for A tiles (16x16 input matrix, row-major layout)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;

    // Fragment for B tiles (16x16 input matrix, column-major layout)
    // Column-major is critical: matches our shared memory layout for efficient loads
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::col_major> b_frag;

    // Accumulator fragments for output tiles (FP32 for numerical stability)
    // Each warp maintains WARP_ROW_TILES x WARP_COL_TILES = 4x2 = 8 accumulators
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];

    // Temporary fragment for loading existing C values (when beta != 0)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize all accumulator fragments to zero
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

    // Double buffering control: which buffer is currently being computed on
    int read_buffer = 0;

    // ===== Prologue: Load the first tile into buffer 0 =====
    {
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS)
        {
            int row = idx / BK;
            int col = idx % BK;
            int global_row = blockIdx.y * BM + row;
            int global_col = col;

            tile_a[0][row * BK + col] = global_a[global_row * num_cols_a + global_col];
        }

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

    // Main K-loop: iterate over K dimension in chunks of size BK (16)
    // Each iteration: load next tile into write_buffer while computing current read_buffer
    for (int block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        // Determine which buffer to write next tile into
        int write_buffer = read_buffer ^ 1; // Toggle between 0 and 1

        // ===== Prefetch next tile into write_buffer (if not last iteration) =====
        if (block_k_idx + BK < num_cols_a)
        {
            // Load next A tile - no bounds check (assumes aligned dimensions)
            for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS)
            {
                int row = idx / BK;
                int col = idx % BK;
                int global_row = blockIdx.y * BM + row;
                int global_col = block_k_idx + BK + col;

                // Direct load - no bounds check
                tile_a[write_buffer][row * BK + col] = global_a[global_row * num_cols_a + global_col];
            }

            // Load next B tile - no bounds check (assumes aligned dimensions)
            for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS)
            {
                int row = idx / BN;
                int col = idx % BN;
                int global_row = block_k_idx + BK + row;
                int global_col = blockIdx.x * BN + col;

                // Direct load - no bounds check
                tile_b[write_buffer][col * BK + row] = global_b[global_row * num_cols_b + global_col];
            }
        }

        // ===== Compute using current read_buffer =====
        // Each warp independently computes WARP_ROW_TILES x WARP_COL_TILES output tiles
        // using WMMA operations on tensor cores
#pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i) // Iterate over warp's row tiles
        {
#pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j) // Iterate over warp's col tiles
            {
                // Compute which 16x16 tile this warp is processing within the block
                int a_tile_row = warp_row * WARP_ROW_TILES + i; // Tile index in A (0 to BLOCK_ROW_TILES-1)
                int b_tile_col = warp_col * WARP_COL_TILES + j; // Tile index in B (0 to BLOCK_COL_TILES-1)

                InputType const *a_tile_ptr = tile_a[read_buffer] + (a_tile_row * WMMA_M) * BK;
                InputType const *b_tile_ptr = tile_b[read_buffer] + (b_tile_col * WMMA_N) * BK;

                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);

                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        // Synchronize before switching buffers (ensures loads complete and computation reads correct data)
        __syncthreads();

        // Switch to the newly loaded buffer for next iteration
        read_buffer = write_buffer;

    } // End of K-loop: accumulation complete in acc_frag

    // ===== Phase 4: Write results to global memory =====
    // Store accumulated results from fragments to output matrix C
    // Apply alpha/beta scaling: C = alpha * (A * B) + beta * C
    // NOTE: Assumes M is multiple of BM and N is multiple of BN (no bounds checking)
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

            // Always load C and compute: C = alpha * AB + beta * C
            nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

#pragma unroll
            for (int t = 0; t < c_frag.num_elements; ++t)
            {
                c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
            }

            // Write result back to global memory
            nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
        }
    }
}

// ============================================================================
// Launcher Functions: FP16 and BF16 variants
// ============================================================================

// Template launcher for Tensor Core GEMM with double buffering
// Handles both FP16 and BF16 input types
// Inputs: matrix_a (M x K, InputType), matrix_b (K x N, InputType)
// Output: output_matrix (M x N, FP32)
// Computes: output_matrix = alpha * (matrix_a @ matrix_b) + beta * output_matrix
template <typename InputType, typename TorchType>
void sgemm_tensorcore_double_buffered_launcher(
    const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix, float alpha, float beta,
    torch::ScalarType expected_dtype, const char* dtype_name)
{
    // Input validation
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == expected_dtype,
                std::string("Matrix A must be ") + dtype_name + " for Tensor Core kernel");
    TORCH_CHECK(matrix_b.dtype() == expected_dtype,
                std::string("Matrix B must be ") + dtype_name + " for Tensor Core kernel");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    // Extract matrix dimensions
    const int num_rows_a = static_cast<int>(matrix_a.size(0)); // M
    const int num_cols_a = static_cast<int>(matrix_a.size(1)); // K
    const int num_cols_b = static_cast<int>(matrix_b.size(1)); // N

    // Dimension consistency checks
    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    // Get device pointers
    const auto *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr<TorchType>());
    const auto *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr<TorchType>());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Kernel configuration (matching template defaults)
    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4;
    constexpr int WARP_COL_TILES = 2;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Block tile dimensions in elements
    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M; // 4*4*16 = 256
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N; // 2*4*16 = 128

    // Grid and block dimensions
    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32); // 16 warps * 32 = 512 threads

    // Launch kernel
    sgemm_tensorcore_double_buffered_kernel<InputType, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, WMMA_M, WMMA_N, WMMA_K>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// FP16 Tensor Core GEMM launcher with double buffering
// Inputs: matrix_a (M x K, FP16), matrix_b (K x N, FP16)
// Output: output_matrix (M x N, FP32)
// Computes: output_matrix = alpha * (matrix_a @ matrix_b) + beta * output_matrix
void sgemm_tensorcore_double_buffered_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                           torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_double_buffered_launcher<half, at::Half>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        torch::kFloat16, "float16");
}

// BF16 Tensor Core GEMM launcher with double buffering
// Inputs: matrix_a (M x K, BF16), matrix_b (K x N, BF16)
// Output: output_matrix (M x N, FP32)
// Computes: output_matrix = alpha * (matrix_a @ matrix_b) + beta * output_matrix
void sgemm_tensorcore_double_buffered_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                           torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_double_buffered_launcher<nv_bfloat16, at::BFloat16>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        torch::kBFloat16, "bfloat16");
}
