// Tensor Core GEMM Implementation using WMMA API
// Performs C = alpha * A * B + beta * C where A, B are FP16/BF16 and C is FP32
//
// Key concepts:
// - Uses NVIDIA Tensor Cores via WMMA (Warp Matrix Multiply-Accumulate) API
// - Each warp cooperatively computes multiple 16x16x16 matrix tiles
// - Shared memory staging with optimized layouts for coalesced loads
// - Template parameterization for different precision types (FP16 or BF16)

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

// WMMA tile dimensions: each tensor core operation processes 16x16x16 tiles
constexpr int WMMA_M = 16; // M dimension of output tile
constexpr int WMMA_N = 16; // N dimension of output tile
constexpr int WMMA_K = 16; // K dimension (reduction axis)

// Tensor Core GEMM kernel with warp tiling
// Template parameters:
//   InputType: half (FP16) or nv_bfloat16 (BF16)
//   BLOCK_ROW_WARPS: number of warps along M dimension per block (default 4)
//   BLOCK_COL_WARPS: number of warps along N dimension per block (default 4)
//   WARP_ROW_TILES: number of 16x16 output tiles per warp along M (default 4)
//   WARP_COL_TILES: number of 16x16 output tiles per warp along N (default 2)
//
// With defaults: 16 warps/block, each warp computes 4x2 output tiles = 64x32 elements
// Block computes: (4*4*16) x (4*2*16) = 256x128 output elements
template <typename InputType,
          const int BLOCK_ROW_WARPS = 4,
          const int BLOCK_COL_WARPS = 4,
          const int WARP_ROW_TILES = 4,
          const int WARP_COL_TILES = 2>
__global__ void
sgemm_tensorcore_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
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

    // Shared memory layout:
    // - tile_a: BM x BK (256x16), stored row-major for coalesced A loads
    // - tile_b: BK x BN (16x128), stored COLUMN-major to match WMMA fragment expectation
    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

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

    // Total threads per block = 16 warps * 32 threads/warp = 512 threads
    constexpr int NUM_THREADS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32;

    // Main K-loop: iterate over K dimension in chunks of size BK (16)
    // Each iteration loads new A and B tiles, performs WMMA operations, and accumulates results
    for (int block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        // ===== Phase 1: Cooperative loading of A tile into shared memory =====
        // Load a BM x BK tile of A (256x16 elements) from global memory
        // All threads participate in a strided loop to load the tile
        // Storage: row-major layout tile_a[row * BK + col]
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS)
        {
            int row = idx / BK; // Local row in tile (0 to BM-1)
            int col = idx % BK; // Local col in tile (0 to BK-1)

            // Map to global matrix A coordinates
            int global_row = blockIdx.y * BM + row; // Row in A (M dimension)
            int global_col = block_k_idx + col;     // Col in A (K dimension)

            // Bounds check and load (pad with zeros if out of bounds)
            if (global_row < num_rows_a && global_col < num_cols_a)
            {
                tile_a[row * BK + col] = global_a[global_row * num_cols_a + global_col];
            }
            else
            {
                // Use __float2half or __float2bfloat16 for proper conversion
                tile_a[row * BK + col] = InputType{};  // Zero initialization
            }
        }

        // ===== Phase 2: Cooperative loading of B tile into shared memory =====
        // Load a BK x BN tile of B (16x128 elements) from global memory
        // CRITICAL: Store in COLUMN-MAJOR order for efficient WMMA fragment loading
        // Storage: tile_b[col * BK + row] where each column is contiguous
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS)
        {
            int row = idx / BN; // Local row in tile (0 to BK-1)
            int col = idx % BN; // Local col in tile (0 to BN-1)

            // Map to global matrix B coordinates
            int global_row = block_k_idx + row;     // Row in B (K dimension)
            int global_col = blockIdx.x * BN + col; // Col in B (N dimension)

            // Bounds check and load, store in column-major order
            if (global_row < num_cols_a && global_col < num_cols_b)
            {
                tile_b[col * BK + row] = global_b[global_row * num_cols_b + global_col];
            }
            else
            {
                // Use zero initialization
                tile_b[col * BK + row] = InputType{};  // Zero initialization
            }
        }

        // Synchronize to ensure all tiles are loaded before computation
        __syncthreads();

        // ===== Phase 3: Tensor Core computation =====
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

                // Pointer to A subtile in shared memory (row-major)
                // Subtile starts at row (a_tile_row * WMMA_M), column 0
                // Leading dimension is BK (stride between rows)
                InputType const *a_tile_ptr = tile_a + (a_tile_row * WMMA_M) * BK;

                // Pointer to B subtile in shared memory (column-major)
                // Since tile_b is stored column-major, each column has BK elements
                // We want columns starting at (b_tile_col * WMMA_N)
                InputType const *b_tile_ptr = tile_b + (b_tile_col * WMMA_N) * BK;

                // Load 16x16 A tile from shared memory into WMMA fragment
                // Layout: row-major, leading dimension = BK
                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);

                // Load 16x16 B tile from shared memory into WMMA fragment
                // Layout: column-major, leading dimension = BK (number of rows in col-major)
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);

                // Perform matrix multiply-accumulate: acc = A * B + acc
                // This executes on tensor cores (extremely fast, low precision)
                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        // Synchronize before loading next K-tile (prevents shared memory hazards)
        __syncthreads();
    } // End of K-loop: accumulation complete in acc_frag

    // ===== Phase 4: Write results to global memory =====
    // Store accumulated results from fragments to output matrix C
    // Apply alpha/beta scaling: C = alpha * (A * B) + beta * C
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i)
    {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j)
        {
            // Compute tile position in block (in units of WMMA tiles)
            int c_tile_row = warp_row * WARP_ROW_TILES + i;
            int c_tile_col = warp_col * WARP_COL_TILES + j;

            // Map to global matrix C coordinates (top-left corner of 16x16 tile)
            int global_row = blockIdx.y * BM + c_tile_row * WMMA_M;
            int global_col = blockIdx.x * BN + c_tile_col * WMMA_N;

            // Bounds check: only write if tile starts within valid bounds
            // WMMA store handles partial tiles at boundaries correctly
            if (global_row < num_rows_a && global_col < num_cols_b)
            {
                // Pointer to top-left of this 16x16 output tile in global C
                float *c_ptr = global_c + global_row * num_cols_b + global_col;

                if (beta != 0.0f)
                {
                    // Case: C = alpha * AB + beta * C (need to load existing C values)
                    // Load existing C tile into fragment (row-major, stride = num_cols_b)
                    nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

                    // Apply alpha/beta scaling element-wise
#pragma unroll
                    for (int t = 0; t < c_frag.num_elements; ++t)
                    {
                        c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
                    }

                    // Write result back to global memory
                    nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
                }
                else
                {
                    // Case: C = alpha * AB (beta is zero, no need to load existing C)
                    // Apply alpha scaling to accumulator
#pragma unroll
                    for (int t = 0; t < acc_frag[i][j].num_elements; ++t)
                    {
                        c_frag.x[t] = alpha * acc_frag[i][j].x[t];
                    }

                    // Write result to global memory
                    nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
                }
            }
        }
    }
}

// ============================================================================
// Launcher Functions: FP16 and BF16 variants
// ============================================================================

// FP16 Tensor Core GEMM launcher
// Inputs: matrix_a (M x K, FP16), matrix_b (K x N, FP16)
// Output: output_matrix (M x N, FP32)
// Computes: output_matrix = alpha * (matrix_a @ matrix_b) + beta * output_matrix
void sgemm_tensorcore_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    // Input validation
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat16, "Matrix A must be float16 for Tensor Core kernel");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat16, "Matrix B must be float16 for Tensor Core kernel");
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
    const auto *d_matrix_a = reinterpret_cast<const half *>(matrix_a.data_ptr<at::Half>());
    const auto *d_matrix_b = reinterpret_cast<const half *>(matrix_b.data_ptr<at::Half>());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Kernel configuration (matching template defaults)
    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4;
    constexpr int WARP_COL_TILES = 2;

    // Block tile dimensions in elements
    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M; // 4*4*16 = 256
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N; // 2*4*16 = 128

    // Grid and block dimensions
    dim3 grid_dim(CEIL_DIV(num_cols_b, BN), CEIL_DIV(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32); // 16 warps * 32 = 512 threads

    // Launch kernel
    sgemm_tensorcore_kernel<half, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
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

// BF16 Tensor Core GEMM launcher
// Inputs: matrix_a (M x K, BF16), matrix_b (K x N, BF16)
// Output: output_matrix (M x N, FP32)
// Computes: output_matrix = alpha * (matrix_a @ matrix_b) + beta * output_matrix
void sgemm_tensorcore_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    // Input validation
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kBFloat16, "Matrix A must be bfloat16 for Tensor Core kernel");
    TORCH_CHECK(matrix_b.dtype() == torch::kBFloat16, "Matrix B must be bfloat16 for Tensor Core kernel");
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
    const auto *d_matrix_a = reinterpret_cast<const nv_bfloat16 *>(matrix_a.data_ptr<at::BFloat16>());
    const auto *d_matrix_b = reinterpret_cast<const nv_bfloat16 *>(matrix_b.data_ptr<at::BFloat16>());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Kernel configuration (matching template defaults)
    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4;
    constexpr int WARP_COL_TILES = 2;

    // Block tile dimensions in elements
    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M; // 4*4*16 = 256
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N; // 2*4*16 = 128

    // Grid and block dimensions
    dim3 grid_dim(CEIL_DIV(num_cols_b, BN), CEIL_DIV(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32); // 16 warps * 32 = 512 threads

    // Launch kernel with BF16 input type
    sgemm_tensorcore_kernel<nv_bfloat16, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES>
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
