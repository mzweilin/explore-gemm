#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"

/*
Matrix sizes:
A: M x K
B: K x N
C: M x N

C = alpha * (A @ B) + beta * C

This kernel uses 2D block tiling to further improve performance over 1D block tiling.
Each thread computes a TM x TN tile of results (2D tile), enabling even better register reuse
and reducing the number of shared memory accesses compared to 1D tiling.

Key improvements over 1D block tiling:
- Each thread computes TM x TN results instead of just TM results
- Loads elements from A into registers (register_m array) and reuses them across TN computations
- Loads elements from B into registers (register_n array) and reuses them across TM computations
- This creates a 2D register blocking pattern that maximizes arithmetic intensity
*/

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                            float alpha, const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    // Shared memory tiles for A and B
    // tile_a: BM x BK (block tile from matrix A)
    // tile_b: BK x BN (block tile from matrix B)
    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    // Calculate thread position within the block
    // Each thread is responsible for computing a TM x TN output tile
    // Total threads per block = (BM / TM) * (BN / TN)
    const uint thread_row = threadIdx.x / (BN / TN);  // Which row of thread tiles
    const uint thread_col = threadIdx.x % (BN / TN);  // Which column of thread tiles

    // Thread count and loading strategy
    // We have (BM/TM) * (BN/TN) = 64 threads
    // tile_a needs BM * BK = 64 * 8 = 512 elements
    // tile_b needs BK * BN = 8 * 64 = 512 elements
    // Each thread must load 512/64 = 8 elements from each tile
    const uint num_threads = (BM / TM) * (BN / TN);

    // Position input/output matrix pointers at the start of this block's tile
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    // Allocate thread-local storage in registers for:
    // 1. Final results: TM x TN output values this thread computes
    // 2. register_m: TM values from matrix A (reused across TN computations)
    // 3. register_n: TN values from matrix B (reused across TM computations)
    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    // Outer loop over block tiles along K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // ==================== LOAD TILES INTO SHARED MEMORY ====================

        // Load tile from matrix A into shared memory with bounds checking
        // Layout: tile_a is BM x BK (64 x 8 = 512 elements)
        // With 64 threads, each thread loads 512/64 = 8 elements
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads) {
            uint load_idx = threadIdx.x + load_offset;
            if (load_idx < BM * BK) {
                uint a_row = load_idx / BK;
                uint a_col = load_idx % BK;
                uint global_row_a = block_row * BM + a_row;
                uint global_col_a = block_k_idx + a_col;
                if (global_row_a < num_rows_a && global_col_a < num_cols_a) {
                    tile_a[load_idx] = matrix_a[a_row * num_cols_a + a_col];
                } else {
                    tile_a[load_idx] = 0.0f;
                }
            }
        }

        // Load tile from matrix B into shared memory with bounds checking
        // Layout: tile_b is BK x BN (8 x 64 = 512 elements)
        // With 64 threads, each thread loads 512/64 = 8 elements
        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads) {
            uint load_idx = threadIdx.x + load_offset;
            if (load_idx < BK * BN) {
                uint b_row = load_idx / BN;
                uint b_col = load_idx % BN;
                uint global_row_b = block_k_idx + b_row;
                uint global_col_b = block_col * BN + b_col;
                if (global_row_b < num_cols_a && global_col_b < num_cols_b) {
                    tile_b[load_idx] = matrix_b[b_row * num_cols_b + b_col];
                } else {
                    tile_b[load_idx] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Advance block tile pointers for next iteration
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        // ==================== COMPUTE USING REGISTER BLOCKING ====================

        // For each element along the K dimension of the current block tile
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load TM elements from tile_a into registers
            // These are the elements in column dot_idx, rows [thread_row*TM : thread_row*TM+TM)
            // We load these once and reuse them for all TN columns
            for (uint i = 0; i < TM; ++i) {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            // Load TN elements from tile_b into registers
            // These are the elements in row dot_idx, columns [thread_col*TN : thread_col*TN+TN)
            // We load these once and reuse them for all TM rows
            for (uint i = 0; i < TN; ++i) {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            // Compute outer product of register_m and register_n, accumulating into thread_results
            // This is the key 2D blocking: we compute TM x TN results using cached values
            // For each result position (res_m, res_n), compute: result += register_m[res_m] * register_n[res_n]
            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                    // Store in row-major order: thread_results[res_idx_m * TN + res_idx_n]
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

    // ==================== WRITE RESULTS TO GLOBAL MEMORY ====================

    // Write the TM x TN tile of results computed by this thread back to global memory
    // Apply scaling: C = alpha * (A @ B) + beta * C
    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            // Calculate global position for this result
            const uint global_row = block_row * BM + thread_row * TM + res_idx_m;
            const uint global_col = block_col * BN + thread_col * TN + res_idx_n;

            // Bounds check before writing
            if (global_row < num_rows_a && global_col < num_cols_b) {
                const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                                   (thread_col * TN + res_idx_n);
                matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                                  beta * matrix_c[c_idx];
            }
        }
    }
}

void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Template parameters for kernel
    // BM, BN: Block tile dimensions (64x64 output block per thread block)
    // BK: Inner dimension block size (8 elements processed per iteration)
    // TM, TN: Thread tile dimensions (8x8 output tile per thread)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Configure kernel launch
    // Number of threads = (BM / TM) * (BN / TN) = (64 / 8) * (64 / 8) = 8 * 8 = 64 threads per block
    dim3 block_dim((BM / TM) * (BN / TN));
    dim3 grid_dim(CEIL_DIV(num_rows_a, BM),
                  CEIL_DIV(num_cols_b, BN));

    // Launch kernel
    sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
