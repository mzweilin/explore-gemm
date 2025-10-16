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

This kernel implements true warp-level tiling for GEMM operations.
Key insight: Each warp computes a WM x WN tile, which is further divided into
warp subtiles (WSUBM x WSUBN) that are processed by threads within the warp.

Key features:
- Block tile (BM x BN): Cached in shared memory
- Warp tile (WM x WN): Computed by a single warp (32 threads)
- Warp subtile (WSUBM x WSUBN): Computed by threads in iterations
- Thread tile (TM x TN): Computed by each thread in registers

Hierarchy:
Block (BM x BN) → Multiple Warps → Each Warp (WM x WN) → Warp Subtiles (WSUBM x WSUBN) → Thread Tiles (TM x TN)

Template Parameters:
- BM, BN, BK: Block tile dimensions (shared memory)
- WM, WN: Warp tile dimensions (what one warp computes)
- WNITER: Number of warp subtile iterations in N dimension
- TM, TN: Thread tile dimensions (registers)
- NUM_THREADS: Total threads per block (typically 128 or 256)

WMITER is computed: (WM * WN) / (WARPSIZE * TM * TN * WNITER)
WSUBM = WM / WMITER (warp subtile M dimension)
WSUBN = WN / WNITER (warp subtile N dimension)
*/

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

constexpr int WARPSIZE = 32;

// ==================== HELPER FUNCTIONS ====================

// Load data from global memory to shared memory with vectorized access
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const float *matrix_a, const float *matrix_b,
                               float *tile_a, float *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    // Load tile_a with float4 vectorized loads and transpose
    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a) {
        const float4 tmp_a = reinterpret_cast<const float4*>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];
        // Transpose while storing to shared memory
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
    }

    // Load tile_b with float4 vectorized loads
    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b) {
        reinterpret_cast<float4*>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4*>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
    }
}

// Process warptile: compute using warp subtiling
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                   const float *tile_a, const float *tile_b,
                                   const uint warp_row, const uint warp_col,
                                   const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    // Loop over BK dimension
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        // Populate registers for entire warptile
        // Load WMITER * TM elements from tile_a (covers all warp subtiles in M dimension)
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint i = 0; i < TM; ++i) {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
            }
        }

        // Load WNITER * TN elements from tile_b (covers all warp subtiles in N dimension)
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
            }
        }

        // Execute warptile matmul across all warp subtiles
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
                // Calculate per-thread results for this warp subtile
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                        thread_results[(wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                      (wsub_col_idx * TN) + res_idx_n] +=
                            register_m[wsub_row_idx * TM + res_idx_m] *
                            register_n[wsub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}

// ==================== WARPTILING KERNEL ====================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_warptiling_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                        float alpha, const float *matrix_a, const float *matrix_b,
                        float beta, float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // Warp-level placement within threadblock
    const uint warp_idx = threadIdx.x / WARPSIZE;          // Which warp this thread belongs to
    const uint warp_col = warp_idx % (BN / WN);            // Warp's column in block tile
    const uint warp_row = warp_idx / (BN / WN);            // Warp's row in block tile

    // Warp subtile dimensions
    // WMITER: number of subtile iterations in M dimension per warp
    // Formula: total warp work (WM*WN) / work per thread per iteration (WARPSIZE*TM*TN*WNITER)
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;  // Warp subtile height
    constexpr uint WSUBN = WN / WNITER;  // Warp subtile width

    // Thread placement within warp subtile
    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;           // [0, 31]
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN); // Column within subtile
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN); // Row within subtile

    // Shared memory for block tiles
    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    // Position matrix pointers at start of this block's tile
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    // Position output pointer at this warp's tile
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    // Thread indices for loading data into shared memory
    // Load 4 floats at a time using float4
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4);
    constexpr uint row_stride_a = (NUM_THREADS * 4) / BK;

    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4);
    constexpr uint row_stride_b = NUM_THREADS / (BN / 4);

    // Thread-local storage in registers
    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    // Cache for warptile computation
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    // Outer loop over block tiles along K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // Load block tile from global memory to shared memory
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        // Process warptile from shared memory
        process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        // Advance to next block tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    // ==================== WRITE RESULTS TO GLOBAL MEMORY ====================

    // Write results for each warp subtile with vectorized stores
    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            // Move pointer to current warp subtile
            float *matrix_c_interim = matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                                     wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1) {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
                    // Load C vector into registers
                    float4 tmp_c = reinterpret_cast<float4*>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                         thread_col_in_warp * TN + res_idx_n])[0];

                    // Perform GEMM update in registers
                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                       wsub_col_idx * TN + res_idx_n;
                    tmp_c.x = alpha * thread_results[res_idx + 0] + beta * tmp_c.x;
                    tmp_c.y = alpha * thread_results[res_idx + 1] + beta * tmp_c.y;
                    tmp_c.z = alpha * thread_results[res_idx + 2] + beta * tmp_c.z;
                    tmp_c.w = alpha * thread_results[res_idx + 3] + beta * tmp_c.w;

                    // Write back with vectorized store
                    reinterpret_cast<float4*>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                         thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
        }
    }
}

// ==================== LAUNCHER FUNCTION ====================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
void sgemm_warptiling(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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

    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    // Validate dimensions are multiples of tile sizes
    TORCH_CHECK(num_rows_a % BM == 0, "Matrix A rows must be multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0, "Matrix A cols must be multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0, "Matrix B cols must be multiple of ", BN);

    // Validate warptiling constraints
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    static_assert(WMITER * WSUBM == WM, "WMITER * WSUBM must equal WM");
    static_assert(WNITER * WSUBN == WN, "WNITER * WSUBN must equal WN");
    static_assert((BM % WM == 0) && (BN % WN == 0), "Block tile must be divisible by warp tile");
    static_assert((WSUBM % TM == 0) && (WSUBN % TN == 0), "Warp subtile must be divisible by thread tile");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch
    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(num_cols_b, BN), CEIL_DIV(num_rows_a, BM));

    // Launch kernel
    sgemm_warptiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

// Default configuration wrapper
void sgemm_warptiling_default(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                               torch::Tensor &output_matrix, float alpha, float beta)
{
    // Default configuration: BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, NUM_THREADS=128
    // This gives: WMITER=2, WSUBM=32, WSUBN=16
    // Warps per block: (128*128)/(64*64) = 4 warps
    // Threads needed: 4 warps * 32 threads/warp = 128 threads
    sgemm_warptiling<128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

// Explicit template instantiations for commonly used configurations
// Configuration 1: BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, NUM_THREADS=128
template void sgemm_warptiling<128, 128, 16, 64, 64, 4, 8, 4, 128>(
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&, float, float);

// Configuration 2: BM=128, BN=128, BK=16, WM=64, WN=32, WNITER=2, TM=8, TN=4, NUM_THREADS=256
// WMITER = (64*32)/(32*8*4*2) = 2048/2048 = 1, WSUBM=64, WSUBN=16
template void sgemm_warptiling<128, 128, 16, 64, 32, 2, 8, 4, 256>(
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&, float, float);

// Configuration 3: BM=64, BN=64, BK=16, WM=32, WN=32, WNITER=2, TM=4, TN=4, NUM_THREADS=64
template void sgemm_warptiling<64, 64, 16, 32, 32, 2, 4, 4, 64>(
    const torch::Tensor&, const torch::Tensor&, torch::Tensor&, float, float);
