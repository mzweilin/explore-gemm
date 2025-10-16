// Warptiling GEMM with Multi-Dtype Support (FP32, FP16, BF16)
// This kernel extends the warptiling approach to support multiple data types
// for inputs (A, B) while keeping the output (C) as FP32 for numerical stability.
//
// Key features:
// - Template parameter for input dtype: float, half, or nv_bfloat16
// - Output always FP32 for accumulation precision
// - Vectorized loads with dtype-specific vector types
// - Same warp-level tiling strategy as 07_kernel_warptiling.cu
//
// Hierarchy:
// Block (BM x BN) → Multiple Warps → Each Warp (WM x WN) → Warp Subtiles (WSUBM x WSUBN) → Thread Tiles (TM x TN)

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"

// Use the same CEIL_DIV and WARPSIZE definitions as other kernels
// (they may already be defined when combining source files)
#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

#ifndef WARPSIZE_
constexpr int WARPSIZE = 32;
#define WARPSIZE_ 32
#endif

// ==================== TYPE TRAITS FOR VECTORIZATION ====================

// Helper to get the vectorized type for each input type
template <typename T>
struct VecType
{
};
template <>
struct VecType<float>
{
    using type = float4;
};
template <>
struct VecType<half>
{
    using type = half2;
}; // Load 2 halfs at a time
template <>
struct VecType<nv_bfloat16>
{
    using type = nv_bfloat162;
}; // Load 2 bf16s at a time

// Vector size in elements
template <typename T>
constexpr int vec_size() { return 4; }
template <>
constexpr int vec_size<half>() { return 2; }
template <>
constexpr int vec_size<nv_bfloat16>() { return 2; }

// Type conversion helper for 16-bit types to float
// (works even when conversion operators are disabled by PyTorch macros)
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(nv_bfloat16 x) { return __bfloat162float(x); }

// ==================== HELPER FUNCTIONS ====================

// Load data from global memory to shared memory with dtype-specific vectorized access
// For FP32: use float4 (4 elements)
// For FP16/BF16: use half2/bfloat162 (2 elements)
template <typename InputType, const int BM, const int BN, const int BK,
          const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const InputType *matrix_a, const InputType *matrix_b,
                               InputType *tile_a, InputType *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    constexpr int VEC_SIZE = vec_size<InputType>();
    using VecT = typename VecType<InputType>::type;

    // Load tile_a with vectorized loads and transpose
    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a)
    {
        const VecT tmp_a = reinterpret_cast<const VecT *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * VEC_SIZE])[0];

        // Transpose while storing to shared memory
        if constexpr (VEC_SIZE == 4)
        {
            // FP32 case
            tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
            tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
            tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
            tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
        }
        else
        {
            // FP16/BF16 case (half2 or bfloat162)
            tile_a[(inner_col_a * 2 + 0) * BM + inner_row_a + offset] = tmp_a.x;
            tile_a[(inner_col_a * 2 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        }
    }

    // Load tile_b with vectorized loads
    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b)
    {
        reinterpret_cast<VecT *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * VEC_SIZE])[0] =
            reinterpret_cast<const VecT *>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * VEC_SIZE])[0];
    }
}

// Process warptile: compute using warp subtiling
// InputType for loads, float for accumulation
template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const InputType *tile_a, const InputType *tile_b,
                                  const uint warp_row, const uint warp_col,
                                  const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    // Loop over BK dimension
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
        // Populate registers for entire warptile
        // Load WMITER * TM elements from tile_a (covers all warp subtiles in M dimension)
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                // Convert to float for computation (works for float, half, nv_bfloat16)
                register_m[wsub_row_idx * TM + i] = to_float(
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i]);
            }
        }

        // Load WNITER * TN elements from tile_b (covers all warp subtiles in N dimension)
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            for (uint i = 0; i < TN; ++i)
            {
                // Convert to float for computation (works for float, half, nv_bfloat16)
                register_n[wsub_col_idx * TN + i] = to_float(
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i]);
            }
        }

        // Execute warptile matmul across all warp subtiles
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
            {
                // Calculate per-thread results for this warp subtile
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
                {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                    {
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

// Specialization for float (no conversion needed)
template <>
__device__ void process_warp_tile<float, 128, 128, 16, 64, 64, 2, 4, 32, 16, 8, 4>(
    float *register_m, float *register_n, float *thread_results,
    const float *tile_a, const float *tile_b,
    const uint warp_row, const uint warp_col,
    const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 64;
    constexpr int WMITER = 2, WNITER = 4;
    constexpr int WSUBM = 32, WSUBN = 16;
    constexpr int TM = 8, TN = 4;

    // Loop over BK dimension
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
        // Populate registers for entire warptile
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
            }
        }

        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            for (uint i = 0; i < TN; ++i)
            {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
            }
        }

        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
            {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
                {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                    {
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

// ==================== WARPTILING KERNEL (MULTI-DTYPE) ====================

template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_multidtype_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                       float alpha, const InputType *matrix_a, const InputType *matrix_b,
                                       float beta, float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // Warp-level placement within threadblock
    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    // Warp subtile dimensions
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    // Thread placement within warp subtile
    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // Shared memory for block tiles
    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

    // Position matrix pointers
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    // Thread indices for loading data (vectorized)
    constexpr int VEC_SIZE = vec_size<InputType>();
    const uint inner_row_a = threadIdx.x / (BK / VEC_SIZE);
    const uint inner_col_a = threadIdx.x % (BK / VEC_SIZE);
    constexpr uint row_stride_a = (NUM_THREADS * VEC_SIZE) / BK;

    const uint inner_row_b = threadIdx.x / (BN / VEC_SIZE);
    const uint inner_col_b = threadIdx.x % (BN / VEC_SIZE);
    constexpr uint row_stride_b = NUM_THREADS / (BN / VEC_SIZE);

    // Thread-local storage in registers (always FP32 for accumulation)
    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    // Outer loop over block tiles along K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        // Load block tile from global memory to shared memory
        load_from_gmem<InputType, BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        // Process warptile from shared memory
        process_warp_tile<InputType, BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        // Advance to next block tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    // ==================== WRITE RESULTS TO GLOBAL MEMORY ====================

    // Write results for each warp subtile with vectorized stores (always FP32 output)
    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
    {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            float *matrix_c_interim = matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                                      wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4)
                {
                    // Load C vector into registers
                    float4 tmp_c = reinterpret_cast<float4 *>(
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
                    reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                          thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
        }
    }
}

// ==================== LAUNCHER FUNCTIONS ====================

// Generic launcher template
template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
void sgemm_warptiling_multidtype(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
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
    const InputType *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr());
    const InputType *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch
    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(num_cols_b, BN), CEIL_DIV(num_rows_a, BM));

    // Launch kernel
    sgemm_warptiling_multidtype_kernel<InputType, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

// ==================== PUBLIC API FUNCTIONS ====================

// FP32 version - delegate to the original FP32-only warptiling kernel
// (declared in 07_kernel_warptiling.cu)
extern void sgemm_warptiling_default(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                     torch::Tensor &output_matrix, float alpha, float beta);

void sgemm_warptiling_fp32(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");

    // Delegate to the original FP32 warptiling kernel (no conversion overhead)
    sgemm_warptiling_default(matrix_a, matrix_b, output_matrix, alpha, beta);
}

// FP16 version - use the multi-dtype kernel (16-bit only)
void sgemm_warptiling_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat16, "Matrix A must be float16");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat16, "Matrix B must be float16");

    sgemm_warptiling_multidtype<half, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

// BF16 version - use the multi-dtype kernel (16-bit only)
void sgemm_warptiling_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.dtype() == torch::kBFloat16, "Matrix B must be bfloat16");

    sgemm_warptiling_multidtype<nv_bfloat16, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}
