/*
Warptiling GEMM with FP16/BF16/FP32 support.
Uses FP32 accumulation internally, converts to output dtype on store.
*/

#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"


template <typename T>
struct VecType {};
template <>
struct VecType<half> { using type = half2; };
template <>
struct VecType<nv_bfloat16> { using type = nv_bfloat162; };

template <typename T>
constexpr int vec_size() { return 2; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ half from_float(float x, half) { return __float2half(x); }
__device__ __forceinline__ nv_bfloat16 from_float(float x, nv_bfloat16) { return __float2bfloat16(x); }
template <typename InputType, const int BM, const int BN, const int BK,
          const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const InputType *matrix_a, const InputType *matrix_b,
                               InputType *tile_a, InputType *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    constexpr int VEC_SIZE = 2;
    using VecT = typename VecType<InputType>::type;

    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a)
    {
        const VecT tmp_a = reinterpret_cast<const VecT *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * VEC_SIZE])[0];
        tile_a[(inner_col_a * 2 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 2 + 1) * BM + inner_row_a + offset] = tmp_a.y;
    }

    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b)
    {
        reinterpret_cast<VecT *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * VEC_SIZE])[0] =
            reinterpret_cast<const VecT *>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * VEC_SIZE])[0];
    }
}
template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const InputType *tile_a, const InputType *tile_b,
                                  const uint warp_row, const uint warp_col,
                                  const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[wsub_row_idx * TM + i] =
                    to_float(tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                                    thread_row_in_warp * TM + i]);
            }
        }

        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            for (uint i = 0; i < TN; ++i)
            {
                register_n[wsub_col_idx * TN + i] =
                    to_float(tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                                    thread_col_in_warp * TN + i]);
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

    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
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

template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_multidtype_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                       float alpha, const InputType *matrix_a, const InputType *matrix_b,
                                       float beta, InputType *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    constexpr int VEC_SIZE = vec_size<InputType>();
    const uint inner_row_a = threadIdx.x / (BK / VEC_SIZE);
    const uint inner_col_a = threadIdx.x % (BK / VEC_SIZE);
    constexpr uint row_stride_a = (NUM_THREADS * VEC_SIZE) / BK;

    const uint inner_row_b = threadIdx.x / (BN / VEC_SIZE);
    const uint inner_col_b = threadIdx.x % (BN / VEC_SIZE);
    constexpr uint row_stride_b = NUM_THREADS / (BN / VEC_SIZE);

    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        load_from_gmem<InputType, BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        process_warp_tile<InputType, BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
    {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            InputType *matrix_c_interim = matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                                          wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                {
                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                        wsub_col_idx * TN + res_idx_n;

                    InputType c_val = matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                                       thread_col_in_warp * TN + res_idx_n];
                    float result = alpha * thread_results[res_idx] + beta * to_float(c_val);
                    matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                     thread_col_in_warp * TN + res_idx_n] = from_float(result, InputType{});
                }
            }
        }
    }
}
template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
void sgemm_warptiling_multidtype(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == matrix_a.dtype(),
                "Matrix C must have same dtype as input matrices");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    TORCH_CHECK(num_rows_a % BM == 0, "Matrix A rows must be multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0, "Matrix A cols must be multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0, "Matrix B cols must be multiple of ", BN);

    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    static_assert(WMITER * WSUBM == WM, "WMITER * WSUBM must equal WM");
    static_assert(WNITER * WSUBN == WN, "WNITER * WSUBN must equal WN");
    static_assert((BM % WM == 0) && (BN % WN == 0), "Block tile must be divisible by warp tile");
    static_assert((WSUBM % TM == 0) && (WSUBN % TN == 0), "Warp subtile must be divisible by thread tile");

    const InputType *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr());
    const InputType *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr());
    InputType *d_output_matrix = reinterpret_cast<InputType *>(output_matrix.data_ptr());

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));

    sgemm_warptiling_multidtype_kernel<InputType, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
void sgemm_warptiling_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat16, "Matrix A must be float16");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat16, "Matrix B must be float16");
    sgemm_warptiling_multidtype<half, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

void sgemm_warptiling_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.dtype() == torch::kBFloat16, "Matrix B must be bfloat16");
    sgemm_warptiling_multidtype<nv_bfloat16, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}
