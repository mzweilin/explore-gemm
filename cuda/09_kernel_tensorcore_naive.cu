#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <typename InputType>
__global__ void
sgemm_tensorcore_naive_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                              float alpha, const InputType *matrix_a,
                              const InputType *matrix_b, float beta,
                              InputType *matrix_c)
{
    // Each warp computes one 16x16 WMMA tile of the output
    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;
    __shared__ float c_tile_smem[WMMA_M * WMMA_N];

    // Accumulator fragment for output (FP32)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    const size_t K_tiles = (num_cols_a + WMMA_K - 1) / WMMA_K;

    for (size_t k = 0; k < K_tiles; ++k)
    {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> b_frag;

        const InputType *a_ptr = matrix_a + warp_row * num_cols_a + k * WMMA_K;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, num_cols_a);

        const InputType *b_ptr = matrix_b + k * WMMA_K * num_cols_b + warp_col;
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, num_cols_b);

        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Load current C tile for beta scaling
    InputType *c_ptr = matrix_c + warp_row * num_cols_b + warp_col;
    if (beta != 0.0f)
    {
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_load_frag;
        load_output_tile_to_smem<InputType, WMMA_M, WMMA_N>(c_ptr, num_cols_b, c_tile_smem);
        __syncwarp();
        nvcuda::wmma::load_matrix_sync(c_load_frag, c_tile_smem, WMMA_N, nvcuda::wmma::mem_row_major);

#pragma unroll
        for (int t = 0; t < c_frag.num_elements; ++t)
        {
            c_frag.x[t] = alpha * c_frag.x[t] + beta * c_load_frag.x[t];
        }
    }
    else
    {
#pragma unroll
        for (int t = 0; t < c_frag.num_elements; ++t)
        {
            c_frag.x[t] *= alpha;
        }
    }

    nvcuda::wmma::store_matrix_sync(c_tile_smem, c_frag, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncwarp();
    store_output_tile_from_smem<InputType, WMMA_M, WMMA_N>(c_tile_smem, c_ptr, num_cols_b);
}

template <typename InputType, typename TorchType>
void sgemm_tensorcore_naive_impl(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta,
                                 torch::ScalarType expected_dtype)
{
    TORCH_CHECK(matrix_a.device().is_cuda() && matrix_b.device().is_cuda(), "Matrices must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == expected_dtype && matrix_b.dtype() == expected_dtype, "Input dtype mismatch");
    TORCH_CHECK(output_matrix.dtype() == expected_dtype, "Matrix C must have the same dtype as the inputs");
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "Matrices must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a && output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix dimensions must match");

    const InputType *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr<TorchType>());
    const InputType *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr<TorchType>());
    InputType *d_output_matrix = reinterpret_cast<InputType *>(output_matrix.data_ptr<TorchType>());

    // Each block is a single warp (32 threads)
    // Grid dimensions: (num_cols_b / WMMA_N, num_rows_a / WMMA_M)
    dim3 grid_dim(ceil_div(num_cols_b, WMMA_N), ceil_div(num_rows_a, WMMA_M));
    dim3 block_dim(32); // Single warp per block

    sgemm_tensorcore_naive_kernel<InputType><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

void sgemm_tensorcore_naive_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_naive_impl<half, at::Half>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kFloat16);
}

void sgemm_tensorcore_naive_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta)
{
    sgemm_tensorcore_naive_impl<nv_bfloat16, at::BFloat16>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kBFloat16);
}
