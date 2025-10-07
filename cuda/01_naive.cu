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
*/

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

__global__ void sgemm_naive_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                   float alpha, const float *matrix_a,
                                   const float *matrix_b, float beta, float *output_matrix)
{
    const uint row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for non-multiple of block size
    if (row_idx < num_rows_a && col_idx < num_cols_b)
    {
        float accumulator = 0.0f;
        for (int k_idx = 0; k_idx < num_cols_a; ++k_idx)
        {
            accumulator += matrix_a[row_idx * num_cols_a + k_idx] *
                           matrix_b[k_idx * num_cols_b + col_idx];
        }
        // C = α*(A@B)+β*C
        const int output_idx = row_idx * num_cols_b + col_idx;
        output_matrix[output_idx] = alpha * accumulator + beta * output_matrix[output_idx];
    }
}

void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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

    // Configure kernel launch: 16x16 threads per block
    constexpr int threads_per_block = 32;
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(CEIL_DIV(num_rows_a, threads_per_block),
                  CEIL_DIV(num_cols_b, threads_per_block));

    // Launch kernel
    sgemm_naive_kernel<<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}