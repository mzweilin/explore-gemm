#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

template <const uint block_size>
__global__ void sgemm_shared_mem_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                        float alpha, const float *matrix_a,
                                        const float *matrix_b, float beta,
                                        float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[block_size * block_size];
    __shared__ float tile_b[block_size * block_size];

    const uint thread_row = threadIdx.x / block_size;
    const uint thread_col = threadIdx.x % block_size;

    // Calculate global row and column indices for this thread
    const uint global_row = block_row * block_size + thread_row;
    const uint global_col = block_col * block_size + thread_col;

    // Move pointers to the starting position for this block
    matrix_a += block_row * block_size * num_cols_a; // row=block_row, col=0
    matrix_b += block_col * block_size;              // row=0, col=block_col
    matrix_c += block_row * block_size * num_cols_b + block_col * block_size;

    float accumulator = 0.0f;

    // Loop over all tiles along K dimension
    for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += block_size)
    {
        // Load tile from matrix A into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if (global_row < num_rows_a && (tile_idx + thread_col) < num_cols_a)
        {
            tile_a[thread_row * block_size + thread_col] =
                matrix_a[thread_row * num_cols_a + thread_col];
        }
        else
        {
            tile_a[thread_row * block_size + thread_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if ((tile_idx + thread_row) < num_cols_a && global_col < num_cols_b)
        {
            tile_b[thread_row * block_size + thread_col] =
                matrix_b[thread_row * num_cols_b + thread_col];
        }
        else
        {
            tile_b[thread_row * block_size + thread_col] = 0.0f;
        }

        // Block threads until cache is fully populated
        __syncthreads();

        // Advance pointers to next tile
        matrix_a += block_size;
        matrix_b += block_size * num_cols_b;

        // Compute partial dot product using shared memory
        for (int dot_idx = 0; dot_idx < block_size; ++dot_idx)
        {
            accumulator += tile_a[thread_row * block_size + dot_idx] *
                           tile_b[dot_idx * block_size + thread_col];
        }

        // Sync again to avoid faster threads fetching next block before slower threads finish
        __syncthreads();
    }

    // Write result to global memory with bounds checking: C = α*(A@B)+β*C
    if (global_row < num_rows_a && global_col < num_cols_b)
    {
        matrix_c[thread_row * num_cols_b + thread_col] =
            alpha * accumulator + beta * matrix_c[thread_row * num_cols_b + thread_col];
    }
}

void sgemm_shared_mem(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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
    auto *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 1D blocks with block_size^2 threads (32x32 = 1024 threads per block)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(ceil_div(num_rows_a, block_size),
                  ceil_div(num_cols_b, block_size));

    // Launch kernel
    sgemm_shared_mem_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}