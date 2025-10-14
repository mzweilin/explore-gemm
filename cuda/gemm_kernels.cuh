#pragma once
#include <torch/torch.h>

// Naive SGEMM implementation
void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                 torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with global memory coalescing
void sgemm_global_mem_coalesce(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                               torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with shared memory tiling
void sgemm_shared_mem(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with 1D block tiling
void sgemm_blocktiling_1d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with 2D block tiling
void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM with vectorized memory access
void sgemm_vectorize(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                     torch::Tensor &output_matrix, float alpha, float beta);
