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

// SGEMM with warp-level tiling (full templatization)
template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int WM = 64, const int WN = 64, const int WNITER = 4,
          const int TM = 8, const int TN = 4, const int NUM_THREADS = 128>
void sgemm_warptiling(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta);

// SGEMM warptiling with default parameters (for Python binding)
void sgemm_warptiling_default(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                              torch::Tensor &output_matrix, float alpha, float beta);
