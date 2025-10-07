#pragma once
#include <torch/torch.h>

// Naive SGEMM implementation
torch::Tensor sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          float alpha = 1.0f, float beta = 0.0f,
                          const torch::Tensor &matrix_c = torch::Tensor());

// SGEMM with global memory coalescing
torch::Tensor sgemm_global_mem_coalesce(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                        float alpha = 1.0f, float beta = 0.0f,
                                        const torch::Tensor &matrix_c = torch::Tensor());
