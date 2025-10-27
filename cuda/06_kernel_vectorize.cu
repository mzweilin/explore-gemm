#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                       float alpha, const float *matrix_a,
                                       const float *matrix_b, float beta,
                                       float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // Thread indices for computing output tile
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Shared memory tiles - stored in column-major for A to enable coalescing
    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    // Position matrix pointers at the start of this block's tile
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    // Thread indices for vectorized loading
    // Load 4 floats at a time using float4
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4);
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4);

    // Allocate register storage
    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    // Outer loop over K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {

        // Load tile_a using float4 vectorized loads
        // Store in transposed layout: tile_a[col][row] for coalesced shared memory access
        float4 tmp_a = reinterpret_cast<const float4*>(
            &matrix_a[inner_row_a * num_cols_a + inner_col_a * 4])[0];
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a] = tmp_a.w;

        // Load tile_b using float4 vectorized loads
        // Store in row-major layout: tile_b[row][col]
        float4 tmp_b = reinterpret_cast<const float4*>(
            &matrix_b[inner_row_b * num_cols_b + inner_col_b * 4])[0];
        tile_b[inner_row_b * BN + inner_col_b * 4 + 0] = tmp_b.x;
        tile_b[inner_row_b * BN + inner_col_b * 4 + 1] = tmp_b.y;
        tile_b[inner_row_b * BN + inner_col_b * 4 + 2] = tmp_b.z;
        tile_b[inner_row_b * BN + inner_col_b * 4 + 3] = tmp_b.w;

        __syncthreads();

        // Advance pointers for next tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load TM elements from tile_a (transposed layout)
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                register_m[i] = tile_a[dot_idx * BM + thread_row * TM + i];
            }

            // Load TN elements from tile_b
            #pragma unroll
            for (uint i = 0; i < TN; ++i) {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            // Outer product accumulation
            #pragma unroll
            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                #pragma unroll
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

    // Write results with alpha/beta scaling
    #pragma unroll
    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        #pragma unroll
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                               (thread_col * TN + res_idx_n);
            matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                              beta * matrix_c[c_idx];
        }
    }
}

void sgemm_vectorize(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Validate dimensions are multiples of tile sizes
    TORCH_CHECK(num_rows_a % BM == 0, "Matrix A rows must be multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0, "Matrix A cols must be multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0, "Matrix B cols must be multiple of ", BN);

    // Configure kernel launch
    dim3 block_dim((BM / TM) * (BN / TN));
    dim3 grid_dim(num_cols_b / BN, num_rows_a / BM);

    // Launch kernel
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
