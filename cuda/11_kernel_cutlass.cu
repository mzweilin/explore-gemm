// 11_kernel_cutlass.cu
// CUTLASS-based GEMM kernel for FP16 and BF16
//
// This implementation uses NVIDIA's CUTLASS library to provide highly optimized
// tensor core-based matrix multiplication for half-precision types.
//
// Build requirements:
//  - CUTLASS library (>= v3.x) headers
//  - CUDA toolkit with Tensor Core support (SM >= 75)
//  - PyTorch for tensor management
//
// Configuration matches the double-buffered tensor core kernel:
// - Threadblock: 256 x 128 x 16 (M x N x K)
// - Warp: 64 x 64 x 16
// - Instruction: 16 x 16 x 16 (Tensor Core)

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "gemm_kernels.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include <iostream>
#include <type_traits>

// Convenience macro for ceiling division
#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

// -----------------------------------------------------------------------------
// Type aliases & configuration
// -----------------------------------------------------------------------------

// Output element type (always float32 for accumulation and output)
using ElementOutput = float;
using ElementAccumulator = float;

// Let CUTLASS choose optimal tile shapes based on data types
// These will be configured automatically per instantiation

// -----------------------------------------------------------------------------
// Generic templated CUTLASS GEMM launcher
// -----------------------------------------------------------------------------
template <
    typename ElementA,
    typename ElementB,
    typename ElementC = float,
    typename LayoutA = cutlass::layout::RowMajor,
    typename LayoutB = cutlass::layout::RowMajor,
    typename LayoutCLayout = cutlass::layout::RowMajor>
cudaError_t cutlass_gemm_launch(
    int M, int N, int K,
    const ElementA *d_A, // A: M x K (row-major)
    const ElementB *d_B, // B: K x N (row-major)
    ElementC *d_C,       // C: M x N (row-major, output)
    float alpha,
    float beta,
    cudaStream_t stream = 0)
{
    // Define the GEMM operation with default configurations
    // CUTLASS will automatically select optimal tile sizes and instruction shapes
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,          // Element type and layout for matrix A
        ElementB, LayoutB,          // Element type and layout for matrix B
        ElementC, LayoutCLayout,    // Element type and layout for matrix C/D
        ElementAccumulator          // Accumulator type
        // All other template parameters use defaults which are optimal for the given types
        >;

    // Construct GEMM arguments
    typename Gemm::Arguments args(
        {M, N, K},    // Problem size
        {d_A, K},     // TensorRef for A: pointer and leading dimension
        {d_B, N},     // TensorRef for B: pointer and leading dimension
        {d_C, N},     // TensorRef for C: pointer and leading dimension
        {d_C, N},     // TensorRef for D: pointer and leading dimension (in-place)
        {alpha, beta} // Epilogue scalars
    );

    // Create GEMM operator instance
    Gemm gemm_op;

    // Check if the problem can be implemented with this configuration
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "CUTLASS: Problem cannot be implemented with this configuration: "
                  << cutlass::cutlassGetStatusString(status) << std::endl;
        return cudaErrorNotSupported;
    }

    // Initialize the GEMM operator
    status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "CUTLASS: Initialization failed: "
                  << cutlass::cutlassGetStatusString(status) << std::endl;
        return cudaErrorUnknown;
    }

    // Execute the GEMM operation
    status = gemm_op();
    if (status != cutlass::Status::kSuccess)
    {
        std::cerr << "CUTLASS: Execution failed: "
                  << cutlass::cutlassGetStatusString(status) << std::endl;
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// PyTorch-facing launcher wrappers
// -----------------------------------------------------------------------------

// FP16 launcher
// Input: FP16 matrices A (M x K) and B (K x N)
// Output: FP32 matrix C (M x N)
void sgemm_cutlass_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == at::kHalf, "Matrix A must be float16");
    TORCH_CHECK(matrix_b.scalar_type() == at::kHalf, "Matrix B must be float16");
    TORCH_CHECK(output_matrix.scalar_type() == at::kFloat, "Output matrix must be float32");

    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous(),
                "A and B must be contiguous");
    TORCH_CHECK(output_matrix.is_contiguous(), "Output matrix must be contiguous");

    // Extract dimensions
    int M = static_cast<int>(matrix_a.size(0));
    int K = static_cast<int>(matrix_a.size(1));
    int N = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == K,
                "Matrix dimension mismatch: A is MxK, B is KxN, but B has wrong K");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N,
                "Output matrix has wrong shape");

    // Get device pointers
    const half *d_A = reinterpret_cast<const half *>(matrix_a.data_ptr<at::Half>());
    const half *d_B = reinterpret_cast<const half *>(matrix_b.data_ptr<at::Half>());
    float *d_C = output_matrix.data_ptr<float>();

    // Launch CUTLASS GEMM (using default stream)
    cudaError_t err = cutlass_gemm_launch<
        cutlass::half_t, cutlass::half_t, float,
        cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
        M, N, K,
        reinterpret_cast<const cutlass::half_t *>(d_A),
        reinterpret_cast<const cutlass::half_t *>(d_B),
        d_C,
        alpha, beta,
        0);  // Use default CUDA stream

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS GEMM (FP16) failed with error: ", cudaGetErrorString(err));
}

// BF16 launcher
// Input: BF16 matrices A (M x K) and B (K x N)
// Output: FP32 matrix C (M x N)
void sgemm_cutlass_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == at::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.scalar_type() == at::kBFloat16, "Matrix B must be bfloat16");
    TORCH_CHECK(output_matrix.scalar_type() == at::kFloat, "Output matrix must be float32");

    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous(),
                "A and B must be contiguous");
    TORCH_CHECK(output_matrix.is_contiguous(), "Output matrix must be contiguous");

    // Extract dimensions
    int M = static_cast<int>(matrix_a.size(0));
    int K = static_cast<int>(matrix_a.size(1));
    int N = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == K,
                "Matrix dimension mismatch: A is MxK, B is KxN, but B has wrong K");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N,
                "Output matrix has wrong shape");

    // Get device pointers
    const nv_bfloat16 *d_A_raw = reinterpret_cast<const nv_bfloat16 *>(
        matrix_a.data_ptr<at::BFloat16>());
    const nv_bfloat16 *d_B_raw = reinterpret_cast<const nv_bfloat16 *>(
        matrix_b.data_ptr<at::BFloat16>());
    float *d_C = output_matrix.data_ptr<float>();

    // Cast to CUTLASS bfloat16_t type
    const cutlass::bfloat16_t *d_A = reinterpret_cast<const cutlass::bfloat16_t *>(d_A_raw);
    const cutlass::bfloat16_t *d_B = reinterpret_cast<const cutlass::bfloat16_t *>(d_B_raw);

    // Launch CUTLASS GEMM (using default stream)
    cudaError_t err = cutlass_gemm_launch<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
        M, N, K,
        d_A, d_B, d_C,
        alpha, beta,
        0);  // Use default CUDA stream

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS GEMM (BF16) failed with error: ", cudaGetErrorString(err));
}
