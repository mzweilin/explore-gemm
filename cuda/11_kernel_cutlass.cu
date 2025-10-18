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
// Configuration:
// - Threadblock: 128 x 128 x 32 (M x N x K)
// - Warp: 64 x 64 x 32
// - Instruction: 16 x 8 x 16 (Tensor Core)
// - Pipeline stages: 2 (double buffering)

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "gemm_kernels.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"

#include <mutex>
#include <map>
#include <tuple>
#include <memory>

// Convenience macro for ceiling division
#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

// -----------------------------------------------------------------------------
// Type aliases & configuration
// -----------------------------------------------------------------------------

using ElementAccumulator = float;       // Accumulate in FP32
using ElementCompute = float;           // Epilogue compute type

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Threadblock, warp, and instruction shapes
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// -----------------------------------------------------------------------------
// FP16 GEMM Configuration
// -----------------------------------------------------------------------------
using ElementA_FP16 = cutlass::half_t;
using ElementB_FP16 = cutlass::half_t;
using ElementC_FP16 = float;  // Output in FP32

using EpilogueOp_FP16 = cutlass::epilogue::thread::LinearCombination<
    ElementC_FP16,
    128 / cutlass::sizeof_bits<ElementC_FP16>::value
>;

using Gemm_FP16 = cutlass::gemm::device::Gemm<
    ElementA_FP16,
    LayoutA,
    ElementB_FP16,
    LayoutB,
    ElementC_FP16,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp_FP16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2  // Pipeline stages (double buffering)
>;

// Cache for FP16 GEMM operators
using GemmKey = std::tuple<int, int, int, int, int, int>;
static std::mutex g_gemm_fp16_cache_mutex;
static std::map<GemmKey, std::shared_ptr<Gemm_FP16>> g_gemm_fp16_cache;

static std::shared_ptr<Gemm_FP16> get_or_create_gemm_fp16(int M, int N, int K, int lda, int ldb, int ldc)
{
    GemmKey key = std::make_tuple(M, N, K, lda, ldb, ldc);
    std::lock_guard<std::mutex> lock(g_gemm_fp16_cache_mutex);
    auto it = g_gemm_fp16_cache.find(key);
    if (it != g_gemm_fp16_cache.end()) {
        return it->second;
    }
    auto gemm_ptr = std::make_shared<Gemm_FP16>();
    g_gemm_fp16_cache[key] = gemm_ptr;
    return gemm_ptr;
}

cudaError_t cutlass_gemm_fp16_launch(
    int M, int N, int K,
    const ElementA_FP16 *d_A, int lda,
    const ElementB_FP16 *d_B, int ldb,
    ElementC_FP16 *d_C, int ldc,
    float alpha,
    float beta,
    cudaStream_t stream = 0)
{
    if (M == 0 || N == 0 || K == 0) return cudaSuccess;

    auto gemm_ptr = get_or_create_gemm_fp16(M, N, K, lda, ldb, ldc);
    Gemm_FP16 &gemm_op = *gemm_ptr;

    typename Gemm_FP16::Arguments args(
        {M, N, K},
        {d_A, lda},
        {d_B, ldb},
        {d_C, ldc},
        {d_C, ldc},
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;

    status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// BF16 GEMM Configuration
// -----------------------------------------------------------------------------
using ElementA_BF16 = cutlass::bfloat16_t;
using ElementB_BF16 = cutlass::bfloat16_t;
using ElementC_BF16 = float;  // Output in FP32

using EpilogueOp_BF16 = cutlass::epilogue::thread::LinearCombination<
    ElementC_BF16,
    128 / cutlass::sizeof_bits<ElementC_BF16>::value,
    ElementAccumulator,
    ElementCompute
>;

using Gemm_BF16 = cutlass::gemm::device::Gemm<
    ElementA_BF16,
    LayoutA,
    ElementB_BF16,
    LayoutB,
    ElementC_BF16,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp_BF16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2  // Pipeline stages (double buffering)
>;

// Cache for BF16 GEMM operators
static std::mutex g_gemm_bf16_cache_mutex;
static std::map<GemmKey, std::shared_ptr<Gemm_BF16>> g_gemm_bf16_cache;

static std::shared_ptr<Gemm_BF16> get_or_create_gemm_bf16(int M, int N, int K, int lda, int ldb, int ldc)
{
    GemmKey key = std::make_tuple(M, N, K, lda, ldb, ldc);
    std::lock_guard<std::mutex> lock(g_gemm_bf16_cache_mutex);
    auto it = g_gemm_bf16_cache.find(key);
    if (it != g_gemm_bf16_cache.end()) {
        return it->second;
    }
    auto gemm_ptr = std::make_shared<Gemm_BF16>();
    g_gemm_bf16_cache[key] = gemm_ptr;
    return gemm_ptr;
}

cudaError_t cutlass_gemm_bf16_launch(
    int M, int N, int K,
    const ElementA_BF16 *d_A, int lda,
    const ElementB_BF16 *d_B, int ldb,
    ElementC_BF16 *d_C, int ldc,
    float alpha,
    float beta,
    cudaStream_t stream = 0)
{
    if (M == 0 || N == 0 || K == 0) return cudaSuccess;

    auto gemm_ptr = get_or_create_gemm_bf16(M, N, K, lda, ldb, ldc);
    Gemm_BF16 &gemm_op = *gemm_ptr;

    typename Gemm_BF16::Arguments args(
        {M, N, K},
        {d_A, lda},
        {d_B, ldb},
        {d_C, ldc},
        {d_C, ldc},
        {alpha, beta}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;

    status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

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

    // Make tensors contiguous
    auto A = matrix_a.contiguous();
    auto B = matrix_b.contiguous();
    auto C = output_matrix.contiguous();

    // Extract dimensions
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));

    TORCH_CHECK(B.size(0) == K, "Matrix dimension mismatch: A is MxK, B is KxN, but B has wrong K");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Output matrix has wrong shape");

    // Get device pointers
    const ElementA_FP16 *d_A = reinterpret_cast<const ElementA_FP16 *>(A.data_ptr<at::Half>());
    const ElementB_FP16 *d_B = reinterpret_cast<const ElementB_FP16 *>(B.data_ptr<at::Half>());
    ElementC_FP16 *d_C = C.data_ptr<float>();

    int lda = K;
    int ldb = N;
    int ldc = N;

    // Get current CUDA stream (use default stream 0 for simplicity)
    cudaStream_t stream = 0;

    // Launch CUTLASS GEMM
    cudaError_t err = cutlass_gemm_fp16_launch(
        M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);

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

    // Make tensors contiguous
    auto A = matrix_a.contiguous();
    auto B = matrix_b.contiguous();
    auto C = output_matrix.contiguous();

    // Extract dimensions
    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));

    TORCH_CHECK(B.size(0) == K, "Matrix dimension mismatch: A is MxK, B is KxN, but B has wrong K");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Output matrix has wrong shape");

    // Get device pointers
    const ElementA_BF16 *d_A = reinterpret_cast<const ElementA_BF16 *>(A.data_ptr<at::BFloat16>());
    const ElementB_BF16 *d_B = reinterpret_cast<const ElementB_BF16 *>(B.data_ptr<at::BFloat16>());
    ElementC_BF16 *d_C = C.data_ptr<float>();

    int lda = K;
    int ldb = N;
    int ldc = N;

    // Get current CUDA stream (use default stream 0 for simplicity)
    cudaStream_t stream = 0;

    // Launch CUTLASS GEMM
    cudaError_t err = cutlass_gemm_bf16_launch(
        M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS GEMM (BF16) failed with error: ", cudaGetErrorString(err));
}
