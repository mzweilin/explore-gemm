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
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"

#include <mutex>
#include <map>
#include <tuple>
#include <memory>

// -----------------------------------------------------------------------------
// Common configuration
// -----------------------------------------------------------------------------

using ElementAccumulator = float;
using ElementCompute = float;
using ElementOutput = float; // Always output FP32

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Tile shapes
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// -----------------------------------------------------------------------------
// Templated GEMM configuration
// -----------------------------------------------------------------------------

template <typename InputElementType>
struct CutlassGemmConfig
{
    using ElementInput = InputElementType;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        // Cant use sm89: https://github.com/NVIDIA/cutlass/issues/1181
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp>;
};

// Type aliases for specific dtypes
using FP16Config = CutlassGemmConfig<cutlass::half_t>;
using BF16Config = CutlassGemmConfig<cutlass::bfloat16_t>;

// -----------------------------------------------------------------------------
// Templated GEMM operator cache
// -----------------------------------------------------------------------------

template <typename GemmType>
class GemmCache
{
private:
    using Key = std::tuple<int, int, int, int, int, int>;
    std::mutex mutex_;
    std::map<Key, std::shared_ptr<GemmType>> cache_;

public:
    std::shared_ptr<GemmType> get_or_create(int M, int N, int K, int lda, int ldb, int ldc)
    {
        Key key = std::make_tuple(M, N, K, lda, ldb, ldc);
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            return it->second;
        }

        auto gemm_ptr = std::make_shared<GemmType>();
        cache_[key] = gemm_ptr;
        return gemm_ptr;
    }
};

// Global cache instances
static GemmCache<FP16Config::Gemm> g_fp16_cache;
static GemmCache<BF16Config::Gemm> g_bf16_cache;

// -----------------------------------------------------------------------------
// Templated GEMM launcher
// -----------------------------------------------------------------------------

template <typename Config>
cudaError_t cutlass_gemm_launch(
    int M, int N, int K,
    const typename Config::ElementInput *d_A, int lda,
    const typename Config::ElementInput *d_B, int ldb,
    ElementOutput *d_C, int ldc,
    float alpha, float beta,
    GemmCache<typename Config::Gemm> &cache,
    cudaStream_t stream = 0)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    auto gemm_ptr = cache.get_or_create(M, N, K, lda, ldb, ldc);
    typename Config::Gemm &gemm_op = *gemm_ptr;

    typename Config::Gemm::Arguments args(
        {M, N, K},
        {d_A, lda},
        {d_B, ldb},
        {d_C, ldc},
        {d_C, ldc},
        {alpha, beta});

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
// PyTorch wrapper template
// -----------------------------------------------------------------------------

template <typename Config, typename TorchType>
void cutlass_gemm_pytorch_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    float alpha, float beta,
    GemmCache<typename Config::Gemm> &cache,
    const char *dtype_name,
    at::ScalarType expected_type)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == expected_type, "Matrix A must be ", dtype_name);
    TORCH_CHECK(matrix_b.scalar_type() == expected_type, "Matrix B must be ", dtype_name);
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

    TORCH_CHECK(B.size(0) == K, "Matrix dimension mismatch");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "Output matrix has wrong shape");

    // Get device pointers
    const typename Config::ElementInput *d_A =
        reinterpret_cast<const typename Config::ElementInput *>(A.data_ptr<TorchType>());
    const typename Config::ElementInput *d_B =
        reinterpret_cast<const typename Config::ElementInput *>(B.data_ptr<TorchType>());
    ElementOutput *d_C = C.data_ptr<float>();

    int lda = K;
    int ldb = N;
    int ldc = N;

    cudaStream_t stream = 0;

    // Launch CUTLASS GEMM
    cudaError_t err = cutlass_gemm_launch<Config>(
        M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, cache, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS GEMM (", dtype_name, ") failed: ", cudaGetErrorString(err));
}

// -----------------------------------------------------------------------------
// Public API functions
// -----------------------------------------------------------------------------

// FP16 launcher
void sgemm_cutlass_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    cutlass_gemm_pytorch_wrapper<FP16Config, at::Half>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        g_fp16_cache, "float16", at::kHalf);
}

// BF16 launcher
void sgemm_cutlass_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    cutlass_gemm_pytorch_wrapper<BF16Config, at::BFloat16>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        g_bf16_cache, "bfloat16", at::kBFloat16);
}
