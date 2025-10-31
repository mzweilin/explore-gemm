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

using ElementAccumulator = float;
using ElementCompute = float;
using ElementOutput = float; // Always output FP32

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Tile shapes
using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template <typename InputElementType>
struct CutlassGemmConfig
{
    using ElementInput = InputElementType;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,  // SM80 for Ampere/Ada architecture
        ThreadBlockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp>;
};

using FP16Config = CutlassGemmConfig<cutlass::half_t>;
using BF16Config = CutlassGemmConfig<cutlass::bfloat16_t>;

using ThreadBlockShapeFP32 = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShapeFP32 = cutlass::gemm::GemmShape<64, 64, 8>;
using InstructionShapeFP32 = cutlass::gemm::GemmShape<>;

struct CutlassGemmConfigFP32
{
    using ElementInput = float;

    // SIMT epilogue must operate on scalars (vector length = 1)
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        1>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,  // SIMT instead of TensorOp
        cutlass::arch::Sm80,          // SM80 for Ampere/Ada architecture
        ThreadBlockShapeFP32,
        WarpShapeFP32>;
};

using FP32Config = CutlassGemmConfigFP32;

template <typename Config>
cudaError_t cutlass_gemm_launch(
    int M, int N, int K,
    const typename Config::ElementInput *d_A, int lda,
    const typename Config::ElementInput *d_B, int ldb,
    ElementOutput *d_C, int ldc,
    float alpha, float beta,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    typename Config::Gemm gemm_op;

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

template <typename Config, typename TorchType>
void cutlass_gemm_pytorch_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    const float alpha, const float beta,
    const char *dtype_name,
    const at::ScalarType expected_type)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == expected_type, "Matrix A must be ", dtype_name);
    TORCH_CHECK(matrix_b.scalar_type() == expected_type, "Matrix B must be ", dtype_name);
    TORCH_CHECK(output_matrix.scalar_type() == at::kFloat, "Output matrix must be float32");

    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "A and B must be 2D tensors");

    // Extract dimensions
    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == K, "Matrix dimension mismatch");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N, "Output matrix has wrong shape");

    // Get device pointers
    const auto *d_A =
        reinterpret_cast<const typename Config::ElementInput *>(matrix_a.data_ptr<TorchType>());
    const auto *d_B =
        reinterpret_cast<const typename Config::ElementInput *>(matrix_b.data_ptr<TorchType>());
    auto *d_C = output_matrix.data_ptr<float>();

    int lda = K;
    int ldb = N;
    int ldc = N;

    cudaStream_t stream = nullptr;

    // Launch CUTLASS GEMM
    const cudaError_t err = cutlass_gemm_launch<Config>(
        M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS GEMM (", dtype_name, ") failed: ", cudaGetErrorString(err));
}

// FP16 launcher
void sgemm_cutlass_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    cutlass_gemm_pytorch_wrapper<FP16Config, at::Half>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        "float16", at::kHalf);
}

// BF16 launcher
void sgemm_cutlass_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    cutlass_gemm_pytorch_wrapper<BF16Config, at::BFloat16>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        "bfloat16", at::kBFloat16);
}

// FP32 launcher
void sgemm_cutlass_fp32(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                        torch::Tensor &output_matrix, float alpha, float beta)
{
    cutlass_gemm_pytorch_wrapper<FP32Config, float>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        "float32", at::kFloat);
}
