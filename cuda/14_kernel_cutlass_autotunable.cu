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
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Enum for all available configurations with descriptive names
enum class CutlassConfig {
    TB_128x256x64_W_64x64x64_S3 = 0,
    TB_64x256x32_W_32x64x32_S4 = 1,
    TB_128x128x32_W_64x64x32_S4 = 2,
    TB_128x64x32_W_64x32x32_S4 = 3,
    TB_64x128x32_W_32x64x32_S4 = 4,
    TB_128x32x32_W_64x32x32_S4 = 5,
    TB_64x32x32_W_32x32x32_S5 = 6,
    TB_32x64x32_W_32x32x32_S5 = 7,
    TB_128x128x64_W_64x64x64_S4 = 8,
    TB_128x64x64_W_64x32x64_S4 = 9,
    TB_64x128x64_W_32x64x64_S4 = 10,
    TB_256x256x32_W_64x64x32_S3 = 11,
    TB_256x128x32_W_64x64x32_S3 = 12,
    TB_128x256x32_W_64x64x32_S3 = 13,
    TB_64x64x32_W_32x32x32_S5 = 14,
};

// Configuration struct to define all tunable parameters
template <int ThreadBlockM, int ThreadBlockN, int ThreadBlockK,
          int WarpM, int WarpN, int WarpK,
          int InstrM, int InstrN, int InstrK,
          int Stages, typename InputElementType>
struct CutlassGemmAutotuneConfig
{
    using ElementInput = InputElementType;

    using ThreadBlockShape = cutlass::gemm::GemmShape<ThreadBlockM, ThreadBlockN, ThreadBlockK>;
    using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<InstrM, InstrN, InstrK>;

    static constexpr int kStages = Stages;

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
        cutlass::arch::Sm80,  // SM80 (compatible with SM89/Ada Lovelace)
        ThreadBlockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages>;
};

// Generic launcher template
template <typename Config>
cudaError_t cutlass_gemm_autotune_launch(
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

// Define all configurations based on Triton-style configs
// Config 0: 128x256x64, Warp 64x64x64, Instr 16x8x16, Stages 3
using Config0_FP16 = CutlassGemmAutotuneConfig<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, cutlass::half_t>;
using Config0_BF16 = CutlassGemmAutotuneConfig<128, 256, 64, 64, 64, 64, 16, 8, 16, 3, cutlass::bfloat16_t>;

// Config 1: 64x256x32, Warp 32x64x32, Instr 16x8x16, Stages 4
using Config1_FP16 = CutlassGemmAutotuneConfig<64, 256, 32, 32, 64, 32, 16, 8, 16, 4, cutlass::half_t>;
using Config1_BF16 = CutlassGemmAutotuneConfig<64, 256, 32, 32, 64, 32, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 2: 128x128x32, Warp 64x64x32, Instr 16x8x16, Stages 4
using Config2_FP16 = CutlassGemmAutotuneConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, cutlass::half_t>;
using Config2_BF16 = CutlassGemmAutotuneConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 3: 128x64x32, Warp 64x32x32, Instr 16x8x16, Stages 4
using Config3_FP16 = CutlassGemmAutotuneConfig<128, 64, 32, 64, 32, 32, 16, 8, 16, 4, cutlass::half_t>;
using Config3_BF16 = CutlassGemmAutotuneConfig<128, 64, 32, 64, 32, 32, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 4: 64x128x32, Warp 32x64x32, Instr 16x8x16, Stages 4
using Config4_FP16 = CutlassGemmAutotuneConfig<64, 128, 32, 32, 64, 32, 16, 8, 16, 4, cutlass::half_t>;
using Config4_BF16 = CutlassGemmAutotuneConfig<64, 128, 32, 32, 64, 32, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 5: 128x32x32, Warp 64x32x32, Instr 16x8x16, Stages 4
using Config5_FP16 = CutlassGemmAutotuneConfig<128, 32, 32, 64, 32, 32, 16, 8, 16, 4, cutlass::half_t>;
using Config5_BF16 = CutlassGemmAutotuneConfig<128, 32, 32, 64, 32, 32, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 6: 64x32x32, Warp 32x32x32, Instr 16x8x16, Stages 5
using Config6_FP16 = CutlassGemmAutotuneConfig<64, 32, 32, 32, 32, 32, 16, 8, 16, 5, cutlass::half_t>;
using Config6_BF16 = CutlassGemmAutotuneConfig<64, 32, 32, 32, 32, 32, 16, 8, 16, 5, cutlass::bfloat16_t>;

// Config 7: 32x64x32, Warp 32x32x32, Instr 16x8x16, Stages 5
using Config7_FP16 = CutlassGemmAutotuneConfig<32, 64, 32, 32, 32, 32, 16, 8, 16, 5, cutlass::half_t>;
using Config7_BF16 = CutlassGemmAutotuneConfig<32, 64, 32, 32, 32, 32, 16, 8, 16, 5, cutlass::bfloat16_t>;

// Config 8: 128x128x64, Warp 64x64x64, Instr 16x8x16, Stages 4
using Config8_FP16 = CutlassGemmAutotuneConfig<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, cutlass::half_t>;
using Config8_BF16 = CutlassGemmAutotuneConfig<128, 128, 64, 64, 64, 64, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 9: 128x64x64, Warp 64, 32x64, Instr 16x8x16, Stages 4
using Config9_FP16 = CutlassGemmAutotuneConfig<128, 64, 64, 64, 32, 64, 16, 8, 16, 4, cutlass::half_t>;
using Config9_BF16 = CutlassGemmAutotuneConfig<128, 64, 64, 64, 32, 64, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 10: 64x128x64, Warp 32x64x64, Instr 16x8x16, Stages 4
using Config10_FP16 = CutlassGemmAutotuneConfig<64, 128, 64, 32, 64, 64, 16, 8, 16, 4, cutlass::half_t>;
using Config10_BF16 = CutlassGemmAutotuneConfig<64, 128, 64, 32, 64, 64, 16, 8, 16, 4, cutlass::bfloat16_t>;

// Config 11: 256x256x32, Warp 64x64x32, Instr 16x8x16, Stages 3 (large blocks)
using Config11_FP16 = CutlassGemmAutotuneConfig<256, 256, 32, 64, 64, 32, 16, 8, 16, 3, cutlass::half_t>;
using Config11_BF16 = CutlassGemmAutotuneConfig<256, 256, 32, 64, 64, 32, 16, 8, 16, 3, cutlass::bfloat16_t>;

// Config 12: 256x128x32, Warp 64x64x32, Instr 16x8x16, Stages 3
using Config12_FP16 = CutlassGemmAutotuneConfig<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, cutlass::half_t>;
using Config12_BF16 = CutlassGemmAutotuneConfig<256, 128, 32, 64, 64, 32, 16, 8, 16, 3, cutlass::bfloat16_t>;

// Config 13: 128x256x32, Warp 64x64x32, Instr 16x8x16, Stages 3
using Config13_FP16 = CutlassGemmAutotuneConfig<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, cutlass::half_t>;
using Config13_BF16 = CutlassGemmAutotuneConfig<128, 256, 32, 64, 64, 32, 16, 8, 16, 3, cutlass::bfloat16_t>;

// Config 14: 64x64x32, Warp 32x32x32, Instr 16x8x16, Stages 5 (small, high stages)
using Config14_FP16 = CutlassGemmAutotuneConfig<64, 64, 32, 32, 32, 32, 16, 8, 16, 5, cutlass::half_t>;
using Config14_BF16 = CutlassGemmAutotuneConfig<64, 64, 32, 32, 32, 32, 16, 8, 16, 5, cutlass::bfloat16_t>;

// Dispatcher function that calls the appropriate config based on config enum
template <typename TorchType, typename CutlassType>
cudaError_t dispatch_cutlass_autotune(
    CutlassConfig config,
    int M, int N, int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    ElementOutput *d_C, int ldc,
    float alpha, float beta,
    cudaStream_t stream = nullptr)
{
    switch (config)
    {
    case CutlassConfig::TB_128x256x64_W_64x64x64_S3:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config0_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config0_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_64x256x32_W_32x64x32_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config1_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config1_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_128x128x32_W_64x64x32_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config2_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config2_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_128x64x32_W_64x32x32_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config3_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config3_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_64x128x32_W_32x64x32_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config4_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config4_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_128x32x32_W_64x32x32_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config5_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config5_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_64x32x32_W_32x32x32_S5:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config6_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config6_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_32x64x32_W_32x32x32_S5:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config7_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config7_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_128x128x64_W_64x64x64_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config8_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config8_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_128x64x64_W_64x32x64_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config9_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config9_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_64x128x64_W_32x64x64_S4:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config10_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config10_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_256x256x32_W_64x64x32_S3:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config11_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config11_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_256x128x32_W_64x64x32_S3:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config12_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config12_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_128x256x32_W_64x64x32_S3:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config13_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config13_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    case CutlassConfig::TB_64x64x32_W_32x32x32_S5:
        if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
            return cutlass_gemm_autotune_launch<Config14_FP16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
        else
            return cutlass_gemm_autotune_launch<Config14_BF16>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    default:
        return cudaErrorInvalidValue;
    }
}

// PyTorch wrapper template
template <typename TorchType, typename CutlassType>
void cutlass_gemm_autotune_pytorch_wrapper(
    int config_id,
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
    const auto *d_A = reinterpret_cast<const CutlassType *>(matrix_a.data_ptr<TorchType>());
    const auto *d_B = reinterpret_cast<const CutlassType *>(matrix_b.data_ptr<TorchType>());
    auto *d_C = output_matrix.data_ptr<float>();

    int lda = K;
    int ldb = N;
    int ldc = N;

    cudaStream_t stream = nullptr;

    // Convert int config_id to enum
    CutlassConfig config = static_cast<CutlassConfig>(config_id);

    // Launch CUTLASS GEMM with specified config
    const cudaError_t err = dispatch_cutlass_autotune<TorchType, CutlassType>(
        config, M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS GEMM Autotune (", dtype_name, ", config ", config_id, ") failed: ", cudaGetErrorString(err));
}

// FP16 launcher
void sgemm_cutlass_autotune_fp16(
    int config_id,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    float alpha,
    float beta)
{
    cutlass_gemm_autotune_pytorch_wrapper<at::Half, cutlass::half_t>(
        config_id, matrix_a, matrix_b, output_matrix, alpha, beta,
        "float16", at::kHalf);
}

// BF16 launcher
void sgemm_cutlass_autotune_bf16(
    int config_id,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    float alpha,
    float beta)
{
    cutlass_gemm_autotune_pytorch_wrapper<at::BFloat16, cutlass::bfloat16_t>(
        config_id, matrix_a, matrix_b, output_matrix, alpha, beta,
        "bfloat16", at::kBFloat16);
}

// Function to get the number of available configs
int get_num_cutlass_configs()
{
    return 15; // We have 15 configurations (0-14)
}
