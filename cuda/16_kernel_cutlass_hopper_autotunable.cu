#include <torch/torch.h>
#include <cuda_runtime.h>
#include "gemm_kernels.cuh"

// CUTLASS 3.x includes for Hopper Collective Builder
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

// Hopper (SM90) Warp-Specialized GEMM using CUTLASS 3.x Collective Builder API
// Autotunable version with configurable tile shapes, cluster shapes, and stages

// Enum for all available Hopper configurations
enum class HopperConfig
{
    T_128x128x64_C_2x1x1 = 0,
    T_128x256x64_C_2x1x1 = 1,
    T_256x128x64_C_1x2x1 = 2,
    T_128x128x128_C_2x1x1 = 3,
    T_256x256x64_C_2x2x1 = 4,
    T_128x64x64_C_2x1x1 = 5,
    T_64x128x64_C_1x2x1 = 6,
    T_64x64x128_C_1x1x1 = 7,
    T_128x128x64_C_1x1x1 = 8,
    // NOTE: Configs 9, 10, 11 removed - too large for StageCountAuto to allocate 2+ stages
    Count // to get the number of configurations
};

template <int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK,
          typename ElementType>
struct CutlassHopperGemmAutotuneConfig
{
    // Element types
    using ElementA = ElementType;
    using ElementB = ElementType;
    using ElementC = float;
    using ElementD = float;
    using ElementAccumulator = float;

    // Layouts
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    // Alignment (16-byte for TMA)
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Tile and cluster configuration for H100
    using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, Int<ClusterK>>;

    // Warp specialization schedules
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

    // Build mainloop collective with automatic stage count calculation
    // Use StageCountAuto instead of StageCountAutoCarveout for better stage selection
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAuto,
        KernelSchedule
    >::CollectiveOp;

    // Build epilogue collective
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueSchedule
    >::CollectiveOp;

    // Assemble the kernel (using non-batched shape)
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config>
cudaError_t cutlass_hopper_gemm_autotune_launch(
    int M, int N, int K,
    const typename Config::ElementA *d_A, int lda,
    const typename Config::ElementB *d_B, int ldb,
    typename Config::ElementD *d_D, int ldd,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    typename Config::Gemm gemm_op;

    // Problem size (non-batched GEMM)
    auto problem_shape = make_shape(M, N, K);

    // Stride types for row-major layouts
    using StrideA = typename Config::GemmKernel::StrideA;
    using StrideB = typename Config::GemmKernel::StrideB;
    using StrideC = typename Config::GemmKernel::StrideC;
    using StrideD = typename Config::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    // Hard-coded alpha = 1.0, beta = 0.0
    float alpha = 1.0f;
    float beta = 0.0f;

    typename Config::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {d_A, stride_A, d_B, stride_B},          // Mainloop args
        {{alpha, beta}, d_D, stride_C, d_D, stride_D},  // Epilogue args
        hw_info
    };

    // Check if the problem size is supported
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorNotSupported;
    }

    // Initialize the kernel
    size_t workspace_size = Config::Gemm::get_workspace_size(args);
    void* workspace = nullptr;

    if (workspace_size > 0)
    {
        cudaError_t result = cudaMalloc(&workspace, workspace_size);
        if (result != cudaSuccess)
            return result;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        if (workspace)
            cudaFree(workspace);
        return cudaErrorUnknown;
    }

    // Run the kernel
    status = gemm_op.run(stream);

    // Free workspace
    if (workspace)
        cudaFree(workspace);

    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    return cudaSuccess;
}

template <int TM, int TN, int TK,
          int CM, int CN, int CK,
          typename T>
using HopperGemmCfg = CutlassHopperGemmAutotuneConfig<TM, TN, TK, CM, CN, CK, T>;

struct HopperConfigEntry
{
    int TM, TN, TK;     // Tile shape
    int CM, CN, CK;     // Cluster shape
};

constexpr HopperConfigEntry kHopperConfigs[] = {
    {128, 128, 64,  2, 1, 1},   // 0: Balanced tile with horizontal cluster
    {128, 256, 64,  2, 1, 1},   // 1: Wide tile with horizontal cluster
    {256, 128, 64,  1, 2, 1},   // 2: Tall tile with vertical cluster
    {128, 128, 128, 2, 1, 1},   // 3: Deeper K with horizontal cluster
    {256, 256, 64,  2, 2, 1},   // 4: Large tile with 2D cluster
    {128, 64,  64,  2, 1, 1},   // 5: Narrow tile with horizontal cluster
    {64,  128, 64,  1, 2, 1},   // 6: Narrow tall tile with vertical cluster
    {64,  64,  128, 1, 1, 1},   // 7: Small tile, deep K, no cluster
    {128, 128, 64,  1, 1, 1},   // 8: Balanced tile, no cluster
    // NOTE: Configs with very large tiles + deep K (256x256x128, etc.) removed
    // They require too much shared memory for StageCountAuto to allocate 2+ stages
};

template <int IDX, typename T>
struct GetHopperConfig
{
    static constexpr auto cfg = kHopperConfigs[IDX];
    using type = HopperGemmCfg<
        cfg.TM, cfg.TN, cfg.TK,
        cfg.CM, cfg.CN, cfg.CK,
        T>;
};

template <typename CutlassType, int IDX>
cudaError_t dispatch_hopper_config(
    int M, int N, int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    float *d_D, int ldd,
    cudaStream_t stream)
{
    using FP16Cfg = typename GetHopperConfig<IDX, cutlass::half_t>::type;
    using BF16Cfg = typename GetHopperConfig<IDX, cutlass::bfloat16_t>::type;

    if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
        return cutlass_hopper_gemm_autotune_launch<FP16Cfg>(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);
    else
        return cutlass_hopper_gemm_autotune_launch<BF16Cfg>(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);
}

template <typename TorchType, typename CutlassType>
cudaError_t dispatch_cutlass_hopper_autotune(
    HopperConfig config,
    const int M, const int N, const int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    float *d_D, int ldd,
    cudaStream_t stream = nullptr)
{
    auto launch = [&](auto I)
    {
        return dispatch_hopper_config<CutlassType, I>(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);
    };

    switch (static_cast<int>(config))
    {
#define CASE_CONFIG(I) \
    case I:            \
        return launch(std::integral_constant<int, I>{});
        CASE_CONFIG(0)
        CASE_CONFIG(1)
        CASE_CONFIG(2)
        CASE_CONFIG(3)
        CASE_CONFIG(4)
        CASE_CONFIG(5)
        CASE_CONFIG(6)
        CASE_CONFIG(7)
        CASE_CONFIG(8)
    default:
        return cudaErrorInvalidValue;
#undef CASE_CONFIG
    }
}

// PyTorch wrapper template
template <typename TorchType, typename CutlassType>
void cutlass_hopper_gemm_autotune_pytorch_wrapper(
    int config_id,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    std::string&& dtype_name,
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
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous(),
                "Input tensors must be contiguous for alignment requirements");
    TORCH_CHECK(output_matrix.is_contiguous(), "Output tensor must be contiguous");

    // Extract dimensions
    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == K, "Matrix dimension mismatch");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N, "Output matrix has wrong shape");

    // Check alignment requirements (16-byte alignment for TMA)
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_a.data_ptr()) % 16 == 0,
                "Matrix A must be 16-byte aligned for Hopper TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_b.data_ptr()) % 16 == 0,
                "Matrix B must be 16-byte aligned for Hopper TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output_matrix.data_ptr()) % 16 == 0,
                "Output matrix must be 16-byte aligned for Hopper TMA");

    // Get device pointers
    const auto *d_A = reinterpret_cast<const CutlassType *>(matrix_a.data_ptr<TorchType>());
    const auto *d_B = reinterpret_cast<const CutlassType *>(matrix_b.data_ptr<TorchType>());
    auto *d_D = output_matrix.data_ptr<float>();

    int lda = K;
    int ldb = N;
    int ldd = N;

    cudaStream_t stream = nullptr;

    // Convert int config_id to enum
    auto config = static_cast<HopperConfig>(config_id);

    // Launch CUTLASS Hopper GEMM with specified config (alpha=1.0, beta=0.0 hard-coded)
    const cudaError_t err = dispatch_cutlass_hopper_autotune<TorchType, CutlassType>(
        config, M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS Hopper GEMM Autotune (", dtype_name, ", config ", config_id, ") failed: ", cudaGetErrorString(err));
}

// FP16 launcher
void sgemm_cutlass_hopper_autotune_fp16(
    const int config_id,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_hopper_gemm_autotune_pytorch_wrapper<at::Half, cutlass::half_t>(
        config_id, matrix_a, matrix_b, output_matrix,
        "float16", at::kHalf);
}

// BF16 launcher
void sgemm_cutlass_hopper_autotune_bf16(
    const int config_id,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_hopper_gemm_autotune_pytorch_wrapper<at::BFloat16, cutlass::bfloat16_t>(
        config_id, matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}

// Function to get the number of available Hopper configs
int get_num_cutlass_hopper_configs()
{
    return static_cast<int>(HopperConfig::Count);
}
