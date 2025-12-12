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

// Stage count variants: -1 = Auto, 3-6 = explicit stage counts
constexpr int STAGE_AUTO = -1;

template <int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK,
          int Stages, // -1 for Auto, or explicit count (3-6)
          typename ElementType>
struct CutlassHopperGemmAutotuneConfig
{
    // Element types
    using ElementA = ElementType;
    using ElementB = ElementType;
    using ElementC = ElementType;
    using ElementD = ElementType;
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
    using KernelSchedule = typename std::conditional<
        (TileM < 128),
        cutlass::gemm::KernelTmaWarpSpecialized,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::type;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedulerType = void;

    // Select stage count policy based on template parameter
    using StageCountType = typename std::conditional<
        Stages == STAGE_AUTO,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::StageCount<Stages>>::type;

    // Build mainloop collective with configurable stage count
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        StageCountType,
        KernelSchedule>::CollectiveOp;

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
        EpilogueSchedule>::CollectiveOp;

    // Assemble the kernel (using non-batched shape)
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileSchedulerType>;

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
        {d_A, stride_A, d_B, stride_B},                // Mainloop args
        {{alpha, beta}, d_D, stride_C, d_D, stride_D}, // Epilogue args
        hw_info};

    // Check if the problem size is supported
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorNotSupported;
    }

    // Initialize the kernel
    size_t workspace_size = Config::Gemm::get_workspace_size(args);
    void *workspace = nullptr;

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
          int Stages,
          typename T>
using HopperGemmCfg = CutlassHopperGemmAutotuneConfig<TM, TN, TK, CM, CN, CK, Stages, T>;

// Base configurations (tile and cluster shapes)
struct HopperBaseConfig
{
    int TM, TN, TK; // Tile shape
    int CM, CN, CK; // Cluster shape
};

constexpr HopperBaseConfig kHopperBaseConfigs[] = {
    {128, 128, 64, 2, 1, 1},  // 0: Balanced tile with horizontal cluster
    {128, 256, 64, 2, 1, 1},  // 1: Wide tile with horizontal cluster
    {256, 128, 64, 1, 2, 1},  // 2: Tall tile with vertical cluster
    {128, 128, 128, 2, 1, 1}, // 3: Deeper K with horizontal cluster
    {256, 256, 64, 2, 2, 1},  // 4: Large tile with 2D cluster
    {128, 64, 64, 2, 1, 1},   // 5: Narrow tile with horizontal cluster
    {64, 128, 64, 1, 2, 1},   // 6: Narrow tall tile with vertical cluster
    {64, 64, 128, 1, 1, 1},   // 7: Small tile, deep K, no cluster
    {128, 128, 64, 1, 1, 1},  // 8: Balanced tile, no cluster
};

constexpr int NUM_BASE_CONFIGS = sizeof(kHopperBaseConfigs) / sizeof(HopperBaseConfig);

// Stage count variants for each base config: Auto, 3, 4, 5
constexpr int kStageVariants[] = {STAGE_AUTO, 3, 4, 5};
constexpr int NUM_STAGE_VARIANTS = std::size(kStageVariants);

// Total number of configurations = base configs × stage variants
constexpr int NUM_HOPPER_CONFIGS = NUM_BASE_CONFIGS * NUM_STAGE_VARIANTS;

// Helper to get base config index and stage variant from flat config ID
constexpr int get_base_config_idx(int config_id) { return config_id / NUM_STAGE_VARIANTS; }
constexpr int get_stage_variant_idx(int config_id) { return config_id % NUM_STAGE_VARIANTS; }

template <int IDX, typename T>
struct GetHopperConfig
{
    static constexpr int base_idx = get_base_config_idx(IDX);
    static constexpr int stage_idx = get_stage_variant_idx(IDX);
    static constexpr auto cfg = kHopperBaseConfigs[base_idx];
    static constexpr int stages = kStageVariants[stage_idx];

    using type = HopperGemmCfg<
        cfg.TM, cfg.TN, cfg.TK,
        cfg.CM, cfg.CN, cfg.CK,
        stages,
        T>;
};

template <typename CutlassType, int IDX>
cudaError_t dispatch_hopper_config(
    int M, int N, int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    CutlassType *d_D, int ldd,
    cudaStream_t stream)
{
    using BF16Cfg = typename GetHopperConfig<IDX, cutlass::bfloat16_t>::type;
    return cutlass_hopper_gemm_autotune_launch<BF16Cfg>(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);
}

template <typename TorchType, typename CutlassType>
cudaError_t dispatch_cutlass_hopper_autotune(
    int config_id,
    const int M, const int N, const int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    CutlassType *d_D, int ldd,
    cudaStream_t stream = nullptr)
{
    auto launch = [&](auto I)
    {
        return dispatch_hopper_config<CutlassType, I>(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);
    };

    switch (config_id)
    {
#define CASE_CONFIG(I) \
    case I:            \
        return launch(std::integral_constant<int, I>{});
        // 9 base configs × 4 stage variants = 36 total configurations
        // Config ID = base_idx * 4 + stage_idx
        // Base 0 (128x128x64_C2x1x1): Auto, S3, S4, S5
        CASE_CONFIG(0)
        CASE_CONFIG(1) CASE_CONFIG(2) CASE_CONFIG(3)
            // Base 1 (128x256x64_C2x1x1): Auto, S3, S4, S5
            CASE_CONFIG(4) CASE_CONFIG(5) CASE_CONFIG(6) CASE_CONFIG(7)
            // Base 2 (256x128x64_C1x2x1): Auto, S3, S4, S5
            CASE_CONFIG(8) CASE_CONFIG(9) CASE_CONFIG(10) CASE_CONFIG(11)
            // Base 3 (128x128x128_C2x1x1): Auto, S3, S4, S5
            CASE_CONFIG(12) CASE_CONFIG(13) CASE_CONFIG(14) CASE_CONFIG(15)
            // Base 4 (256x256x64_C2x2x1): Auto, S3, S4, S5
            CASE_CONFIG(16) CASE_CONFIG(17) CASE_CONFIG(18) CASE_CONFIG(19)
            // Base 5 (128x64x64_C2x1x1): Auto, S3, S4, S5
            CASE_CONFIG(20) CASE_CONFIG(21) CASE_CONFIG(22) CASE_CONFIG(23)
            // Base 6 (64x128x64_C1x2x1): Auto, S3, S4, S5
            CASE_CONFIG(24) CASE_CONFIG(25) CASE_CONFIG(26) CASE_CONFIG(27)
            // Base 7 (64x64x128_C1x1x1): Auto, S3, S4, S5
            CASE_CONFIG(28) CASE_CONFIG(29) CASE_CONFIG(30) CASE_CONFIG(31)
            // Base 8 (128x128x64_C1x1x1): Auto, S3, S4, S5
            CASE_CONFIG(32) CASE_CONFIG(33) CASE_CONFIG(34) CASE_CONFIG(35) default : return cudaErrorInvalidValue;
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
    std::string &&dtype_name,
    const at::ScalarType expected_type)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == expected_type, "Matrix A must be ", dtype_name);
    TORCH_CHECK(matrix_b.scalar_type() == expected_type, "Matrix B must be ", dtype_name);
    TORCH_CHECK(output_matrix.scalar_type() == expected_type, "Output matrix must be ", dtype_name);

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
    auto *d_D = reinterpret_cast<CutlassType *>(output_matrix.data_ptr<TorchType>());

    int lda = K;
    int ldb = N;
    int ldd = N;

    cudaStream_t stream = nullptr;

    // Launch CUTLASS Hopper GEMM with specified config (alpha=1.0, beta=0.0 hard-coded)
    const cudaError_t err = dispatch_cutlass_hopper_autotune<TorchType, CutlassType>(
        config_id, M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS Hopper GEMM Autotune (", dtype_name, ", config ", config_id, ") failed: ", cudaGetErrorString(err));
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
    return NUM_HOPPER_CONFIGS;
}
