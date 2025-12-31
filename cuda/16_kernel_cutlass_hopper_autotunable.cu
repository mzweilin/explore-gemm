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
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cute/tensor.hpp"

using namespace cute;

// Hopper (SM90) Warp-Specialized GEMM using CUTLASS 3.x Collective Builder API
// Autotunable version with runtime-configurable tile shapes, cluster shapes, stages, and schedulers
// This version doesn't pre-compile all configurations; instead, configurations are passed at runtime.

// Kernel schedule types
enum class HopperSchedulerType
{
    TmaWarpSpecialized,           // Basic TMA with warp specialization (no tile scheduler)
    TmaWarpSpecializedPersistent, // TMA with persistent scheduling
    TmaWarpSpecializedPingpong,   // TMA with ping-pong cooperative scheduling
    TmaWarpSpecializedStreamK     // TMA with Stream K scheduling
};

// Stage count type: Auto or explicit value
enum class StageCountMode
{
    Auto,     // Automatic stage count calculation
    Explicit  // Use provided stage count value
};

// Hopper configuration structure (passed at runtime)
struct HopperGemmConfig
{
    int tile_m, tile_n, tile_k;          // Tile shape
    int cluster_m, cluster_n, cluster_k; // Cluster shape
    StageCountMode stage_mode;           // Auto or explicit
    int num_stages;                      // Stage count (if explicit mode)
    HopperSchedulerType scheduler;       // Kernel scheduler type

    // Scheduler-specific parameters (for StreamK and Persistent schedulers)
    int raster_order;      // RasterOrderOptions: 0=AlongM, 1=AlongN, 2=Heuristic
    int decomposition;     // DecompositionMode: 0=Heuristic, 1=DataParallel, 2=SplitK, 3=StreamK
    int swizzle;           // Swizzle log (typically 1)
    int splits;            // Number of splits for SplitK (default 1)
};

// Helper to get raster order options and decomposition mode types
using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

// Runtime-configurable GEMM launcher template
// This template will be instantiated for a few common tile/cluster/stage combinations
template <int TileM, int TileN, int TileK,
          int ClusterM, int ClusterN, int ClusterK,
          int Stages,
          HopperSchedulerType SchedulerType,
          typename ElementType>
struct CutlassHopperGemmKernel
{
    using ElementA = ElementType;
    using ElementB = ElementType;
    using ElementC = ElementType;
    using ElementD = ElementType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    using ClusterShape = Shape<Int<ClusterM>, Int<ClusterN>, Int<ClusterK>>;

    // Select kernel schedule based on scheduler type and tile size
    using KernelSchedule = typename std::conditional<
        TileM < 128 || SchedulerType == HopperSchedulerType::TmaWarpSpecialized,
        cutlass::gemm::KernelTmaWarpSpecialized,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::type;

    // Select epilogue schedule based on scheduler type
    using EpilogueSchedule = typename std::conditional<
        SchedulerType == HopperSchedulerType::TmaWarpSpecializedPersistent ||
        SchedulerType == HopperSchedulerType::TmaWarpSpecializedStreamK,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        cutlass::epilogue::TmaWarpSpecialized>::type;

    // Select stage count type
    using StageCountType = typename std::conditional<
        Stages == -1,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::StageCount<Stages>>::type;

    // Build mainloop collective
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

    // Select tile scheduler type based on scheduler enum
    using TileSchedulerType = typename std::conditional<
        SchedulerType == HopperSchedulerType::TmaWarpSpecialized,
        void,
        typename std::conditional<
            SchedulerType == HopperSchedulerType::TmaWarpSpecializedStreamK,
            cutlass::gemm::StreamKScheduler,
            cutlass::gemm::PersistentScheduler>::type>::type;

    // Helper to create the appropriate GemmKernel type
    template <typename Scheduler>
    static auto make_gemm_kernel_type()
    {
        if constexpr (std::is_void_v<Scheduler>)
        {
            return cutlass::gemm::kernel::GemmUniversal<
                Shape<int, int, int>,
                CollectiveMainloop,
                CollectiveEpilogue>{};
        }
        else
        {
            return cutlass::gemm::kernel::GemmUniversal<
                Shape<int, int, int>,
                CollectiveMainloop,
                CollectiveEpilogue,
                Scheduler>{};
        }
    }

    using GemmKernel = decltype(make_gemm_kernel_type<TileSchedulerType>());
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Helper to check if scheduler is Stream-K
template <typename Scheduler>
struct is_streamk_scheduler : std::false_type {};

template <>
struct is_streamk_scheduler<cutlass::gemm::StreamKScheduler> : std::true_type {};

// Template launch function with scheduler-specific argument handling
template <typename Config>
cudaError_t cutlass_hopper_gemm_autotune_launch(
    int M, int N, int K,
    const typename Config::ElementA *d_A, int lda,
    const typename Config::ElementB *d_B, int ldb,
    typename Config::ElementD *d_D, int ldd,
    const HopperGemmConfig& config,
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

    // Convert config values to CUTLASS types
    RasterOrderOptions raster = static_cast<RasterOrderOptions>(config.raster_order);
    DecompositionMode decomp = static_cast<DecompositionMode>(config.decomposition);

    // Create arguments - different for Stream-K vs other schedulers
    typename Config::Gemm::Arguments args = [&]() {
        if constexpr (is_streamk_scheduler<typename Config::TileSchedulerType>::value)
        {
            // Stream-K scheduler requires additional arguments
            typename Config::GemmKernel::TileScheduler::Arguments scheduler_args{
                config.splits,
                config.swizzle,
                raster,
                decomp
            };

            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},
                {{alpha, beta}, d_D, stride_C, d_D, stride_D},
                hw_info,
                scheduler_args
            };
        }
        else if constexpr (std::is_void_v<typename Config::TileSchedulerType>)
        {
            // No tile scheduler (basic TMA Warp Specialized)
            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},
                {{alpha, beta}, d_D, stride_C, d_D, stride_D},
                hw_info
            };
        }
        else
        {
            // Persistent scheduler (also supports raster order)
            typename Config::GemmKernel::TileScheduler::Arguments scheduler_args{
                config.swizzle,
                raster
            };

            return typename Config::Gemm::Arguments{
                cutlass::gemm::GemmUniversalMode::kGemm,
                problem_shape,
                {d_A, stride_A, d_B, stride_B},
                {{alpha, beta}, d_D, stride_C, d_D, stride_D},
                hw_info,
                scheduler_args
            };
        }
    }();

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

// Simplified runtime dispatch - only instantiates base configurations with Auto stage count
// This minimizes compilation time while still allowing runtime config exploration
#define DISPATCH_TILE_CONFIG(TM, TN, TK, CM, CN, CK) \
    do { \
        using KernelConfig = CutlassHopperGemmKernel<TM, TN, TK, CM, CN, CK, -1, HopperSchedulerType::TmaWarpSpecialized, CutlassType>; \
        return cutlass_hopper_gemm_autotune_launch<KernelConfig>(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, config, stream); \
    } while(0)

template <typename CutlassType>
cudaError_t dispatch_cutlass_hopper_runtime(
    const HopperGemmConfig& config,
    int M, int N, int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    CutlassType *d_D, int ldd,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    // Only support TmaWarpSpecialized scheduler
    if (config.scheduler != HopperSchedulerType::TmaWarpSpecialized) {
        return cudaErrorNotSupported;
    }

    // Only support Auto stage count (stage-specific variants removed)
    if (config.stage_mode != StageCountMode::Auto) {
        return cudaErrorNotSupported;
    }

    // Dispatch based on tile and cluster configuration
    // We instantiate only one kernel per tile/cluster combination (with Auto stages)
    // This gives us 9 kernel instantiations instead of 36

    if (config.tile_m == 128 && config.tile_n == 128 && config.tile_k == 64 &&
        config.cluster_m == 2 && config.cluster_n == 1 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(128, 128, 64, 2, 1, 1);
    }
    else if (config.tile_m == 128 && config.tile_n == 256 && config.tile_k == 64 &&
             config.cluster_m == 2 && config.cluster_n == 1 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(128, 256, 64, 2, 1, 1);
    }
    else if (config.tile_m == 256 && config.tile_n == 128 && config.tile_k == 64 &&
             config.cluster_m == 1 && config.cluster_n == 2 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(256, 128, 64, 1, 2, 1);
    }
    else if (config.tile_m == 128 && config.tile_n == 128 && config.tile_k == 128 &&
             config.cluster_m == 2 && config.cluster_n == 1 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(128, 128, 128, 2, 1, 1);
    }
    else if (config.tile_m == 256 && config.tile_n == 256 && config.tile_k == 64 &&
             config.cluster_m == 2 && config.cluster_n == 2 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(256, 256, 64, 2, 2, 1);
    }
    else if (config.tile_m == 128 && config.tile_n == 64 && config.tile_k == 64 &&
             config.cluster_m == 2 && config.cluster_n == 1 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(128, 64, 64, 2, 1, 1);
    }
    else if (config.tile_m == 64 && config.tile_n == 128 && config.tile_k == 64 &&
             config.cluster_m == 1 && config.cluster_n == 2 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(64, 128, 64, 1, 2, 1);
    }
    else if (config.tile_m == 64 && config.tile_n == 64 && config.tile_k == 128 &&
             config.cluster_m == 1 && config.cluster_n == 1 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(64, 64, 128, 1, 1, 1);
    }
    else if (config.tile_m == 128 && config.tile_n == 128 && config.tile_k == 64 &&
             config.cluster_m == 1 && config.cluster_n == 1 && config.cluster_k == 1) {
        DISPATCH_TILE_CONFIG(128, 128, 64, 1, 1, 1);
    }

    return cudaErrorNotSupported;
}

#undef DISPATCH_TILE_CONFIG

// PyTorch wrapper template - now accepts runtime config parameters
template <typename TorchType, typename CutlassType>
void cutlass_hopper_gemm_autotune_pytorch_wrapper(
    int tile_m, int tile_n, int tile_k,
    int cluster_m, int cluster_n, int cluster_k,
    int stages, // -1 for Auto, or explicit count (3-5)
    int scheduler_type, // 0 = TmaWarpSpecialized, 1 = Persistent, 2 = Pingpong, 3 = StreamK
    int raster_order, // 0 = AlongM, 1 = AlongN, 2 = Heuristic
    int decomposition, // 0 = Heuristic, 1 = DataParallel, 2 = SplitK, 3 = StreamK
    int swizzle, // Swizzle log (typically 1)
    int splits, // Number of splits for SplitK
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

    // Build config
    HopperGemmConfig config;
    config.tile_m = tile_m;
    config.tile_n = tile_n;
    config.tile_k = tile_k;
    config.cluster_m = cluster_m;
    config.cluster_n = cluster_n;
    config.cluster_k = cluster_k;
    config.stage_mode = (stages == -1) ? StageCountMode::Auto : StageCountMode::Explicit;
    config.num_stages = stages;
    config.scheduler = static_cast<HopperSchedulerType>(scheduler_type);
    config.raster_order = raster_order;
    config.decomposition = decomposition;
    config.swizzle = swizzle;
    config.splits = splits;

    // Launch CUTLASS Hopper GEMM with specified config (alpha=1.0, beta=0.0 hard-coded)
    const cudaError_t err = dispatch_cutlass_hopper_runtime<CutlassType>(
        config, M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS Hopper GEMM Autotune (", dtype_name,
                ", T", tile_m, "x", tile_n, "x", tile_k,
                ", C", cluster_m, "x", cluster_n, "x", cluster_k,
                ", stages=", stages, ") failed: ", cudaGetErrorString(err));
}

// BF16 launcher - now accepts config parameters
void sgemm_cutlass_hopper_autotune_bf16(
    int tile_m, int tile_n, int tile_k,
    int cluster_m, int cluster_n, int cluster_k,
    int stages,
    int scheduler_type,
    int raster_order,
    int decomposition,
    int swizzle,
    int splits,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    cutlass_hopper_gemm_autotune_pytorch_wrapper<at::BFloat16, cutlass::bfloat16_t>(
        tile_m, tile_n, tile_k,
        cluster_m, cluster_n, cluster_k,
        stages, scheduler_type,
        raster_order, decomposition, swizzle, splits,
        matrix_a, matrix_b, output_matrix,
        "bfloat16", at::kBFloat16);
}
