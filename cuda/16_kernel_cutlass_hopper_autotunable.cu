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
// Fixed configuration: 128x128x64 tile, 2x1x1 cluster, bfloat16, StreamK scheduler
// Only scheduler parameters (raster_order, decomposition, swizzle, splits) are runtime-configurable

// Simplified configuration structure - only runtime scheduler parameters
struct HopperGemmConfig
{
    // StreamK scheduler parameters (runtime-configurable)
    int raster_order;   // RasterOrderOptions: 0=AlongM, 1=AlongN, 2=Heuristic
    int decomposition;  // DecompositionMode: 0=Heuristic, 1=DataParallel, 2=SplitK, 3=StreamK
    int swizzle;        // Swizzle log (typically 1)
    int splits;         // Number of splits for SplitK (default 1)
};

// Helper to get raster order options and decomposition mode types
using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

// Simplified GEMM kernel with fixed configuration matching benchmark_hopper.cu
// Fixed tile shape: 128x128x64, cluster shape: 2x1x1, bfloat16 element type
// Uses TmaWarpSpecializedCooperative for mainloop/epilogue and StreamKScheduler
struct CutlassHopperGemmKernel
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_2, _1, _1>;

    // Fixed kernel schedules
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using TileSchedulerType = cutlass::gemm::StreamKScheduler;

    // Auto stage count
    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            TileShape,
            ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator,
            ElementAccumulator,
            ElementC, LayoutC, AlignmentC,
            ElementD, LayoutD, AlignmentD,
            EpilogueSchedule>::CollectiveOp::SharedStorage))>;

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

    // Assemble the kernel with StreamK scheduler
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileSchedulerType>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Launch function for fixed StreamK configuration
cudaError_t cutlass_hopper_gemm_launch(
    int M, int N, int K,
    const cutlass::bfloat16_t *d_A, int lda,
    const cutlass::bfloat16_t *d_B, int ldb,
    cutlass::bfloat16_t *d_D, int ldd,
    const HopperGemmConfig& config,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    CutlassHopperGemmKernel::Gemm gemm_op;

    // Problem size (non-batched GEMM)
    auto problem_shape = make_shape(M, N, K);

    // Stride types (A: RowMajor, B/C/D: ColumnMajor)
    using StrideA = CutlassHopperGemmKernel::GemmKernel::StrideA;
    using StrideB = CutlassHopperGemmKernel::GemmKernel::StrideB;
    using StrideC = CutlassHopperGemmKernel::GemmKernel::StrideC;
    using StrideD = CutlassHopperGemmKernel::GemmKernel::StrideD;

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

    // Stream-K scheduler arguments
    CutlassHopperGemmKernel::GemmKernel::TileScheduler::Arguments scheduler_args{
        config.splits,
        config.swizzle,
        raster,
        decomp
    };

    // Create arguments for StreamK scheduler
    CutlassHopperGemmKernel::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {d_A, stride_A, d_B, stride_B},
        {{alpha, beta}, d_D, stride_C, d_D, stride_D},
        hw_info,
        scheduler_args
    };

    // Check if the problem size is supported
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorNotSupported;
    }

    // Initialize the kernel
    size_t workspace_size = CutlassHopperGemmKernel::Gemm::get_workspace_size(args);
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

// Simplified runtime dispatch - single kernel instantiation
// Tile shape: 128x128x64, cluster: 2x1x1, element type: bfloat16
cudaError_t dispatch_cutlass_hopper_runtime(
    const HopperGemmConfig& config,
    int M, int N, int K,
    const cutlass::bfloat16_t *d_A, int lda,
    const cutlass::bfloat16_t *d_B, int ldb,
    cutlass::bfloat16_t *d_D, int ldd,
    cudaStream_t stream = nullptr)
{
    return cutlass_hopper_gemm_launch(M, N, K, d_A, lda, d_B, ldb, d_D, ldd, config, stream);
}

// PyTorch wrapper - simplified with fixed tile/cluster configuration
// Only raster_order, decomposition, swizzle, and splits are runtime-configurable
void sgemm_cutlass_hopper_autotune_bf16(
    int raster_order,
    int decomposition,
    int swizzle,
    int splits,
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    // Validate input tensors
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == at::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.scalar_type() == at::kBFloat16, "Matrix B must be bfloat16");
    TORCH_CHECK(output_matrix.scalar_type() == at::kBFloat16, "Output matrix must be bfloat16");

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
    const auto *d_A = reinterpret_cast<const cutlass::bfloat16_t *>(matrix_a.data_ptr<at::BFloat16>());
    const auto *d_B = reinterpret_cast<const cutlass::bfloat16_t *>(matrix_b.data_ptr<at::BFloat16>());
    auto *d_D = reinterpret_cast<cutlass::bfloat16_t *>(output_matrix.data_ptr<at::BFloat16>());

    int lda = K;
    int ldb = N;
    int ldd = N;

    cudaStream_t stream = nullptr;

    // Build config with runtime scheduler parameters
    HopperGemmConfig config{raster_order, decomposition, swizzle, splits};

    // Launch CUTLASS Hopper GEMM with specified config (alpha=1.0, beta=0.0 hard-coded)
    const cudaError_t err = dispatch_cutlass_hopper_runtime(
        config, M, N, K, d_A, lda, d_B, ldb, d_D, ldd, stream);

    TORCH_CHECK(err == cudaSuccess,
                "CUTLASS Hopper GEMM (bfloat16, 128x128x64, 2x1x1) failed: ", cudaGetErrorString(err));
}
