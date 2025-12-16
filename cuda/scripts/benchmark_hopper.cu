#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "common.h"

using namespace cute;

using ElementA = half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = half_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = half_t;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;
using StageCountType = cutlass::gemm::collective::StageCountAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StreamKScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using DeviceGemmReference = cutlass::reference::device::Gemm<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
using DecompositionMode = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;

struct Options
{

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  RasterOrderOptions raster;
  DecompositionMode decomp;
  int swizzle;
  int splits;
  bool csv;

  Options() : help(false),
              m(2048), n(2048), k(2048),
              alpha(1.f), beta(0.f),
              iterations(100),
              raster(RasterOrderOptions::Heuristic),
              swizzle(1),
              decomp(DecompositionMode::Heuristic),
              splits(1),
              csv(false)
  {
  }

  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help"))
    {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n')
    {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm')
    {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h')
    {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);

    std::string decomposition;
    cmd.get_cmd_line_argument("decomposition", decomposition);
    if (decomposition == "heuristic")
    {
      decomp = DecompositionMode::Heuristic;
    }
    else if (decomposition == "streamk")
    {
      decomp = DecompositionMode::StreamK;
    }
    else if (decomposition == "splitk")
    {
      decomp = DecompositionMode::SplitK;
      cmd.get_cmd_line_argument("splits", splits, 1);
    }
    else if (decomposition == "dataparallel")
    {
      decomp = DecompositionMode::DataParallel;
    }
    cmd.get_cmd_line_argument("csv", csv, false);
  }

  std::ostream &print_usage(std::ostream &out) const
  {

    out << "benchmark_hopper\n\n"
        << "  Hopper GEMM benchmark for various tile scheduler configurations.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
        << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
        << "  --decomposition=<string>    Decomposition Mode (heuristic, streamk, splitk, dataparallel)\n\n"
        << "  --splits=<int>              Number of K split (Only used in splitk mode)\n\n"
        << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
        << "  --csv                       Output results in CSV format\n\n";

    out
        << "\n\nExamples:\n\n"
        << "$ ./benchmark_hopper --m=1024 --n=1024 --k=4096 --decomposition=streamk --raster=H \n\n";

    return out;
  }

  double gflops(double runtime_s) const
  {
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
      double avg_runtime_ms = 0,
      double gflops = 0,
      cutlass::Status status = cutlass::Status::kSuccess,
      cudaError_t error = cudaSuccess)
      : avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {
  }
};

template <class Element>
bool initialize_block(
    cutlass::DeviceAllocation<Element> &block,
    uint64_t seed = 1091)
{

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1)
  {
    scope_max = Element(2);
    scope_min = Element(0);
  }
  else if (bits_input <= 8)
  {
    scope_max = Element(2);
    scope_min = Element(-2);
  }
  else
  {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

void initialize(const Options &options)
{

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2394);
  initialize_block(block_B, seed + 4323);
  initialize_block(block_C, seed + 3293);
}

typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
  scheduler_args = {static_cast<int>(options.splits), static_cast<int>(options.swizzle), options.raster, options.decomp};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k},
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info,
      scheduler_args};

  return arguments;
}

bool verify(const Options &options)
{
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  DeviceGemmReference gemm_reference;
  gemm_reference(
      {options.m, options.n, options.k},
      ElementAccumulator(options.alpha),
      ref_A,
      ref_B,
      ElementAccumulator(options.beta),
      ref_C,
      ref_D);

  CUDA_CHECK(cudaDeviceSynchronize());
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
}

template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  Gemm gemm;
  auto arguments = args_from_options(options);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());

  Result result;
  result.passed = verify(options);

  if (!options.csv)
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << '\n';

  if (!result.passed)
  {
    exit(-1);
  }

  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter)
    {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN)
    {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM)
    {
      raster = "Along M";
    }

    std::string decomp = "Heuristic";
    if (options.decomp == DecompositionMode::StreamK)
    {
      decomp = "StreamK";
    }
    else if (options.decomp == DecompositionMode::SplitK)
    {
      decomp = "SplitK";
    }
    else if (options.decomp == DecompositionMode::DataParallel)
    {
      decomp = "DataParallel";
    }

    const int worktile_count = options.m / cute::get<0>(TileShape{}) * options.n / cute::get<1>(TileShape{});

    if (options.csv)
    {
      std::cout << options.m << ',' << options.n << ',' << options.k << ','
                << raster << ',' << options.swizzle << ','
                << decomp << ',' << options.splits << ','
                << result.avg_runtime_ms << "," << result.gflops << ','
                << worktile_count << '\n';
    }
    else
    {
      std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << '\n';
      std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << '\n';
      std::cout << "  Decomposition: " << decomp << ((options.decomp == DecompositionMode::SplitK) ? "with split of " + std::to_string(options.splits) : "") << '\n';
      std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << '\n';
      std::cout << "  GFLOPS: " << result.gflops << '\n';
      std::cout << "  Worktile Count: " << worktile_count << '\n';
    }
  }

  return 0;
}

int main(int argc, char const **args)
{
  Options options;
  options.parse(argc, args);

  if (options.help)
  {
    options.print_usage(std::cout) << '\n';
    return 0;
  }

  run<Gemm>(options);
  return 0;
}
