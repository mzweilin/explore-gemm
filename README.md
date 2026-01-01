# [Learning CUTLASS the hard way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) Code

[IN ACTIVE DEVELOPMENT] If any script doesn't work, please file an issue and I will address it.

## Setup

### Prerequisites

1. **Create and activate a Python environment** (recommended):

   ```bash
   # Using conda (recommended)
   conda create -n gemm python=3.12
   conda activate gemm

   # Or using virtualenv
   python3 -m venv gemm_env
   source gemm_env/bin/activate
   ```

   > **Note:** The setup script will warn you if you're using the conda `base` environment or system Python and ask for confirmation before proceeding.

2. **Install PyTorch with CUDA support** in your environment:

   ```bash
   # Using conda (recommended)
   conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia

   # Or using pip
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

### Running Setup

Once PyTorch is installed and your environment is activated, run:

```bash
./setup.sh
```

The setup script will:
- **Detect your Python environment** (conda/virtualenv/system)
  - Warns if using conda `base` environment (asks for confirmation)
  - Warns if using system Python (asks for confirmation)
- **Detect PyTorch installation** and verify C++ components
  - Automatically finds PyTorch version and CUDA version
  - Validates that libtorch C++ libraries are present
- **Create symlink** at `third-party/libtorch` pointing to your PyTorch installation
- **Download CUTLASS** (version 4.3.0 by default) in `third-party/cutlass`
- **Download Catch2** test framework in `third-party/catch.hpp`
- **Install additional Python packages**: loguru, pandas, plotly, pytest, click, ninja

### Custom CUTLASS Version

You can specify a different CUTLASS version:

```bash
# Specify CUTLASS version
./setup.sh -t 4.2.0              # CUTLASS 4.2.0
./setup.sh --cutlass 4.5.0       # CUTLASS 4.5.0

# See all options
./setup.sh --help
```

**Important Notes:**
- The setup script uses PyTorch from your current conda/virtualenv environment
- This ensures version consistency between Python code and C++ code
- The `third-party/libtorch` directory will be a symlink to your PyTorch installation
- Make sure CUDA is installed and accessible at `/usr/local/cuda-12.8` (or update paths in CMakeLists.txt)

## CUDA Kernels

This repository contains 14 GEMM (General Matrix Multiply) kernel implementations, progressing from naive to optimized CUTLASS:

### FP32 Kernels (Single-Precision)

1. **Naive GEMM** ([cuda/01_naive.cu](cuda/01_naive.cu))
   Basic implementation with one thread per output element

2. **Global Memory Coalescing** ([cuda/02_kernel_global_mem_coalesce.cu](cuda/02_kernel_global_mem_coalesce.cu))
   Optimized thread-to-memory mapping for coalesced global memory access

3. **Shared Memory Tiling** ([cuda/03_kernel_shared_mem.cu](cuda/03_kernel_shared_mem.cu))
   Block-level tiling using shared memory to reduce global memory accesses

![Shared Memory](./shared_mem_gemm.gif)

4. **1D Block Tiling** ([cuda/04_kernel_blocktiling_1d.cu](cuda/04_kernel_blocktiling_1d.cu))
   Enhanced block tiling with 1D thread-level tiling (TM) for increased work per thread

5. **2D Block Tiling** ([cuda/05_kernel_blocktiling_2d.cu](cuda/05_kernel_blocktiling_2d.cu))
   Full 2D thread-level tiling (TM x TN) with register blocking

![2D Block TIled](./2D_tiled_gemm.gif)

6. **Vectorized Memory Access** ([cuda/06_kernel_vectorize.cu](cuda/06_kernel_vectorize.cu))
   Uses float4 vectorized loads/stores for improved memory bandwidth utilization

7. **Warp Tiling (FP32)** ([cuda/07_kernel_warptiling.cu](cuda/07_kernel_warptiling.cu))
   Warp-level tiling hierarchy: Block → Warp → Warp Subtile → Thread tiles

### Mixed Precision Kernels (FP16/BF16/FP32)

8. **Warp Tiling (All Dtypes)** ([cuda/08_kernel_warptiling_all_dtypes.cu](cuda/08_kernel_warptiling_all_dtypes.cu))
   Extended warp tiling kernel with support for FP16, BF16, and FP32 inputs

### Tensor Core Kernels (FP16/BF16 inputs, FP32 accumulation)

9. **Tensor Core Naive** ([cuda/09_kernel_tensorcore_naive.cu](cuda/09_kernel_tensorcore_naive.cu))
   Basic Tensor Core implementation using WMMA API (16x16x16 tiles)

10. **Tensor Core Warp-Tiled** ([cuda/10_kernel_tensorcore_warptiled.cu](cuda/10_kernel_tensorcore_warptiled.cu))
    Warp-level tiling with Tensor Cores for improved occupancy and performance

11. **Tensor Core Double-Buffered** ([cuda/11_kernel_tensorcore_double_buffered.cu](cuda/11_kernel_tensorcore_double_buffered.cu))
    Double buffering technique to overlap memory transfers with computation

12. **Tensor Core Async Pipeline** ([cuda/12_kernel_tensorcore_async.cu](cuda/12_kernel_tensorcore_async.cu))
    Hardware async memory copy (cp.async) with 2-stage pipeline (requires SM 8.0+)

### CUTLASS Library Kernels

13. **CUTLASS GEMM** ([cuda/13_kernel_cutlass.cu](cuda/13_kernel_cutlass.cu))
    NVIDIA CUTLASS library implementation with FP32, FP16, and BF16 support

14. **CUTLASS Autotunable** ([cuda/14_kernel_cutlass_autotunable.cu](cuda/14_kernel_cutlass_autotunable.cu))
    Multiple CUTLASS configurations for autotuning different tile sizes and stages

### CUTLASS Hopper Kernels (SM90+)

15. **CUTLASS Hopper GEMM** ([cuda/15_kernel_cutlass_hopper.cu](cuda/15_kernel_cutlass_hopper.cu))
    CUTLASS 3.x Collective Builder API with Hopper-specific optimizations:
    - TMA (Tensor Memory Accelerator) support
    - GMMA (General Matrix Multiply Accumulate) tensor operations
    - Multiple warp specialization strategies:
      - TMA Warp Specialized
      - TMA Warp Specialized Persistent
      - TMA Warp Specialized Pingpong
      - TMA Warp Specialized StreamK
    - Supports FP16 and BF16 precision
    - Requires H100, H200, or newer Hopper GPUs (SM90+)

16. **CUTLASS Hopper Autotunable** ([cuda/16_kernel_cutlass_hopper_autotunable.cu](cuda/16_kernel_cutlass_hopper_autotunable.cu))
    Runtime-configurable Hopper kernel with extensive parameter control:
    - **Tile sizes**: 128x256x64, 128x128x64
    - **Decomposition modes**: Heuristic, StreamK, SplitK, DataParallel
    - **Raster orders**: Along M, Along N, Heuristic
    - **Swizzle factors**: 1, 2, 4, 8
    - **Split K values**: 1, 2, 3, 4 (for SplitK decomposition)
    - Cluster configuration: 2x1x1
    - Supports BF16 precision (FP16 planned)
    - Used by Python autotuning infrastructure

All kernels include PyTorch tensor wrappers for easy integration. See [cuda/gemm_kernels.cuh](cuda/gemm_kernels.cuh) for the API.

## GPU Architecture Configuration

The CUTLASS kernels (13 and 14) support different GPU architectures through a configuration setting in [cuda/utils.cuh](cuda/utils.cuh:17).

### Supported Architectures

- **SM80**: Ampere (A100, RTX 3090, etc.)
- **SM89**: Ada Lovelace (RTX 4090, etc.) - **Default**
- **SM90**: Hopper (H100, etc.)

### Changing Architecture

To target a different GPU architecture, edit the `GPU_SM_ARCH` constant in [cuda/utils.cuh](cuda/utils.cuh):

```cpp
// For H100 GPUs (Hopper)
constexpr int GPU_SM_ARCH = 90;

// For A100 GPUs (Ampere)
constexpr int GPU_SM_ARCH = 80;

// For RTX 4090 GPUs (Ada Lovelace) - Default
constexpr int GPU_SM_ARCH = 89;
```

After changing the architecture, rebuild the project:

```bash
cmake --build build --target test_gemm_cutlass
```

## Building

Build the project with CMake:

```bash
# Configure build (automatically detects GPU architecture via nvidia-smi)
cmake -B build

# Or specify CUDA architecture explicitly
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89   # Ada (RTX 4090, etc.)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a  # Hopper (H100, etc.)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100  # Blackwell (GB10, etc.)

# Build all targets
cmake --build build

# Build specific test executable
cmake --build build --target test_gemm_tensorcore
```

**Auto-Detection Features:**
- CMake automatically detects your GPU's compute capability using `nvidia-smi`
- Falls back to SM89 if GPU detection fails
- Prevents "CMAKE_CUDA_ARCHITECTURES must be non-empty" errors on all systems

### Hopper Kernel Compilation

Hopper-specific kernels (SM90+) are **automatically detected and conditionally compiled**:

- **Auto-detection**: CMake uses `nvidia-smi` to detect your GPU's compute capability
- **SM89 or lower** (RTX 4090, A100, etc.): Hopper kernels are **skipped**
- **SM90 or higher** (H100, H200, etc.): Hopper kernels are **built**

To manually control Hopper kernel compilation:

```bash
# Force enable Hopper kernels (requires setting SM90a architecture)
cmake -B build -DBUILD_HOPPER_KERNELS=ON -DCMAKE_CUDA_ARCHITECTURES=90a

# Force disable Hopper kernels
cmake -B build -DBUILD_HOPPER_KERNELS=OFF
```

> **Note**: Attempting to run Hopper kernels on non-Hopper GPUs will result in runtime errors. The Python autotuning scripts automatically detect GPU capability and skip Hopper kernels on incompatible hardware.

## Testing

### Running C++ Tests

Run all tests with CTest:

```bash
# Run all tests
ctest --test-dir build

# Run tests with verbose output
ctest --test-dir build --verbose

# Run specific test
ctest --test-dir build -R test_gemm_naive
```

Or run test executables directly:

```bash
# Run specific kernel tests
./build/test_gemm_naive
./build/test_gemm_coalesce
./build/test_gemm_shared_mem
./build/test_gemm_blocktiling1d
./build/test_gemm_blocktiling2d
./build/test_gemm_vectorize
./build/test_gemm_warptiling
./build/test_gemm_warptiling_all_dtypes
./build/test_gemm_tensorcore_naive
./build/test_gemm_tensorcore
./build/test_gemm_tensorcore_double_buffered
./build/test_gemm_tensorcore_async
./build/test_gemm_cutlass

# Run Hopper kernel tests (requires H100/H200 or newer)
./build/test_gemm_cutlass_hopper  # Tests BF16 precision only
```

**Hopper Test Details:**
- Tests multiple matrix sizes: 128x128, 256x256, 512x512, 1024x1024, 2048x2048
- Validates numerical correctness against reference implementations
- Only available when compiled on SM90+ GPUs
- Automatically skipped on non-Hopper hardware

### Running Python Tests

Run Python unit tests with pytest:

```bash
cd cuda/py

# Run all tests
pytest test_kernels.py -v

# Run specific test
pytest test_kernels.py::test_fp32_kernels -v

# Run tests for specific matrix size
pytest test_kernels.py -v -k "size512"
```

## Benchmarking

Benchmark kernels against PyTorch using the Python benchmark script:

```bash
cd cuda/py

# Benchmark all FP32 kernels (default)
python benchmark.py

# Benchmark all FP16-compatible kernels
python benchmark.py -d float16

# Benchmark all BF16-compatible kernels
python benchmark.py -d bfloat16

# Benchmark specific kernels with FP32
python benchmark.py -k pytorch -k naive -k tensorcore_async_fp16

# Benchmark specific kernels with FP16
python benchmark.py -d float16 -k pytorch -k warptiling -k tensorcore_fp16
```

**Available kernel names for benchmarking:**
- All dtypes: `pytorch`, `warptiling`
- FP32 only: `naive`, `global_mem_coalesce`, `shared_mem`, `blocktiling_1d`, `blocktiling_2d`, `vectorize`, `cutlass_fp32`
- FP16/BF16 only: `tensorcore_fp16`, `tensorcore_bf16`, `tensorcore_db_fp16`, `tensorcore_db_bf16`, `tensorcore_async_fp16`, `tensorcore_async_bf16`, `cutlass_fp16`, `cutlass_bf16`

The benchmark script generates interactive HTML plots showing performance comparisons across different matrix sizes.

### C++ Benchmark Binaries

For low-level performance profiling with direct C++ implementations:

```bash
# Build benchmark binaries
cmake --build build --target benchmark_hopper
cmake --build build --target benchmark_blackwell

# Run Hopper benchmark (SM90, H100/H200)
./build/benchmark_hopper --m=1024 --n=1024 --k=1024

# Run Blackwell benchmark (SM100, GB10/GB200)
./build/benchmark_blackwell --m=1024 --n=1024 --k=1024
```

**Hopper Benchmark Options** ([cuda/scripts/benchmark_hopper.cu](cuda/scripts/benchmark_hopper.cu)):
```bash
# Basic usage
./build/benchmark_hopper --m=<int> --n=<int> --k=<int>

# Configuration options:
--m=<int>                    Matrix dimension M (rows of A and C)
--n=<int>                    Matrix dimension N (columns of B and C)
--k=<int>                    Matrix dimension K (columns of A, rows of B)
--alpha=<float>              Epilogue scalar alpha (default: 1.0)
--beta=<float>               Epilogue scalar beta (default: 0.0)
--raster=<char>              CTA rasterization order:
                               N = Along N dimension
                               M = Along M dimension
                               H = Heuristic (default)
--swizzle=<int>              CTA rasterization swizzle factor (default: 1)
--decomposition=<string>     Work decomposition mode:
                               heuristic (default)
                               streamk
                               splitk
                               dataparallel
--splits=<int>               Number of K splits (for splitk mode, default: 1)
--iterations=<int>           Profiling iterations (default: 20)
--csv                        Output results in CSV format

# Example: StreamK with heuristic rasterization
./build/benchmark_hopper --m=1024 --n=1024 --k=4096 --decomposition=streamk --raster=H

# Example: SplitK with 4 splits along N dimension
./build/benchmark_hopper --m=2048 --n=2048 --k=2048 --decomposition=splitk --splits=4 --raster=N
```

**Blackwell Benchmark Options** ([cuda/scripts/benchmark_blackwell.cu](cuda/scripts/benchmark_blackwell.cu)):
Similar options to Hopper benchmark but optimized for SM100 architecture with:
- Dynamic cluster shapes
- 2-SM TCGEN05 MMA instructions
- MMA tile shape: 256x128x64
- Requires GB10, GB200, or newer Blackwell GPUs

**Benchmark Features:**
- **Data type**: FP16 (half precision) with FP32 accumulation
- **Validation**: Compares results against reference GEMM implementation
- **Performance metrics**: Runtime (ms), TFLOPS, relative performance
- **Tile configuration** (Hopper): 128x256x64 with 1x1x1 cluster
- **Tile configuration** (Blackwell): 256x128x64 with dynamic cluster
- **Memory layouts**: A=RowMajor, B=ColumnMajor, C=ColumnMajor

## Autotuning

### CUTLASS Autotuning (Ampere/Ada)

Autotune CUTLASS kernel configurations to find optimal settings for different matrix sizes:

```bash
cd cuda/py

# Autotune FP16 kernels for all power-of-2 sizes from 64 to 8192
python autotune_cutlass.py -d float16

# Autotune BF16 kernels
python autotune_cutlass.py -d bfloat16

# Autotune specific matrix sizes
python autotune_cutlass.py -d float16 --sizes 128 256 512 1024

# Load and use cached autotuning results
python autotune_cutlass.py -d float16 --load-cache --size 1024
```

The autotuner tests all available CUTLASS configurations (different tile sizes, warp configurations, and pipeline stages) and caches the best configuration for each matrix size. Results are saved as JSON files for future use.

### CUTLASS Hopper Autotuning (H100/H200)

Autotune CUTLASS Hopper kernels with comprehensive scheduler parameter exploration:

```bash
cd cuda/py

# Autotune BF16 Hopper kernels for all sizes (128 to 8192)
python autotune_cutlass_hopper.py

# Autotune specific matrix sizes
python autotune_cutlass_hopper.py --sizes 128 256 512 1024

# Disable L2 cache flushing for faster (but less accurate) benchmarking
python autotune_cutlass_hopper.py --no-cache-flush

# Enable adaptive iteration counts based on kernel runtime
python autotune_cutlass_hopper.py --adaptive-iters --target-time 2000

# Load and visualize cached results
python autotune_cutlass_hopper.py --load-cache
```

**Hopper Autotuning Features:**
- **Tile configurations tested**: 128x256x64, 128x128x64
- **Decomposition modes**: Heuristic, StreamK, SplitK, DataParallel
- **Raster orders**: Along M, Along N, Heuristic
- **Swizzle factors**: 1, 2, 4, 8
- **Total configurations**: ~192 combinations per matrix size
- **Default matrix sizes**: 128, 256, 512, 1024, 2048, 4096, 6144, 8192
- **Precision support**: BF16 (FP16 planned)

**Benchmarking Best Practices:**
- L2 cache flushing enabled by default (following Triton benchmarking methodology)
- Fixed iteration counts (warmup=10, iterations=100) for reproducibility
- Median-based statistical comparison with outlier removal
- Proper CUDA synchronization and event timing

**Output Files** (saved to `cuda/py/autotune_results/`):
- `autotune_hopper_results_bfloat16.html`: Interactive visualization with:
  - Performance comparison chart (TFLOPS)
  - Speedup vs PyTorch chart
  - Heatmap showing all configurations
  - Best configuration summary table
- `autotune_hopper_cache_bfloat16.json`: Cached benchmark results
- `autotune_hopper_results_bfloat16.csv`: Full results for all configurations
- `best_configs_summary_bfloat16.csv`: Summary of optimal configurations per size

**Requirements:**
- H100, H200, or newer Hopper GPU (SM90+)
- Automatically detects GPU capability and skips on non-Hopper hardware
- Python packages: torch, loguru, pandas, plotly, click

### Hopper Benchmark Binary

For C++ benchmarking of Hopper kernels:

```bash
# Run comprehensive benchmark suite (auto-detects GPU architecture)
cd cuda/scripts
./benchmarks.sh

# The script automatically:
# - Detects Hopper (SM90) vs Blackwell (SM100) architecture
# - Runs benchmarks for matrix sizes 128 to 8192
# - Tests all decomposition modes and raster orders
# - Saves results to CSV files with progress tracking
```

**Benchmark Shell Script** ([cuda/scripts/benchmarks.sh](cuda/scripts/benchmarks.sh:1)):
- **Auto-detects GPU architecture**: Hopper (SM90) vs Blackwell (SM100)
- **Problem sizes**: 128x128x128 to 8192x8192x8192
- **Decomposition modes**: Heuristic, StreamK, SplitK, DataParallel
- **Raster orders**: Along M, Along N, Heuristic
- **Swizzle factors**: 1, 2, 4, 8
- **CSV output**: Detailed performance metrics with progress tracking
- **Execution**: Runs all combinations systematically for comprehensive performance analysis

