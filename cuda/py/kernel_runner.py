"""Run CUDA GEMM kernels for NCU profiling.

This script allows you to dry run any registered kernel N times for profiling with NVIDIA Nsight Compute (ncu).

Usage:
    # Run naive kernel 100 times with default matrix size (1024)
    python kernel_runner.py -k naive -n 100

    # Run with custom matrix size
    python kernel_runner.py -k shared_mem -n 50 -s 2048

    # Profile with ncu
    ncu --set full python kernel_runner.py -k naive -n 100
    ncu --metrics all python kernel_runner.py -k global_mem_coalesce -n 100 -s 512

Available kernels:
    - naive: Naive CUDA GEMM kernel
    - global_mem_coalesce: CUDA GEMM with global memory coalescing
    - shared_mem: CUDA GEMM with shared memory tiling
    - blocktiling_1d: CUDA GEMM with 1D block tiling
    - blocktiling_2d: CUDA GEMM with 2D block tiling
"""

import os
from pathlib import Path
from typing import Callable

import click

# Set CUDA paths to match CMakeLists.txt configuration
# Must be set BEFORE importing torch
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
os.environ["CUDA_PATH"] = "/usr/local/cuda-12.8"
os.environ["PATH"] = f"/usr/local/cuda-12.8/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/cuda-12.8/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import torch
from torch.utils.cpp_extension import load_inline
from loguru import logger


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_cuda_code(cuda_file: str, header_file: str) -> tuple[str, str]:
    """Load CUDA source and header files, removing #include and #pragma once directives."""
    with open(cuda_file) as f:
        cuda_code = "".join(
            [line for line in f.readlines() if not line.startswith("#include")]
        )

    with open(header_file) as f:
        header_code = "".join(
            [
                line
                for line in f.readlines()
                if not line.startswith("#include")
                and not line.startswith("#pragma once")
            ]
        )

    return cuda_code, header_code


def create_cuda_extension():
    """Create PyTorch extension for CUDA GEMM kernels."""
    file_dir = Path(__file__).parent.parent

    # Load all CUDA source files
    naive_cu = file_dir / "01_naive.cu"
    coalesce_cu = file_dir / "02_kernel_global_mem_coalesce.cu"
    shared_mem_cu = file_dir / "03_kernel_shared_mem.cu"
    blocktiling_1d_cu = file_dir / "04_kernel_blocktiling_1d.cu"
    blocktiling_2d_cu = file_dir / "05_kernel_blocktiling_2d.cu"
    header_file = file_dir / "gemm_kernels.cuh"

    # Read all source files
    naive_code, _ = get_cuda_code(str(naive_cu), str(header_file))
    coalesce_code, _ = get_cuda_code(str(coalesce_cu), str(header_file))
    shared_mem_code, _ = get_cuda_code(str(shared_mem_cu), str(header_file))
    blocktiling_1d_code, _ = get_cuda_code(str(blocktiling_1d_cu), str(header_file))
    blocktiling_2d_code, header_code = get_cuda_code(str(blocktiling_2d_cu), str(header_file))

    # Combine CUDA sources
    combined_cuda_code = naive_code + "\n" + coalesce_code + "\n" + shared_mem_code + "\n" + blocktiling_1d_code + "\n" + blocktiling_2d_code

    # Create build directory
    build_dir = file_dir / "build" / "gemm_extension"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Load the extension
    extension = load_inline(
        name="gemm_cuda_extension",
        cpp_sources=header_code,
        cuda_sources=combined_cuda_code,
        functions=["sgemm_naive", "sgemm_global_mem_coalesce", "sgemm_shared_mem", "sgemm_blocktiling_1d", "sgemm_blocktiling_2d"],
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_dir),
    )

    return extension


# Load CUDA kernels
logger.info("🚀 Loading CUDA kernels...")
cuda_kernels = create_cuda_extension()
logger.success("✅ CUDA kernels loaded successfully!")


def run_naive_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run naive CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_naive(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_coalesced_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run coalesced global memory CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_global_mem_coalesce(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_shared_mem_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run shared memory CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_shared_mem(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_blocktiling_1d_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run 1D block tiling CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_blocktiling_1d(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_blocktiling_2d_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run 2D block tiling CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_blocktiling_2d(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_kernel_n_times(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    n: int,
) -> None:
    """Run a kernel N times for profiling."""
    logger.info(f"⚡ Running kernel {n} times...")

    for i in range(n):
        _ = kernel_fn(a, b)

    # Synchronize to ensure all kernels complete
    torch.cuda.synchronize()
    logger.success(f"✅ Completed {n} iterations")


@click.command()
@click.option(
    "-k",
    "--kernel",
    type=click.Choice(
        ["naive", "global_mem_coalesce", "shared_mem", "blocktiling_1d", "blocktiling_2d"], case_sensitive=False
    ),
    required=True,
    help="Kernel to run",
)
@click.option(
    "-n",
    "--iterations",
    type=int,
    default=100,
    help="Number of times to run the kernel (default: 100)",
)
@click.option(
    "-s",
    "--size",
    type=int,
    default=4096,
    help="Matrix size (M=N=K, default: 4096)",
)
def main(kernel: str, iterations: int, size: int):
    """Run CUDA GEMM kernels for NCU profiling.

    Examples:
        # Run naive kernel 100 times
        python kernel_runner.py -k naive -n 100

        # Run with custom matrix size
        python kernel_runner.py -k shared_mem -n 50 -s 2048

        # Profile with ncu
        ncu --set full python kernel_runner.py -k naive -n 100
        ncu --metrics all python kernel_runner.py -k global_mem_coalesce -n 100 -s 512
    """
    # Map kernel names to functions
    kernel_map = {
        "naive": run_naive_kernel,
        "global_mem_coalesce": run_coalesced_kernel,
        "shared_mem": run_shared_mem_kernel,
        "blocktiling_1d": run_blocktiling_1d_kernel,
        "blocktiling_2d": run_blocktiling_2d_kernel,
    }

    kernel_fn = kernel_map[kernel]
    M = N = K = size

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Kernel: {kernel}")
    logger.info(f"📐 Matrix size: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    logger.info(f"🔄 Iterations: {iterations}")
    logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"{'='*60}\n")

    # Create input tensors
    logger.info("💾 Allocating input tensors...")
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    logger.success("✅ Input tensors allocated")

    # Run the kernel N times
    logger.info("")
    run_kernel_n_times(kernel_fn, a, b, iterations)

    logger.info(f"\n{'='*60}")
    logger.success("🎉 Done!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
