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
    - vectorize: CUDA GEMM with vectorized memory access
    - warptiling: CUDA GEMM with warp-level tiling (most optimized FP32)
    - tensorcore_fp16: CUDA Tensor Core with FP16 inputs
    - tensorcore_bf16: CUDA Tensor Core with BF16 inputs
    - tensorcore_db_fp16: CUDA Tensor Core with double buffering (FP16)
    - tensorcore_db_bf16: CUDA Tensor Core with double buffering (BF16)
"""

import os
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
from loguru import logger

# Import the shared CUDA extension loader
from cuda_extension_loader import create_cuda_extension


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Load CUDA kernels (with less verbose output for kernel runner)
logger.info("🚀 Loading CUDA kernels...")
cuda_kernels = create_cuda_extension(verbose=False)
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


def run_vectorize_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run vectorized CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_vectorize(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_warptiling_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run warptiling CUDA GEMM kernel."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_warptiling_default(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_fp16_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with FP16 inputs."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_bf16_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with BF16 inputs."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_db_fp16_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with FP16 inputs and double buffering."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_double_buffered_fp16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_tensorcore_db_bf16_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run Tensor Core CUDA GEMM kernel with BF16 inputs and double buffering."""
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_tensorcore_double_buffered_bf16(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def run_kernel_n_times(
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    n: int,
) -> None:
    """Run a kernel N times for profiling."""
    logger.info(f"⚡ Running kernel {n} times...")

    for _ in range(n):
        _ = kernel_fn(a, b)

    # Synchronize to ensure all kernels complete
    torch.cuda.synchronize()
    logger.success(f"✅ Completed {n} iterations")


@click.command()
@click.option(
    "-k",
    "--kernel",
    type=click.Choice(
        [
            "naive",
            "global_mem_coalesce",
            "shared_mem",
            "blocktiling_1d",
            "blocktiling_2d",
            "vectorize",
            "warptiling",
            "tensorcore_fp16",
            "tensorcore_bf16",
            "tensorcore_db_fp16",
            "tensorcore_db_bf16",
        ],
        case_sensitive=False,
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
@click.option(
    "-d",
    "--dtype",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default="float32",
    help="Data type for input matrices (default: float32). Tensor Core kernels require float16 or bfloat16.",
)
def main(kernel: str, iterations: int, size: int, dtype: str):
    """Run CUDA GEMM kernels for NCU profiling.

    Examples:
        # Run naive kernel 100 times
        python kernel_runner.py -k naive -n 100

        # Run with FP16 and Tensor Cores
        python kernel_runner.py -k tensorcore_fp16 -n 100 -d float16

        # Run with custom matrix size
        python kernel_runner.py -k shared_mem -n 50 -s 2048

        # Profile with ncu
        ncu --set full python kernel_runner.py -k naive -n 100
        ncu --metrics all python kernel_runner.py -k tensorcore_fp16 -n 100 -d float16
    """
    # Map kernel names to functions
    kernel_map = {
        "naive": run_naive_kernel,
        "global_mem_coalesce": run_coalesced_kernel,
        "shared_mem": run_shared_mem_kernel,
        "blocktiling_1d": run_blocktiling_1d_kernel,
        "blocktiling_2d": run_blocktiling_2d_kernel,
        "vectorize": run_vectorize_kernel,
        "warptiling": run_warptiling_kernel,
        "tensorcore_fp16": run_tensorcore_fp16_kernel,
        "tensorcore_bf16": run_tensorcore_bf16_kernel,
        "tensorcore_db_fp16": run_tensorcore_db_fp16_kernel,
        "tensorcore_db_bf16": run_tensorcore_db_bf16_kernel,
    }

    # Auto-detect required dtype for Tensor Core kernels if not specified
    if kernel in ["tensorcore_fp16", "tensorcore_db_fp16"] and dtype == "float32":
        logger.warning(
            f"⚠️  {kernel} requires float16 dtype, auto-switching to float16"
        )
        dtype = "float16"
    elif kernel in ["tensorcore_bf16", "tensorcore_db_bf16"] and dtype == "float32":
        logger.warning(
            f"⚠️  {kernel} requires bfloat16 dtype, auto-switching to bfloat16"
        )
        dtype = "bfloat16"

    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    kernel_fn = kernel_map[kernel]
    M = N = K = size

    logger.info(f"\n{'='*60}")
    logger.info(f"🎯 Kernel: {kernel}")
    logger.info(f"📐 Matrix size: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    logger.info(f"🔢 Data type: {dtype}")
    logger.info(f"🔄 Iterations: {iterations}")
    logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"{'='*60}\n")

    # Create input tensors with specified dtype
    logger.info("💾 Allocating input tensors...")
    a = torch.randn((M, K), device="cuda", dtype=torch_dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch_dtype)
    logger.success("✅ Input tensors allocated")

    # Run the kernel N times
    logger.info("")
    run_kernel_n_times(kernel_fn, a, b, iterations)

    logger.info(f"\n{'='*60}")
    logger.success("🎉 Done!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
