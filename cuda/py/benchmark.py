"""Benchmark CUDA GEMM kernels against PyTorch."""

import os
from pathlib import Path
from typing import Tuple

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


def get_cuda_code(cuda_file: str, header_file: str) -> Tuple[str, str]:
    """Load CUDA source and header files, removing #include and #pragma once directives."""
    with open(cuda_file) as f:
        cuda_code = "".join(
            [line for line in f.readlines() if not line.startswith("#include")]
        )

    with open(header_file) as f:
        header_code = "".join(
            [line for line in f.readlines()
             if not line.startswith("#include") and not line.startswith("#pragma once")]
        )

    return cuda_code, header_code


def create_cuda_extension():
    """Create PyTorch extension for CUDA GEMM kernels."""
    file_dir = Path(__file__).parent.parent

    # Load all CUDA source files
    naive_cu = file_dir / "01_naive.cu"
    coalesce_cu = file_dir / "02_kernel_global_mem_coalesce.cu"
    header_file = file_dir / "gemm_kernels.cuh"

    print(f"Loading CUDA sources:")
    print(f"  - Naive: {naive_cu}")
    print(f"  - Coalesced: {coalesce_cu}")
    print(f"  - Header: {header_file}")

    # Read all source files
    naive_code, _ = get_cuda_code(str(naive_cu), str(header_file))
    coalesce_code, header_code = get_cuda_code(str(coalesce_cu), str(header_file))

    # Combine CUDA sources
    combined_cuda_code = naive_code + "\n" + coalesce_code

    # Create build directory
    build_dir = file_dir / "build" / "gemm_extension"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"Build directory: {build_dir}")

    # Load the extension
    extension = load_inline(
        name="gemm_cuda_extension",
        cpp_sources=header_code,
        cuda_sources=combined_cuda_code,
        functions=["sgemm_naive", "sgemm_global_mem_coalesce"],
        with_cuda=True,
        verbose=True,
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_dir),
    )

    print("CUDA extension loaded successfully!")
    return extension


# Load CUDA kernels
print("Loading CUDA kernels...")
cuda_kernels = create_cuda_extension()


def cuda_naive_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for naive CUDA GEMM kernel."""
    output = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    return cuda_kernels.sgemm_naive(a, b, 1.0, 0.0, output)


def cuda_coalesced_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for coalesced global memory CUDA GEMM kernel."""
    output = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    return cuda_kernels.sgemm_global_mem_coalesce(a, b, 1.0, 0.0, output)


def torch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation."""
    return torch.matmul(a, b)


def benchmark_kernel(kernel_fn, a, b, warmup=10, iterations=100):
    """Benchmark a GEMM kernel function."""
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(a, b)

    # Benchmark
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        _ = kernel_fn(a, b)
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate times
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Trim outliers (top and bottom 10%)
    times_ms_sorted = sorted(times_ms)
    trim_count = max(1, iterations // 10)
    times_ms_trimmed = times_ms_sorted[trim_count:-trim_count]

    avg_time_ms = sum(times_ms_trimmed) / len(times_ms_trimmed)
    min_time_ms = min(times_ms_trimmed)
    max_time_ms = max(times_ms_trimmed)

    return avg_time_ms, min_time_ms, max_time_ms


def calculate_metrics(M, N, K, avg_time_ms):
    """Calculate TFLOPS and bandwidth."""
    # FLOPs: 2MNK for matrix multiplication
    flops = 2 * M * N * K
    tflops = (flops / (avg_time_ms * 1e-3)) * 1e-12

    # Memory bandwidth: read A (MxK), read B (KxN), write C (MxN)
    bytes_per_element = 4  # float32
    bytes_total = (M * K + K * N + M * N) * bytes_per_element
    bandwidth_gbps = (bytes_total * 1e-9) / (avg_time_ms * 1e-3)

    return tflops, bandwidth_gbps


if __name__ == "__main__":
    # Enable TF32 for PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\nRunning GEMM benchmarks...\n")

    # Test sizes
    test_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for M, N, K in test_sizes:
        print("=" * 80)
        print(f"Matrix dimensions: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
        print("=" * 80)

        # Generate random inputs
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)

        results = []

        # Benchmark PyTorch
        print("\nBenchmarking PyTorch matmul...")
        avg_ms, min_ms, max_ms = benchmark_kernel(torch_gemm, a, b)
        tflops, bandwidth = calculate_metrics(M, N, K, avg_ms)
        results.append(("PyTorch", avg_ms, min_ms, max_ms, tflops, bandwidth))
        print(f"  Avg: {avg_ms:.4f} ms, Min: {min_ms:.4f} ms, Max: {max_ms:.4f} ms")
        print(f"  Performance: {tflops:.2f} TFLOPS, Bandwidth: {bandwidth:.2f} GB/s")

        # Benchmark CUDA naive
        print("\nBenchmarking CUDA naive kernel...")
        avg_ms, min_ms, max_ms = benchmark_kernel(cuda_naive_gemm, a, b)
        tflops, bandwidth = calculate_metrics(M, N, K, avg_ms)
        results.append(("CUDA Naive", avg_ms, min_ms, max_ms, tflops, bandwidth))
        print(f"  Avg: {avg_ms:.4f} ms, Min: {min_ms:.4f} ms, Max: {max_ms:.4f} ms")
        print(f"  Performance: {tflops:.2f} TFLOPS, Bandwidth: {bandwidth:.2f} GB/s")

        # Benchmark CUDA coalesced
        print("\nBenchmarking CUDA coalesced kernel...")
        avg_ms, min_ms, max_ms = benchmark_kernel(cuda_coalesced_gemm, a, b)
        tflops, bandwidth = calculate_metrics(M, N, K, avg_ms)
        results.append(("CUDA Coalesced", avg_ms, min_ms, max_ms, tflops, bandwidth))
        print(f"  Avg: {avg_ms:.4f} ms, Min: {min_ms:.4f} ms, Max: {max_ms:.4f} ms")
        print(f"  Performance: {tflops:.2f} TFLOPS, Bandwidth: {bandwidth:.2f} GB/s")

        # Print comparison
        print("\n" + "-" * 80)
        print("Comparison (baseline: PyTorch)")
        print("-" * 80)
        baseline_time = results[0][1]

        for name, avg_ms, _, _, tflops, bandwidth in results:
            speedup = baseline_time / avg_ms
            if name == "PyTorch":
                print(f"{name:20s}: 1.00x (baseline)")
            else:
                print(f"{name:20s}: {speedup:.2f}x")

        print("\n")
