"""Generic benchmark utilities for GEMM kernels."""

from typing import Any, Callable, TypedDict
import torch
from loguru import logger
import pandas as pd


# Registry for GEMM kernels
GEMM_KERNELS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}


class BenchmarkResults(TypedDict):
    """Type definition for benchmark results."""

    backend: str
    M: int
    N: int
    K: int
    dtype: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    tflops: float
    bandwidth_gbps: float
    arithmetic_intensity: float
    total_flops: int
    total_bytes: int


def benchmark_gemm(
    gemm_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    M: int,
    N: int,
    K: int,
    backend: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    warmup: int = 10,
    iterations: int = 100,
) -> BenchmarkResults:
    """
    Generic benchmark for GEMM (General Matrix Multiply) kernels.

    Args:
        gemm_fn: Matrix multiplication function to benchmark.
                 Should accept two tensors (A, B) and return C = A @ B.
        M: Number of rows in matrix A
        N: Number of columns in matrix B
        K: Number of columns in A / rows in B
        backend: Name/label for the backend being benchmarked
        dtype: Data type for matrices
        device: Device to run on
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        BenchmarkResults dict with performance metrics including:
        - Timing statistics (avg, min, max)
        - TFLOPS (trillions of floating point operations per second)
        - Memory bandwidth (GB/s)
        - Arithmetic intensity (FLOPs/byte)
    """
    # Generate random inputs
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        _ = gemm_fn(a, b)

    # Benchmark
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        _ = gemm_fn(a, b)
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

    # Calculate performance metrics
    # FLOPs: For matrix multiplication C = A @ B where A is MxK, B is KxN
    # Each output element requires K multiplications and K-1 additions = 2K-1 ops
    # Total FLOPs = M * N * (2K - 1) ≈ 2MNK for large K
    flops = 2 * M * N * K
    tflops = (flops / (avg_time_ms * 1e-3)) * 1e-12

    # Memory bandwidth:
    # Read A (MxK elements), Read B (KxN elements), Write C (MxN elements)
    bytes_per_element = a.element_size()
    bytes_read = (M * K + K * N) * bytes_per_element
    bytes_write = M * N * bytes_per_element
    bytes_total = bytes_read + bytes_write
    bandwidth_gbps = (bytes_total * 1e-9) / (avg_time_ms * 1e-3)

    # Arithmetic intensity (FLOPs per byte)
    arithmetic_intensity = flops / bytes_total

    return {
        "backend": backend,
        "M": M,
        "N": N,
        "K": K,
        "dtype": str(dtype),
        "avg_time_ms": avg_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "tflops": tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "arithmetic_intensity": arithmetic_intensity,
        "total_flops": flops,
        "total_bytes": bytes_total,
    }


def print_benchmark_results(results: BenchmarkResults) -> None:
    """Pretty print benchmark results."""
    logger.info(f"🚀 Backend: {results['backend']}")
    logger.info(
        f"📊 Matrix dimensions: ({results['M']}, {results['K']}) @ ({results['K']}, {results['N']}) = ({results['M']}, {results['N']})"
    )
    logger.info(f"🔢 Data type: {results['dtype']}")
    logger.info("=" * 40)
    logger.info(f"⏱️  Average time: {results['avg_time_ms']:.4f} ms")
    logger.info(f"⚡ Min time:     {results['min_time_ms']:.4f} ms")
    logger.info(f"🐌 Max time:     {results['max_time_ms']:.4f} ms")
    logger.success(f"💪 Performance:  {results['tflops']:.2f} TFLOPS")
    logger.success(f"🌊 Bandwidth:    {results['bandwidth_gbps']:.2f} GB/s")
    logger.info(
        f"🧮 Arithmetic Intensity: {results['arithmetic_intensity']:.2f} FLOPs/byte"
    )
    logger.info("=" * 40)


def compare_benchmarks(
    results_list: list[BenchmarkResults], baseline_idx: int = 0
) -> None:
    """
    Compare multiple benchmark results and print speedup analysis.

    Args:
        results_list: List of BenchmarkResults to compare
        baseline_idx: Index of the baseline result for speedup calculation
    """
    if len(results_list) < 2:
        logger.warning("Need at least 2 results to compare")
        return

    baseline = results_list[baseline_idx]
    logger.info("=" * 60)
    logger.info(
        f"📈 Comparison (baseline: {baseline['backend']} @ {baseline['avg_time_ms']:.4f} ms)"
    )
    logger.info("=" * 60)

    for i, result in enumerate(results_list):
        if i == baseline_idx:
            continue

        speedup = baseline["avg_time_ms"] / result["avg_time_ms"]
        tflops_improvement = (
            (result["tflops"] - baseline["tflops"]) / baseline["tflops"] * 100
        )

        if speedup > 1:
            logger.success(
                f"🏆 {result['backend']}: {speedup:.2f}x faster ({tflops_improvement:+.1f}% TFLOPS)"
            )
        elif speedup < 1:
            logger.warning(
                f"🐢 {result['backend']}: {speedup:.2f}x slower ({tflops_improvement:+.1f}% TFLOPS)"
            )
        else:
            logger.info(
                f"🤝 {result['backend']}: {speedup:.2f}x (tied, {tflops_improvement:+.1f}% TFLOPS)"
            )

    logger.info("=" * 60)


def print_markdown_comparison_table(
    results_list: list[BenchmarkResults], baseline_idx: int = 0
) -> None:
    """
    Print a markdown table comparing benchmark results across all kernels using pandas.

    Args:
        results_list: List of BenchmarkResults to compare
        baseline_idx: Index of the baseline result for speedup calculation
    """
    if len(results_list) == 0:
        logger.warning("No results to display")
        return

    baseline = results_list[baseline_idx]

    # Prepare data for dataframe
    data = []
    for i, result in enumerate(results_list):
        if i == baseline_idx:
            speedup = 1.0
            speedup_str = "1.00x 🎯"
            improvement = 0.0
        else:
            speedup = baseline["avg_time_ms"] / result["avg_time_ms"]
            if speedup > 1:
                speedup_str = f"{speedup:.2f}x 🏆"
            elif speedup < 1:
                speedup_str = f"{speedup:.2f}x 🐢"
            else:
                speedup_str = f"{speedup:.2f}x 🤝"
            improvement = (
                (result["tflops"] - baseline["tflops"]) / baseline["tflops"] * 100
            )

        data.append(
            {
                "Backend": result["backend"],
                "Avg Time (ms)": f"{result['avg_time_ms']:.4f}",
                "TFLOPS": f"{result['tflops']:.2f}",
                "Bandwidth (GB/s)": f"{result['bandwidth_gbps']:.2f}",
                "Speedup": speedup_str,
                "TFLOPS Δ": f"{improvement:+.1f}%" if i != baseline_idx else "-",
            }
        )

    # Create dataframe and print as markdown
    df = pd.DataFrame(data)
    logger.info("## 📊 Benchmark Comparison\n")
    logger.info("\n" + df.to_markdown(index=False))
    logger.info(
        f"**Matrix Dimensions:** ({results_list[0]['M']}, {results_list[0]['K']}) @ ({results_list[0]['K']}, {results_list[0]['N']}) = ({results_list[0]['M']}, {results_list[0]['N']})"
    )
    logger.info(f"**Data Type:** {results_list[0]['dtype']}")


def register_kernel(name: str):
    """
    Decorator to register a GEMM kernel for benchmarking.

    Example:
        @register_kernel("naive")
        def matmul_naive(a, b):
            return c
    """

    def decorator(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        GEMM_KERNELS[name] = fn
        return fn

    return decorator


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Import kernel implementations
    # Add the triton directory to path if needed
    sys.path.insert(0, str(Path(__file__).parent))

    from naive import matmul_naive
    from coalesced import matmul_coalesced

    # Register kernels
    GEMM_KERNELS["triton_naive"] = matmul_naive
    GEMM_KERNELS["triton_coalesced"] = matmul_coalesced
    GEMM_KERNELS["torch"] = torch.matmul

    logger.info("🎯 Running GEMM benchmarks...")

    # First, print IR and PTX for a small example
    logger.info("\n📝 Generating IR and PTX for 512x512x512 matrix...")
    a_small = torch.randn((512, 512), device="cuda", dtype=torch.float32)
    b_small = torch.randn((512, 512), device="cuda", dtype=torch.float32)
    _ = matmul_naive(a_small, b_small, print_ir=True, print_ptx=True)

    # Test multiple sizes
    test_sizes = [
        (1024, 1024, 1024),
    ]

    for M, N, K in test_sizes:
        logger.info(f"\n📐 Benchmarking {M}x{K} @ {K}x{N}:")

        all_results = []

        # Benchmark all registered kernels
        for kernel_name, kernel_fn in GEMM_KERNELS.items():
            logger.info(f"\n🔥 Testing {kernel_name}...")
            results = benchmark_gemm(
                gemm_fn=kernel_fn,
                M=M,
                N=N,
                K=K,
                backend=kernel_name,
            )
            print_benchmark_results(results)
            all_results.append(results)

        # Compare results (using torch as baseline)
        torch_idx = next(
            (i for i, r in enumerate(all_results) if r["backend"] == "torch"), 0
        )
        compare_benchmarks(all_results, baseline_idx=torch_idx)

        # Print markdown comparison table
        print_markdown_comparison_table(all_results, baseline_idx=torch_idx)
