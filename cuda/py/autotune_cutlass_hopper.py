"""Autotune CUTLASS Hopper GEMM kernels to find optimal configurations.

This script benchmarks all CUTLASS Hopper (SM90) kernel configurations across
different matrix sizes and finds the best configuration for each size. These
kernels use the CUTLASS 3.x Collective Builder API with TMA and warp specialization.

Note: Only BF16 is supported. FP16 support has been removed.

Usage:
    # Autotune BF16 kernels for all power-of-2 sizes from 64 to 8192
    python autotune_cutlass_hopper.py -d bfloat16

    # Autotune specific sizes
    python autotune_cutlass_hopper.py -d bfloat16 --sizes 128 256 512 1024

    # Load and use cached results
    python autotune_cutlass_hopper.py -d bfloat16 --load-cache --size 1024
"""

import os
import json
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import click

# Set CUDA paths BEFORE importing torch
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["CUDA_PATH"] = "/usr/local/cuda"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import torch
import numpy as np
from loguru import logger
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the shared CUDA extension loader
from cuda_extension_loader import create_cuda_extension


# Cache-flushing tensor size (16MB to flush typical L2 caches)
_CACHE_FLUSH_SIZE = 4 * 1024 * 1024  # 16MB in float32 elements

# Global cache flush buffer (lazily initialized)
_cache_flush_buffer = None


def get_cache_flush_buffer():
    """Get or create the cache flush buffer."""
    global _cache_flush_buffer
    if _cache_flush_buffer is None:
        _cache_flush_buffer = torch.empty(
            _CACHE_FLUSH_SIZE, dtype=torch.int8, device="cuda"
        )
    return _cache_flush_buffer


def flush_l2_cache():
    """Flush the L2 cache by performing a dummy operation on a large buffer.

    This helps eliminate cache state artifacts between benchmark iterations.
    Following Triton's approach of clearing L2 before each measurement.
    """
    cache_buffer = get_cache_flush_buffer()
    # Perform a simple operation to flush cache
    cache_buffer.zero_()
    torch.cuda.synchronize()


# Load CUDA kernels
logger.info("🚀 Loading CUTLASS Hopper autotunable kernels...")
cuda_kernels = create_cuda_extension(verbose=True, load_autotune_kernels=True)

# Base configurations (tile and cluster shapes)
HOPPER_BASE_CONFIGS = [
    {"name": "T128x128x64_C2x1x1", "tile": (128, 128, 64), "cluster": (2, 1, 1)},
    {"name": "T128x256x64_C2x1x1", "tile": (128, 256, 64), "cluster": (2, 1, 1)},
    {"name": "T256x128x64_C1x2x1", "tile": (256, 128, 64), "cluster": (1, 2, 1)},
    {"name": "T128x128x128_C2x1x1", "tile": (128, 128, 128), "cluster": (2, 1, 1)},
    {"name": "T256x256x64_C2x2x1", "tile": (256, 256, 64), "cluster": (2, 2, 1)},
    {"name": "T128x64x64_C2x1x1", "tile": (128, 64, 64), "cluster": (2, 1, 1)},
    {"name": "T64x128x64_C1x2x1", "tile": (64, 128, 64), "cluster": (1, 2, 1)},
    {"name": "T64x64x128_C1x1x1", "tile": (64, 64, 128), "cluster": (1, 1, 1)},
    {"name": "T128x128x64_C1x1x1", "tile": (128, 128, 64), "cluster": (1, 1, 1)},
]

# Stage count variants: Auto, 3, 4, 5, 6
STAGE_VARIANTS = ["Auto", "S3", "S4", "S5"]

# Generate all 45 configurations (9 base configs × 5 stage variants)
HOPPER_CONFIG_METADATA = []
config_id = 0
for base_idx, base_config in enumerate(HOPPER_BASE_CONFIGS):
    for stage_idx, stage_name in enumerate(STAGE_VARIANTS):
        HOPPER_CONFIG_METADATA.append(
            {
                "id": config_id,
                "name": f"{base_config['name']}_{stage_name}",
                "tile": base_config["tile"],
                "cluster": base_config["cluster"],
                "stages": stage_name,
                "base_idx": base_idx,
                "stage_idx": stage_idx,
            }
        )
        config_id += 1


def benchmark_config(
    config_id: int,
    a: torch.Tensor,
    b: torch.Tensor,
    dtype: str,
    warmup: int = 10,
    iterations: int = 100,
    flush_cache: bool = True,
    adaptive_iterations: bool = False,
    target_time_ms: float = 1000.0,
) -> Optional[Tuple[float, float, float]]:
    """Benchmark a single CUTLASS Hopper configuration with robust measurement practices.

    Implements best practices from Triton benchmarking:
    - L2 cache flushing between iterations to eliminate cache artifacts
    - Adaptive iteration counts based on kernel runtime
    - Proper CUDA synchronization
    - Statistical outlier removal and median-based comparison

    Args:
        config_id: Configuration ID (0-11)
        a: Input matrix A
        b: Input matrix B
        dtype: Data type ("float16" or "bfloat16")
        warmup: Number of warmup iterations (adaptive if adaptive_iterations=True)
        iterations: Number of benchmark iterations (adaptive if adaptive_iterations=True)
        flush_cache: Whether to flush L2 cache between iterations
        adaptive_iterations: Scale iterations based on kernel runtime
        target_time_ms: Target total benchmark time for adaptive iterations (ms)

    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms) or None if config failed
    """
    # Create output tensor (same dtype as input)
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=a.dtype)

    # Select the appropriate kernel function (only bfloat16 is supported)
    kernel_fn = cuda_kernels.sgemm_cutlass_hopper_autotune_bf16  # type: ignore

    try:
        # Estimate runtime with a few iterations (like Triton's approach)
        if adaptive_iterations:
            estimate_iters = 5
            torch.cuda.synchronize()
            start_estimate = torch.cuda.Event(enable_timing=True)
            end_estimate = torch.cuda.Event(enable_timing=True)

            start_estimate.record()
            for _ in range(estimate_iters):
                kernel_fn(config_id, a, b, c)
            end_estimate.record()
            torch.cuda.synchronize()

            estimate_ms = start_estimate.elapsed_time(end_estimate) / estimate_iters

            # Calculate adaptive iteration counts
            # Target: total benchmark time ~1 second
            iterations = max(10, min(1000, int(target_time_ms / estimate_ms)))
            warmup = max(5, min(50, iterations // 10))

        # Warmup to reach thermal/clock steady-state
        for _ in range(warmup):
            if flush_cache:
                flush_l2_cache()
            kernel_fn(config_id, a, b, c)

        torch.cuda.synchronize()

        # Benchmark with cache flushing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
            # Flush L2 cache before each iteration to eliminate cache state artifacts
            if flush_cache:
                flush_l2_cache()

            start_events[i].record()
            kernel_fn(config_id, a, b, c)
            end_events[i].record()

        torch.cuda.synchronize()

        # Calculate times
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # Trim outliers (top and bottom 10%)
        times_ms_sorted = sorted(times_ms)
        trim_count = max(1, iterations // 10)
        times_ms_trimmed = times_ms_sorted[trim_count:-trim_count]

        # Use median instead of average for more robust comparison
        median_time_ms = float(np.median(times_ms_trimmed))
        min_time_ms = min(times_ms_trimmed)
        max_time_ms = max(times_ms_trimmed)

        return median_time_ms, min_time_ms, max_time_ms

    except RuntimeError as e:
        logger.warning(f"Config {config_id} failed: {e}")
        return None


def calculate_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Calculate TFLOPS from matrix dimensions and time."""
    flops = 2 * M * N * K
    return (flops / (time_ms * 1e-3)) * 1e-12


def benchmark_pytorch(
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int = 10,
    iterations: int = 100,
    flush_cache: bool = True,
    adaptive_iterations: bool = False,
    target_time_ms: float = 1000.0,
) -> Optional[Tuple[float, float, float]]:
    """Benchmark PyTorch matmul as baseline with robust measurement practices.

    Implements best practices from Triton benchmarking:
    - L2 cache flushing between iterations to eliminate cache artifacts
    - Adaptive iteration counts based on kernel runtime
    - Proper CUDA synchronization
    - Statistical outlier removal and median-based comparison

    Args:
        a: Input matrix A
        b: Input matrix B
        warmup: Number of warmup iterations (adaptive if adaptive_iterations=True)
        iterations: Number of benchmark iterations (adaptive if adaptive_iterations=True)
        flush_cache: Whether to flush L2 cache between iterations
        adaptive_iterations: Scale iterations based on kernel runtime
        target_time_ms: Target total benchmark time for adaptive iterations (ms)

    Returns:
        Tuple of (median_time_ms, min_time_ms, max_time_ms) or None if failed
    """
    try:
        # Estimate runtime with a few iterations (like Triton's approach)
        if adaptive_iterations:
            estimate_iters = 5
            torch.cuda.synchronize()
            start_estimate = torch.cuda.Event(enable_timing=True)
            end_estimate = torch.cuda.Event(enable_timing=True)

            start_estimate.record()
            for _ in range(estimate_iters):
                _ = torch.matmul(a, b)
            end_estimate.record()
            torch.cuda.synchronize()

            estimate_ms = start_estimate.elapsed_time(end_estimate) / estimate_iters

            # Calculate adaptive iteration counts
            iterations = max(10, min(1000, int(target_time_ms / estimate_ms)))
            warmup = max(5, min(50, iterations // 10))

        # Warmup to reach thermal/clock steady-state
        for _ in range(warmup):
            if flush_cache:
                flush_l2_cache()
            _ = torch.matmul(a, b)

        torch.cuda.synchronize()

        # Benchmark with cache flushing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
            # Flush L2 cache before each iteration
            if flush_cache:
                flush_l2_cache()

            start_events[i].record()
            _ = torch.matmul(a, b)
            end_events[i].record()

        torch.cuda.synchronize()

        # Calculate times
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # Trim outliers (top and bottom 10%)
        times_ms_sorted = sorted(times_ms)
        trim_count = max(1, iterations // 10)
        times_ms_trimmed = times_ms_sorted[trim_count:-trim_count]

        # Use median instead of average for more robust comparison
        median_time_ms = float(np.median(times_ms_trimmed))
        min_time_ms = min(times_ms_trimmed)
        max_time_ms = max(times_ms_trimmed)

        return median_time_ms, min_time_ms, max_time_ms

    except RuntimeError as e:
        logger.warning(f"PyTorch benchmark failed: {e}")
        return None


def autotune_size(
    size: int,
    dtype: str,
    num_configs: int = 12,
    flush_cache: bool = True,
    adaptive_iterations: bool = False,
    target_time_ms: float = 1000.0,
) -> Dict:
    """Autotune all Hopper configurations for a given matrix size.

    Args:
        size: Matrix size (M=N=K=size)
        dtype: Data type ("float16" or "bfloat16")
        num_configs: Number of configurations to test
        flush_cache: Whether to flush L2 cache between iterations
        adaptive_iterations: Whether to use adaptive iteration counts
        target_time_ms: Target total benchmark time for adaptive iterations

    Returns:
        Dictionary with autotuning results
    """
    M = N = K = size

    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 Autotuning for size {size}x{size}x{size} ({dtype.upper()})")
    if flush_cache:
        logger.info(f"   L2 cache flushing: ✅ Enabled")
    else:
        logger.info(f"   L2 cache flushing: ❌ Disabled")
    if adaptive_iterations:
        logger.info(f"   Adaptive iterations: ✅ Enabled (target: {target_time_ms}ms)")
    else:
        logger.info(
            f"   Adaptive iterations: ❌ Disabled (fixed: warmup=10, iters=100)"
        )
    logger.info(f"{'='*80}")

    # Map dtype string to torch dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    # Create input matrices
    try:
        a = torch.randn((M, K), device="cuda", dtype=torch_dtype)
        b = torch.randn((K, N), device="cuda", dtype=torch_dtype)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"❌ Out of memory for size {size}!")
            return {"size": size, "error": "OOM"}
        raise

    # Benchmark PyTorch baseline
    logger.info(f"\n📊 Benchmarking PyTorch baseline")
    pytorch_result = benchmark_pytorch(
        a,
        b,
        flush_cache=flush_cache,
        adaptive_iterations=adaptive_iterations,
        target_time_ms=target_time_ms,
    )
    if pytorch_result:
        pytorch_time, _, _ = pytorch_result
        pytorch_tflops = calculate_tflops(M, N, K, pytorch_time)
        logger.success(f"   ✅ {pytorch_time:.4f} ms ({pytorch_tflops:.2f} TFLOPS)")
    else:
        pytorch_time = None
        pytorch_tflops = None
        logger.warning("   ❌ PyTorch benchmark failed")

    results = []
    best_config = None
    best_time = float("inf")

    # Test all configurations
    for config_id in range(num_configs):
        config_meta = HOPPER_CONFIG_METADATA[config_id]
        logger.info(f"\n📊 Testing Config {config_id}: {config_meta['name']}")
        logger.info(
            f"   Tile: {config_meta['tile']}, Cluster: {config_meta['cluster']}"
        )

        result = benchmark_config(
            config_id,
            a,
            b,
            dtype,
            flush_cache=flush_cache,
            adaptive_iterations=adaptive_iterations,
            target_time_ms=target_time_ms,
        )

        if result is None:
            logger.warning(f"   ❌ Failed")
            results.append(
                {
                    "config_id": config_id,
                    "config_name": config_meta["name"],
                    "status": "failed",
                    "median_time_ms": None,
                    "tflops": None,
                }
            )
            continue

        median_ms, min_ms, max_ms = result
        tflops = calculate_tflops(M, N, K, median_ms)

        results.append(
            {
                "config_id": config_id,
                "config_name": config_meta["name"],
                "status": "success",
                "median_time_ms": median_ms,
                "min_time_ms": min_ms,
                "max_time_ms": max_ms,
                "tflops": tflops,
            }
        )

        logger.success(f"   ✅ {median_ms:.4f} ms ({tflops:.2f} TFLOPS)")

        if median_ms < best_time:
            best_time = median_ms
            best_config = config_id

    # Clean up
    del a, b
    torch.cuda.empty_cache()

    if best_config is not None:
        best_result = results[best_config]
        logger.info(
            f"\n🏆 Best configuration for size {size}: Config {best_config} ({HOPPER_CONFIG_METADATA[best_config]['name']})"
        )
        logger.success(f"   ⏱️  Time: {best_result['median_time_ms']:.4f} ms")
        logger.success(f"   💪 Performance: {best_result['tflops']:.2f} TFLOPS")

        if pytorch_time is not None:
            speedup = pytorch_time / best_result["median_time_ms"]
            if speedup > 1:
                logger.info(f"   🚀 Speedup vs PyTorch: {speedup:.2f}x faster")
            else:
                logger.info(f"   🐢 Speedup vs PyTorch: {1/speedup:.2f}x slower")

    return {
        "size": size,
        "dtype": dtype,
        "best_config": best_config,
        "best_time_ms": best_time if best_config is not None else None,
        "best_tflops": (
            results[best_config]["tflops"] if best_config is not None else None
        ),
        "pytorch_time_ms": pytorch_time,
        "pytorch_tflops": pytorch_tflops,
        "speedup_vs_pytorch": (
            pytorch_time / best_time
            if (pytorch_time and best_config is not None)
            else None
        ),
        "all_results": results,
    }


def save_cache(results: List[Dict], dtype: str, cache_dir: Path):
    """Save autotuning results to cache."""
    cache_file = cache_dir / f"autotune_hopper_cache_{dtype}.json"

    # Convert to serializable format
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "results": results,
    }

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

    logger.success(f"💾 Saved cache to {cache_file}")

    # Also save a CSV for easy viewing
    csv_data = []
    for r in results:
        if "error" not in r:
            csv_data.append(
                {
                    "size": r["size"],
                    "best_config": r["best_config"],
                    "best_config_name": (
                        HOPPER_CONFIG_METADATA[r["best_config"]]["name"]
                        if r["best_config"] is not None
                        else "N/A"
                    ),
                    "best_time_ms": r["best_time_ms"],
                    "best_tflops": r["best_tflops"],
                    "pytorch_time_ms": r.get("pytorch_time_ms"),
                    "pytorch_tflops": r.get("pytorch_tflops"),
                    "speedup_vs_pytorch": r.get("speedup_vs_pytorch"),
                }
            )

    df = pd.DataFrame(csv_data)
    csv_file = cache_dir / f"autotune_hopper_results_{dtype}.csv"
    df.to_csv(csv_file, index=False)
    logger.success(f"📊 Saved summary to {csv_file}")


def load_cache(dtype: str, cache_dir: Path) -> Optional[Dict]:
    """Load autotuning results from cache."""
    cache_file = cache_dir / f"autotune_hopper_cache_{dtype}.json"

    if not cache_file.exists():
        logger.warning(f"⚠️  Cache file not found: {cache_file}")
        return None

    with open(cache_file, "r") as f:
        cache_data = json.load(f)

    logger.success(f"✅ Loaded cache from {cache_file}")
    logger.info(f"   Generated: {cache_data['timestamp']}")
    logger.info(f"   GPU: {cache_data['gpu']}")
    logger.info(f"   Sizes: {len(cache_data['results'])}")

    return cache_data


def create_visualization(results: List[Dict], dtype: str, output_dir: Path):
    """Create visualization of Hopper autotuning results."""
    logger.info("📊 Creating visualization...")

    # Extract data for plotting
    sizes = []
    best_tflops = []
    pytorch_tflops = []
    speedups = []
    best_config_ids = []
    best_config_names = []

    # For heatmap: collect speedup data for all configs
    num_configs = len(HOPPER_CONFIG_METADATA)
    heatmap_data = []  # List of lists: [size][config] = speedup
    heatmap_sizes = []

    for r in results:
        if "error" not in r and r["best_config"] is not None:
            sizes.append(r["size"])
            best_tflops.append(r["best_tflops"])
            pytorch_tflops.append(r.get("pytorch_tflops"))
            speedups.append(r.get("speedup_vs_pytorch"))
            best_config_ids.append(r["best_config"])
            best_config_names.append(HOPPER_CONFIG_METADATA[r["best_config"]]["name"])

            # Build heatmap row for this size
            heatmap_sizes.append(r["size"])
            heatmap_row = []
            pytorch_time = r.get("pytorch_time_ms")

            for config_id in range(num_configs):
                config_result = r["all_results"][config_id]
                if config_result["status"] == "success" and pytorch_time:
                    # Calculate speedup for this config
                    speedup = pytorch_time / config_result["median_time_ms"]
                    heatmap_row.append(speedup)
                else:
                    heatmap_row.append(None)  # Failed config

            heatmap_data.append(heatmap_row)

    # Create 2-row layout: Row 1 has 2 plots, Row 2 has heatmap (full width)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Performance Comparison ({dtype.upper()})",
            f"Speedup vs PyTorch ({dtype.upper()})",
            f"Speedup Heatmap: All Hopper Configs vs PyTorch ({dtype.upper()})",
            "",  # Empty for heatmap colspan
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap", "colspan": 2}, None],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        row_heights=[0.35, 0.65],  # More space for heatmap
    )

    # Plot 1 (top-left): TFLOPS comparison
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=best_tflops,
            mode="lines+markers",
            name="Best Hopper Config",
            line=dict(color="#667eea", width=3),
            marker=dict(size=10),
            text=best_config_names,
            hovertemplate="<b>Size:</b> %{x}<br><b>TFLOPS:</b> %{y:.2f}<br><b>Config:</b> %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add PyTorch baseline to plot 1
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=pytorch_tflops,
            mode="lines+markers",
            name="PyTorch",
            line=dict(color="#FF6692", width=2, dash="dash"),
            marker=dict(size=8),
            hovertemplate="<b>Size:</b> %{x}<br><b>TFLOPS:</b> %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Plot 2 (top-right): Speedup
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=speedups,
            mode="lines+markers",
            name="Hopper Speedup",
            line=dict(color="#00CC96", width=3),
            marker=dict(size=10),
            text=best_config_names,
            hovertemplate="<b>Size:</b> %{x}<br><b>Speedup:</b> %{y:.2f}x<br><b>Config:</b> %{text}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Add baseline at 1.0x to plot 2
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="gray",
        annotation_text="PyTorch Baseline (1.0x)",
        row=1,  # type: ignore
        col=2,  # type: ignore
    )

    # Plot 3 (row 2, full width): Heatmap
    # Transpose heatmap_data for better visualization (configs on y-axis, sizes on x-axis)
    heatmap_data_transposed = list(map(list, zip(*heatmap_data)))

    # Filter out configs that have no data (all None values)
    filtered_configs = []
    filtered_data = []
    filtered_labels = []
    config_id_mapping = {}  # Map original config_id to filtered index

    for config_id in range(num_configs):
        row_data = heatmap_data_transposed[config_id]
        # Check if this config has at least one non-None value
        if any(val is not None for val in row_data):
            config_id_mapping[config_id] = len(filtered_configs)
            filtered_configs.append(config_id)
            filtered_data.append(row_data)
            # Include tile and cluster in the label
            meta = HOPPER_CONFIG_METADATA[config_id]
            tile_str = f"{meta['tile'][0]}x{meta['tile'][1]}x{meta['tile'][2]}"
            cluster_str = (
                f"{meta['cluster'][0]}x{meta['cluster'][1]}x{meta['cluster'][2]}"
            )
            filtered_labels.append(f"{config_id}: T{tile_str}_C{cluster_str}")

    # Create custom text with bold formatting for best configs
    text_matrix = []
    for filtered_idx, config_id in enumerate(filtered_configs):
        row_text = []
        for size_idx, size in enumerate(heatmap_sizes):
            speedup_val = filtered_data[filtered_idx][size_idx]
            if speedup_val is not None:
                # Check if this is the best config for this size
                if config_id == best_config_ids[size_idx]:
                    row_text.append(f"<b>{speedup_val:.2f}x</b>")
                else:
                    row_text.append(f"{speedup_val:.2f}x")
            else:
                row_text.append("")
        text_matrix.append(row_text)

    # Create customdata for hover template with detailed config info
    customdata = []
    for filtered_idx, config_id in enumerate(filtered_configs):
        row_customdata = []
        meta = HOPPER_CONFIG_METADATA[config_id]
        for size_idx, size in enumerate(heatmap_sizes):
            row_customdata.append(
                [
                    meta["tile"],
                    meta["cluster"],
                    config_id,
                ]
            )
        customdata.append(row_customdata)

    fig.add_trace(
        go.Heatmap(
            z=filtered_data,
            x=[str(s) for s in heatmap_sizes],
            y=filtered_labels,
            text=text_matrix,
            customdata=customdata,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale="RdYlGn",
            zmid=1.0,  # Center colorscale at 1.0x (PyTorch baseline)
            colorbar=dict(
                title="Speedup vs<br>PyTorch",
                x=1.02,  # Position colorbar next to heatmap
                len=0.6,  # Longer to match heatmap row height
                y=0.25,  # Align with heatmap position
                yanchor="middle",
            ),
            hovertemplate="<b>Size:</b> %{x}<br>"
            + "<b>Config ID:</b> %{customdata[2]}<br>"
            + "<b>Tile Shape:</b> %{customdata[0]}<br>"
            + "<b>Cluster Shape:</b> %{customdata[1]}<br>"
            + "<b>Speedup:</b> %{z:.2f}x<extra></extra>",
            showscale=True,
        ),
        row=2,
        col=1,
    )

    # Add bold borders around best config cells
    shapes = []
    size_to_idx = {size: idx for idx, size in enumerate(heatmap_sizes)}

    for size_idx, size in enumerate(sizes):
        best_config_id = best_config_ids[size_idx]

        # Map original config_id to filtered index
        if best_config_id in config_id_mapping:
            filtered_idx = config_id_mapping[best_config_id]

            # Get the actual heatmap column index for this size
            heatmap_col_idx = size_to_idx.get(size, size_idx)

            # Heatmap uses categorical x-axis, coordinates are 0-indexed for both axes
            shapes.append(
                dict(
                    type="rect",
                    xref="x3",
                    yref="y3",  # x3/y3 for row=2, col=1 (3rd subplot)
                    x0=heatmap_col_idx - 0.45,
                    y0=filtered_idx - 0.45,
                    x1=heatmap_col_idx + 0.45,
                    y1=filtered_idx + 0.45,
                    line=dict(color="black", width=4),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                )
            )

    # Update axes
    # Row 1, Col 1: TFLOPS comparison
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="TFLOPS", row=1, col=1)

    # Row 1, Col 2: Speedup
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Speedup (×)", row=1, col=2)

    # Row 2: Heatmap
    fig.update_xaxes(
        title_text="Matrix Size (M=N=K)",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        layer="below traces",
    )
    fig.update_yaxes(
        title_text="Configuration",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        layer="below traces",
    )

    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"<b>CUTLASS Hopper Autotuning Results ({dtype.upper()})</b>",
        title_x=0.5,
        title_font=dict(size=20),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
        shapes=shapes,  # Add the border shapes
    )

    # Save
    html_file = output_dir / f"autotune_hopper_results_{dtype}.html"
    fig.write_html(str(html_file))
    logger.success(f"✅ Saved visualization to {html_file}")


@click.command()
@click.option(
    "--dtype",
    "-d",
    type=click.Choice(["bfloat16"], case_sensitive=False),
    required=True,
    help="Data type for inputs (only bfloat16 is supported)",
)
@click.option(
    "--sizes",
    "-s",
    multiple=True,
    type=int,
    help="Matrix sizes to test. If not specified, uses power-of-2 from 64 to 8192",
)
@click.option(
    "--load-cache",
    "load_cache_flag",
    is_flag=True,
    help="Load results from cache instead of running autotuning",
)
@click.option(
    "--no-cache-flush",
    is_flag=True,
    help="Disable L2 cache flushing between iterations (faster but less accurate)",
)
@click.option(
    "--adaptive-iters",
    is_flag=True,
    help="Enable adaptive iteration counts based on kernel runtime (default: disabled, uses fixed warmup=10, iterations=100)",
)
@click.option(
    "--target-time",
    type=float,
    default=1000.0,
    help="Target total benchmark time in milliseconds for adaptive iterations (default: 1000ms)",
)
def main(dtype, sizes, load_cache_flag, no_cache_flush, adaptive_iters, target_time):
    """Autotune CUTLASS Hopper GEMM kernels to find optimal configurations.

    Implements robust benchmarking practices:
    - L2 cache flushing to eliminate cache artifacts (enabled by default)
    - Fixed iteration counts (warmup=10, iterations=100) by default
    - Median-based statistical comparison with outlier removal
    - Proper CUDA synchronization and event timing

    Examples:
        # Autotune BF16 for all power-of-2 sizes (with default settings)
        python autotune_cutlass_hopper.py -d bfloat16

        # Autotune specific sizes
        python autotune_cutlass_hopper.py -d bfloat16 -s 128 -s 256 -s 512

        # Disable cache flushing for faster benchmarking
        python autotune_cutlass_hopper.py -d bfloat16 --no-cache-flush

        # Enable adaptive iterations with custom target time (2 seconds per config)
        python autotune_cutlass_hopper.py -d bfloat16 --adaptive-iters --target-time 2000

        # Load cached results
        python autotune_cutlass_hopper.py -d bfloat16 --load-cache
    """
    # Create output directory
    output_dir = Path(__file__).parent / "autotune_results"
    output_dir.mkdir(exist_ok=True)

    # Load cache if requested
    if load_cache_flag:
        cache_data = load_cache(dtype, output_dir)
        if cache_data is None:
            logger.error(
                "❌ Cache not found. Run autotuning first without --load-cache"
            )
            return

        create_visualization(cache_data["results"], dtype, output_dir)
        return

    # Determine sizes to test
    if not sizes:
        # Power of 2 from 64 to 8192
        test_sizes = [2**i for i in range(6, 14)]  # 64, 128, 256, ..., 8192
    else:
        test_sizes = sorted(sizes)

    # Convert CLI flags to function parameters
    flush_cache = not no_cache_flush
    adaptive_iterations = adaptive_iters

    logger.info(f"🎯 Autotuning CUTLASS Hopper kernels for {dtype.upper()}")
    logger.info(f"📏 Sizes to test: {test_sizes}")
    logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"⚙️  Benchmarking settings:")
    logger.info(
        f"   - L2 cache flushing: {'✅ Enabled' if flush_cache else '❌ Disabled'}"
    )
    logger.info(
        f"   - Adaptive iterations: {'✅ Enabled' if adaptive_iterations else '❌ Disabled'}"
    )
    if adaptive_iterations:
        logger.info(f"   - Target benchmark time: {target_time}ms")
    logger.info("")

    # Check if Hopper kernels are available
    if not hasattr(cuda_kernels, "get_num_cutlass_hopper_configs"):
        logger.error("❌ Hopper kernels are not available on this GPU!")
        logger.error(
            "   This script requires a GPU with compute capability SM90+ (H100, H200, etc.)"
        )
        logger.error(f"   Your GPU: {torch.cuda.get_device_name(0)}")
        logger.error(
            "   Hopper-specific features (TMA, warp specialization) are not supported."
        )
        logger.error(
            "\n💡 Tip: Use autotune_cutlass.py for non-Hopper GPUs (Ampere, Ada, etc.)"
        )
        return

    # Get number of configs
    num_configs = cuda_kernels.get_num_cutlass_hopper_configs()  # type: ignore
    logger.info(f"🔧 Number of Hopper configurations: {num_configs}\n")

    # Run autotuning for each size
    all_results = []
    for size in test_sizes:
        result = autotune_size(
            size,
            dtype,
            num_configs,
            flush_cache=flush_cache,
            adaptive_iterations=adaptive_iterations,
            target_time_ms=target_time,
        )
        all_results.append(result)

    # Save results
    save_cache(all_results, dtype, output_dir)

    # Create visualization
    create_visualization(all_results, dtype, output_dir)

    logger.success(f"\n✨ Autotuning complete!")
    logger.info(f"📂 Results saved to: {output_dir}")

    # Open visualization in browser
    html_file = output_dir / f"autotune_hopper_results_{dtype}.html"
    logger.info(f"🌐 Opening visualization in browser: {html_file}")
    webbrowser.open(f"file://{html_file.absolute()}")


if __name__ == "__main__":
    main()
