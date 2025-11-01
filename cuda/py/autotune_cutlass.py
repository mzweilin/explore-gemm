"""Autotune CUTLASS GEMM kernels to find optimal configurations.

This script benchmarks all CUTLASS kernel configurations across different matrix sizes
and finds the best configuration for each size. Results are cached for future use.

Usage:
    # Autotune FP16 kernels for all power-of-2 sizes from 64 to 8192
    python autotune_cutlass.py -d float16

    # Autotune BF16 kernels
    python autotune_cutlass.py -d bfloat16

    # Autotune specific sizes
    python autotune_cutlass.py -d float16 --sizes 128 256 512 1024

    # Load and use cached results
    python autotune_cutlass.py -d float16 --load-cache --size 1024
"""

import os
import json
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import click

# Set CUDA paths BEFORE importing torch
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
os.environ["CUDA_PATH"] = "/usr/local/cuda-12.8"
os.environ["PATH"] = f"/usr/local/cuda-12.8/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"/usr/local/cuda-12.8/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

import torch
import numpy as np
from loguru import logger
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the shared CUDA extension loader
from cuda_extension_loader import create_cuda_extension

# Load CUDA kernels
logger.info("🚀 Loading CUTLASS autotunable kernels...")
cuda_kernels = create_cuda_extension(verbose=True)

# Configuration metadata - describes each config for reporting
CONFIG_METADATA = [
    {"id": 0, "name": "128x256x64_W64x64x64_S3", "block": (128, 256, 64), "warp": (64, 64, 64), "stages": 3},
    {"id": 1, "name": "64x256x32_W32x64x32_S4", "block": (64, 256, 32), "warp": (32, 64, 32), "stages": 4},
    {"id": 2, "name": "128x128x32_W64x64x32_S4", "block": (128, 128, 32), "warp": (64, 64, 32), "stages": 4},
    {"id": 3, "name": "128x64x32_W64x32x32_S4", "block": (128, 64, 32), "warp": (64, 32, 32), "stages": 4},
    {"id": 4, "name": "64x128x32_W32x64x32_S4", "block": (64, 128, 32), "warp": (32, 64, 32), "stages": 4},
    {"id": 5, "name": "128x32x32_W64x32x32_S4", "block": (128, 32, 32), "warp": (64, 32, 32), "stages": 4},
    {"id": 6, "name": "64x32x32_W32x32x32_S5", "block": (64, 32, 32), "warp": (32, 32, 32), "stages": 5},
    {"id": 7, "name": "32x64x32_W32x32x32_S5", "block": (32, 64, 32), "warp": (32, 32, 32), "stages": 5},
    {"id": 8, "name": "128x128x64_W64x64x64_S4", "block": (128, 128, 64), "warp": (64, 64, 64), "stages": 4},
    {"id": 9, "name": "128x64x64_W64x32x64_S4", "block": (128, 64, 64), "warp": (64, 32, 64), "stages": 4},
    {"id": 10, "name": "64x128x64_W32x64x64_S4", "block": (64, 128, 64), "warp": (32, 64, 64), "stages": 4},
    {"id": 11, "name": "256x256x32_W64x64x32_S3", "block": (256, 256, 32), "warp": (64, 64, 32), "stages": 3},
    {"id": 12, "name": "256x128x32_W64x64x32_S3", "block": (256, 128, 32), "warp": (64, 64, 32), "stages": 3},
    {"id": 13, "name": "128x256x32_W64x64x32_S3", "block": (128, 256, 32), "warp": (64, 64, 32), "stages": 3},
    {"id": 14, "name": "64x64x32_W32x32x32_S5", "block": (64, 64, 32), "warp": (32, 32, 32), "stages": 5},
]


def benchmark_config(
    config_id: int,
    a: torch.Tensor,
    b: torch.Tensor,
    dtype: str,
    warmup: int = 5,
    iterations: int = 50,
) -> Optional[Tuple[float, float, float]]:
    """Benchmark a single CUTLASS configuration.

    Args:
        config_id: Configuration ID (0-14)
        a: Input matrix A
        b: Input matrix B
        dtype: Data type ("float16" or "bfloat16")
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Tuple of (avg_time_ms, min_time_ms, max_time_ms) or None if config failed
    """
    # Create output tensor (FP32)
    c = torch.empty((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)

    # Select the appropriate kernel function
    if dtype == "float16":
        kernel_fn = cuda_kernels.sgemm_cutlass_autotune_fp16
    else:  # bfloat16
        kernel_fn = cuda_kernels.sgemm_cutlass_autotune_bf16

    try:
        # Warmup
        for _ in range(warmup):
            kernel_fn(config_id, a, b, c, 1.0, 0.0)

        # Benchmark
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
            start_events[i].record()
            kernel_fn(config_id, a, b, c, 1.0, 0.0)
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
    warmup: int = 5,
    iterations: int = 50,
) -> Optional[Tuple[float, float, float]]:
    """Benchmark PyTorch matmul as baseline.

    Args:
        a: Input matrix A
        b: Input matrix B
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Tuple of (avg_time_ms, min_time_ms, max_time_ms) or None if failed
    """
    try:
        # Warmup
        for _ in range(warmup):
            _ = torch.matmul(a, b)

        # Benchmark
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
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

        avg_time_ms = sum(times_ms_trimmed) / len(times_ms_trimmed)
        min_time_ms = min(times_ms_trimmed)
        max_time_ms = max(times_ms_trimmed)

        return avg_time_ms, min_time_ms, max_time_ms

    except RuntimeError as e:
        logger.warning(f"PyTorch benchmark failed: {e}")
        return None


def autotune_size(
    size: int,
    dtype: str,
    num_configs: int = 15,
) -> Dict:
    """Autotune all configurations for a given matrix size.

    Args:
        size: Matrix size (M=N=K=size)
        dtype: Data type ("float16" or "bfloat16")
        num_configs: Number of configurations to test

    Returns:
        Dictionary with autotuning results
    """
    M = N = K = size

    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 Autotuning for size {size}x{size}x{size} ({dtype.upper()})")
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
    pytorch_result = benchmark_pytorch(a, b)
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
    best_time = float('inf')

    # Test all configurations
    for config_id in range(num_configs):
        config_meta = CONFIG_METADATA[config_id]
        logger.info(f"\n📊 Testing Config {config_id}: {config_meta['name']}")
        logger.info(f"   Block: {config_meta['block']}, Warp: {config_meta['warp']}, Stages: {config_meta['stages']}")

        result = benchmark_config(config_id, a, b, dtype)

        if result is None:
            logger.warning(f"   ❌ Failed")
            results.append({
                "config_id": config_id,
                "config_name": config_meta['name'],
                "status": "failed",
                "avg_time_ms": None,
                "tflops": None,
            })
            continue

        avg_ms, min_ms, max_ms = result
        tflops = calculate_tflops(M, N, K, avg_ms)

        results.append({
            "config_id": config_id,
            "config_name": config_meta['name'],
            "status": "success",
            "avg_time_ms": avg_ms,
            "min_time_ms": min_ms,
            "max_time_ms": max_ms,
            "tflops": tflops,
        })

        logger.success(f"   ✅ {avg_ms:.4f} ms ({tflops:.2f} TFLOPS)")

        if avg_ms < best_time:
            best_time = avg_ms
            best_config = config_id

    # Clean up
    del a, b
    torch.cuda.empty_cache()

    if best_config is not None:
        best_result = results[best_config]
        logger.info(f"\n🏆 Best configuration for size {size}: Config {best_config} ({CONFIG_METADATA[best_config]['name']})")
        logger.success(f"   ⏱️  Time: {best_result['avg_time_ms']:.4f} ms")
        logger.success(f"   💪 Performance: {best_result['tflops']:.2f} TFLOPS")

        if pytorch_time is not None:
            speedup = pytorch_time / best_result['avg_time_ms']
            if speedup > 1:
                logger.info(f"   🚀 Speedup vs PyTorch: {speedup:.2f}x faster")
            else:
                logger.info(f"   🐢 Speedup vs PyTorch: {1/speedup:.2f}x slower")

    return {
        "size": size,
        "dtype": dtype,
        "best_config": best_config,
        "best_time_ms": best_time if best_config is not None else None,
        "best_tflops": results[best_config]['tflops'] if best_config is not None else None,
        "pytorch_time_ms": pytorch_time,
        "pytorch_tflops": pytorch_tflops,
        "speedup_vs_pytorch": pytorch_time / best_time if (pytorch_time and best_config is not None) else None,
        "all_results": results,
    }


def save_cache(results: List[Dict], dtype: str, cache_dir: Path):
    """Save autotuning results to cache."""
    cache_file = cache_dir / f"autotune_cache_{dtype}.json"

    # Convert to serializable format
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "results": results,
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    logger.success(f"💾 Saved cache to {cache_file}")

    # Also save a CSV for easy viewing
    csv_data = []
    for r in results:
        if "error" not in r:
            csv_data.append({
                "size": r["size"],
                "best_config": r["best_config"],
                "best_config_name": CONFIG_METADATA[r["best_config"]]["name"] if r["best_config"] is not None else "N/A",
                "best_time_ms": r["best_time_ms"],
                "best_tflops": r["best_tflops"],
                "pytorch_time_ms": r.get("pytorch_time_ms"),
                "pytorch_tflops": r.get("pytorch_tflops"),
                "speedup_vs_pytorch": r.get("speedup_vs_pytorch"),
            })

    df = pd.DataFrame(csv_data)
    csv_file = cache_dir / f"autotune_results_{dtype}.csv"
    df.to_csv(csv_file, index=False)
    logger.success(f"📊 Saved summary to {csv_file}")


def load_cache(dtype: str, cache_dir: Path) -> Optional[Dict]:
    """Load autotuning results from cache."""
    cache_file = cache_dir / f"autotune_cache_{dtype}.json"

    if not cache_file.exists():
        logger.warning(f"⚠️  Cache file not found: {cache_file}")
        return None

    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    logger.success(f"✅ Loaded cache from {cache_file}")
    logger.info(f"   Generated: {cache_data['timestamp']}")
    logger.info(f"   GPU: {cache_data['gpu']}")
    logger.info(f"   Sizes: {len(cache_data['results'])}")

    return cache_data


def create_visualization(results: List[Dict], dtype: str, output_dir: Path):
    """Create visualization of autotuning results."""
    logger.info("📊 Creating visualization...")

    # Extract data for plotting
    sizes = []
    best_tflops = []
    pytorch_tflops = []
    speedups = []
    best_config_ids = []
    best_config_names = []

    for r in results:
        if "error" not in r and r["best_config"] is not None:
            sizes.append(r["size"])
            best_tflops.append(r["best_tflops"])
            pytorch_tflops.append(r.get("pytorch_tflops"))
            speedups.append(r.get("speedup_vs_pytorch"))
            best_config_ids.append(r["best_config"])
            best_config_names.append(CONFIG_METADATA[r["best_config"]]["name"])

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"Performance Comparison ({dtype.upper()})",
            f"Speedup vs PyTorch ({dtype.upper()})",
            f"Best Configuration ID vs Matrix Size ({dtype.upper()})",
        ),
        vertical_spacing=0.10,
    )

    # Plot 1: TFLOPS comparison
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=best_tflops,
            mode='lines+markers',
            name='Best CUTLASS Config',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10),
            text=best_config_names,
            hovertemplate='<b>Size:</b> %{x}<br><b>TFLOPS:</b> %{y:.2f}<br><b>Config:</b> %{text}<extra></extra>',
        ),
        row=1, col=1,
    )

    # Add PyTorch baseline
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=pytorch_tflops,
            mode='lines+markers',
            name='PyTorch',
            line=dict(color='#FF6692', width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate='<b>Size:</b> %{x}<br><b>TFLOPS:</b> %{y:.2f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # Plot 2: Speedup
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=speedups,
            mode='lines+markers',
            name='Speedup',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=10),
            text=best_config_names,
            hovertemplate='<b>Size:</b> %{x}<br><b>Speedup:</b> %{y:.2f}x<br><b>Config:</b> %{text}<extra></extra>',
        ),
        row=2, col=1,
    )

    # Add baseline at 1.0x
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="gray",
        annotation_text="PyTorch Baseline (1.0x)",
        row=2,
        col=1,
    )

    # Plot 3: Config IDs
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=best_config_ids,
            mode='lines+markers',
            name='Best Config ID',
            line=dict(color='#FFA15A', width=3),
            marker=dict(size=10),
            text=best_config_names,
            hovertemplate='<b>Size:</b> %{x}<br><b>Config ID:</b> %{y}<br><b>Config:</b> %{text}<extra></extra>',
        ),
        row=3, col=1,
    )

    # Update axes
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", type="log", row=3, col=1)
    fig.update_yaxes(title_text="TFLOPS", row=1, col=1)
    fig.update_yaxes(title_text="Speedup (×)", row=2, col=1)
    fig.update_yaxes(title_text="Config ID", row=3, col=1)

    # Update layout
    fig.update_layout(
        height=1200,
        title_text=f"<b>CUTLASS Autotuning Results ({dtype.upper()})</b>",
        title_x=0.5,
        title_font=dict(size=20),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # Save
    html_file = output_dir / f"autotune_results_{dtype}.html"
    fig.write_html(str(html_file))
    logger.success(f"✅ Saved visualization to {html_file}")


@click.command()
@click.option(
    "--dtype",
    "-d",
    type=click.Choice(["float16", "bfloat16"], case_sensitive=False),
    required=True,
    help="Data type for inputs (float16 or bfloat16)",
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
def main(dtype, sizes, load_cache_flag):
    """Autotune CUTLASS GEMM kernels to find optimal configurations.

    Examples:
        # Autotune FP16 for all power-of-2 sizes
        python autotune_cutlass.py -d float16

        # Autotune specific sizes
        python autotune_cutlass.py -d float16 -s 128 -s 256 -s 512

        # Load cached results
        python autotune_cutlass.py -d float16 --load-cache
    """
    # Create output directory
    output_dir = Path(__file__).parent / "autotune_results"
    output_dir.mkdir(exist_ok=True)

    # Load cache if requested
    if load_cache_flag:
        cache_data = load_cache(dtype, output_dir)
        if cache_data is None:
            logger.error("❌ Cache not found. Run autotuning first without --load-cache")
            return

        create_visualization(cache_data["results"], dtype, output_dir)
        return

    # Determine sizes to test
    if not sizes:
        # Power of 2 from 64 to 8192
        test_sizes = [2**i for i in range(6, 14)]  # 64, 128, 256, ..., 8192
    else:
        test_sizes = sorted(sizes)

    logger.info(f"🎯 Autotuning CUTLASS kernels for {dtype.upper()}")
    logger.info(f"📏 Sizes to test: {test_sizes}")
    logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}\n")

    # Get number of configs
    num_configs = cuda_kernels.get_num_cutlass_configs()
    logger.info(f"🔧 Number of configurations: {num_configs}\n")

    # Run autotuning for each size
    all_results = []
    for size in test_sizes:
        result = autotune_size(size, dtype, num_configs)
        all_results.append(result)

    # Save results
    save_cache(all_results, dtype, output_dir)

    # Create visualization
    create_visualization(all_results, dtype, output_dir)

    logger.success(f"\n✨ Autotuning complete!")
    logger.info(f"📂 Results saved to: {output_dir}")

    # Open visualization in browser
    html_file = output_dir / f"autotune_results_{dtype}.html"
    logger.info(f"🌐 Opening visualization in browser: {html_file}")
    webbrowser.open(f"file://{html_file.absolute()}")


if __name__ == "__main__":
    main()
