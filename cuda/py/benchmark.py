"""Benchmark CUDA GEMM kernels against PyTorch with visualization.

Usage:
    # Run all kernels (default)
    python benchmark.py

    # Run specific kernels
    python benchmark.py -k pytorch -k naive
    python benchmark.py --kernels naive --kernels global_mem_coalesce

    # Run only one kernel
    python benchmark.py -k naive

Available kernels:
    - pytorch: PyTorch baseline implementation
    - naive: Naive CUDA GEMM kernel
    - global_mem_coalesce: CUDA GEMM with global memory coalescing
"""

import os
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import webbrowser

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import click


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_cuda_code(cuda_file: str, header_file: str) -> Tuple[str, str]:
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
    header_file = file_dir / "gemm_kernels.cuh"

    logger.info("📂 Loading CUDA sources:")
    logger.info(f"   • Naive: {naive_cu}")
    logger.info(f"   • Coalesced: {coalesce_cu}")
    logger.info(f"   • Header: {header_file}")

    # Read all source files
    naive_code, _ = get_cuda_code(str(naive_cu), str(header_file))
    coalesce_code, header_code = get_cuda_code(str(coalesce_cu), str(header_file))

    # Combine CUDA sources
    combined_cuda_code = naive_code + "\n" + coalesce_code

    # Create build directory
    build_dir = file_dir / "build" / "gemm_extension"
    build_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🔨 Build directory: {build_dir}")

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

    logger.success("✅ CUDA extension loaded successfully!")
    return extension


# Load CUDA kernels
logger.info("🚀 Loading CUDA kernels...")
cuda_kernels = create_cuda_extension()


def cuda_naive_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for naive CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_naive(a, b, c, 1.0, 0.0)  # type: ignore
    return c


def cuda_coalesced_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for coalesced global memory CUDA GEMM kernel."""
    # Create output tensor on CUDA device
    c = torch.zeros((a.size(0), b.size(1)), device="cuda", dtype=torch.float32)
    cuda_kernels.sgemm_global_mem_coalesce(a, b, c, 1.0, 0.0)  # type: ignore
    return c


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


def create_visualization(results_df: pd.DataFrame, output_dir: Path):
    """Create interactive Plotly visualizations."""
    logger.info("📊 Creating visualizations...")

    # RTX 4090 theoretical peak performance
    RTX_4090_PEAK_TFLOPS = 82.58  # FP32 TFLOPS
    RTX_4090_PEAK_BANDWIDTH_GBPS = 1008.0  # 1.008 TB/s = 1008 GB/s

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "🚀 TFLOPS Performance vs Matrix Size",
            "💾 Memory Bandwidth vs Matrix Size",
            "⏱️ Execution Time vs Matrix Size",
            "📈 Speedup vs PyTorch (Baseline)",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    kernels = results_df["kernel"].unique()
    colors = {
        "PyTorch": "#636EFA",
        "CUDA Naive": "#EF553B",
        "CUDA Coalesced": "#00CC96",
    }

    # Plot 1: TFLOPS
    for kernel in kernels:
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["tflops"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
            ),
            row=1,
            col=1,
        )

    # Plot 2: Bandwidth
    for kernel in kernels:
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["bandwidth_gbps"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Plot 3: Execution Time
    for kernel in kernels:
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["avg_time_ms"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Plot 4: Speedup
    for kernel in kernels:
        if kernel == "PyTorch":
            continue
        kernel_data = results_df[results_df["kernel"] == kernel]
        fig.add_trace(
            go.Scatter(
                x=kernel_data["size"],
                y=kernel_data["speedup"],
                name=kernel,
                mode="lines+markers",
                line=dict(color=colors.get(kernel, "#000000"), width=2),
                marker=dict(size=8),
                legendgroup=kernel,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # Add baseline line for speedup
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="PyTorch Baseline",
        row=2,  # type: ignore
        col=2,  # type: ignore
    )

    # Add RTX 4090 theoretical peak lines using add_shape for subplots
    # TFLOPS peak line (row 1, col 1)
    fig.add_hline(
        y=RTX_4090_PEAK_TFLOPS,
        line_dash="dot",
        line_color="red",
        line_width=2,
        annotation_text=f"RTX 4090 Peak: {RTX_4090_PEAK_TFLOPS} TFLOPS",
        annotation_position="left",
        row=1,  # type: ignore
        col=1,  # type: ignore
    )

    # Bandwidth peak line (row 1, col 2)
    fig.add_hline(
        y=RTX_4090_PEAK_BANDWIDTH_GBPS,
        line_dash="dot",
        line_color="red",
        line_width=2,
        annotation_text=f"RTX 4090 Peak: {RTX_4090_PEAK_BANDWIDTH_GBPS} GB/s",
        annotation_position="right",
        row=1,  # type: ignore
        col=2,  # type: ignore
    )

    # Update axes
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=1, col=1, type="log")
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=1, col=2, type="log")
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=2, col=1, type="log")
    fig.update_xaxes(title_text="Matrix Size (M=N=K)", row=2, col=2, type="log")

    fig.update_yaxes(title_text="TFLOPS", row=1, col=1)
    fig.update_yaxes(title_text="Bandwidth (GB/s)", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1, type="log")
    fig.update_yaxes(title_text="Speedup (×)", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=900,
        title_text="<b>CUDA GEMM Kernel Benchmarks</b>",
        title_x=0.5,
        title_font=dict(size=20),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )

    # Save interactive HTML
    html_file = output_dir / "benchmark_results.html"
    fig.write_html(str(html_file))
    logger.success(f"✅ Saved interactive visualization to {html_file}")

    return fig


def create_html_report(results_df: pd.DataFrame, output_dir: Path):
    """Create HTML report with results and visualizations."""
    logger.info("📝 Creating HTML report...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA GEMM Benchmark Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 3px solid #e9ecef;
        }}

        .info-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .info-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .info-card p {{
            color: #333;
            font-size: 1.3em;
            font-weight: bold;
        }}

        .chart-container {{
            padding: 30px;
        }}

        .chart-container iframe {{
            width: 100%;
            height: 950px;
            border: none;
            border-radius: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px;
            max-width: calc(100% - 60px);
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}

        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .kernel-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-pytorch {{
            background: #e3f2fd;
            color: #1976d2;
        }}

        .badge-pytorch-jit {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}

        .badge-naive {{
            background: #ffebee;
            color: #c62828;
        }}

        .badge-coalesced {{
            background: #e8f5e9;
            color: #2e7d32;
        }}

        .speedup {{
            font-weight: bold;
        }}

        .speedup.faster {{
            color: #2e7d32;
        }}

        .speedup.slower {{
            color: #c62828;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 CUDA GEMM Kernel Benchmarks</h1>
            <p class="subtitle">Performance Comparison: PyTorch vs Custom CUDA Kernels</p>
        </header>

        <div class="info-grid">
            <div class="info-card">
                <h3>🖥️ GPU</h3>
                <p>{gpu_name}</p>
            </div>
            <div class="info-card">
                <h3>⏰ Timestamp</h3>
                <p>{timestamp}</p>
            </div>
            <div class="info-card">
                <h3>🔢 Test Sizes</h3>
                <p>{len(results_df['size'].unique())} configurations</p>
            </div>
            <div class="info-card">
                <h3>⚡ Kernels Tested</h3>
                <p>{len(results_df['kernel'].unique())} kernels</p>
            </div>
        </div>

        <div class="chart-container">
            <iframe src="benchmark_results.html"></iframe>
        </div>

        <h2 style="padding: 30px 30px 0 30px; color: #333;">📊 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Matrix Size</th>
                    <th>Kernel</th>
                    <th>Avg Time (ms)</th>
                    <th>TFLOPS</th>
                    <th>Bandwidth (GB/s)</th>
                    <th>Speedup</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add table rows
    for _, row in results_df.iterrows():
        if row["kernel"] == "PyTorch":
            kernel_class = "pytorch"
        elif "Naive" in row["kernel"]:
            kernel_class = "naive"
        else:
            kernel_class = "coalesced"
        speedup_class = (
            "faster" if row["speedup"] > 1 else "slower" if row["speedup"] < 1 else ""
        )
        speedup_text = (
            f"{row['speedup']:.2f}×" if row["kernel"] != "PyTorch" else "baseline"
        )

        html_content += f"""
                <tr>
                    <td><strong>{row['size']}×{row['size']}</strong></td>
                    <td><span class="kernel-badge badge-{kernel_class}">{row['kernel']}</span></td>
                    <td>{row['avg_time_ms']:.4f}</td>
                    <td><strong>{row['tflops']:.2f}</strong></td>
                    <td>{row['bandwidth_gbps']:.2f}</td>
                    <td><span class="speedup {speedup_class}">{speedup_text}</span></td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <footer>
            <p>Generated by CUDA GEMM Benchmark Suite • Powered by PyTorch & Plotly</p>
        </footer>
    </div>
</body>
</html>
"""

    # Save HTML report
    report_file = output_dir / "index.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    logger.success(f"✅ Saved HTML report to {report_file}")


def run_benchmarks(kernels_to_run: List[str]):
    """Run benchmarks for specified kernels."""
    # Enable TF32 for PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info("🎯 Running GEMM benchmarks...\n")
    logger.info(f"📋 Kernels selected: {', '.join(kernels_to_run)}\n")

    # Get GPU memory info
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 Total GPU Memory: {gpu_mem_gb:.2f} GB\n")

    # Test sizes - expanded range up to memory limits
    # Each float32 matrix of size NxN takes 4N^2 bytes
    # For GEMM we need 3 matrices (A, B, C) = 12N^2 bytes
    test_sizes = [
        64,
        96,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]

    # Store all results
    all_results = []

    for size in test_sizes:
        M = N = K = size
        logger.info(f"{'='*80}")
        logger.info(f"📐 Matrix dimensions: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")

        # Calculate expected memory usage
        memory_per_matrix_gb = (M * K * 4) / 1e9  # float32 = 4 bytes
        total_memory_gb = 3 * memory_per_matrix_gb  # A, B, C
        logger.info(
            f"💾 Expected memory usage: {total_memory_gb:.2f} GB ({memory_per_matrix_gb:.2f} GB per matrix)"
        )
        logger.info(f"{'='*80}")

        # Try to allocate tensors, catch OOM errors
        try:
            # Generate random inputs
            a = torch.randn((M, K), device="cuda", dtype=torch.float32)
            b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"⚠️  Out of memory! Skipping size {size} and larger sizes."
                )
                logger.warning(
                    f"   Could not allocate {total_memory_gb:.2f} GB for matrices"
                )
                break
            else:
                raise

        # All available kernels
        all_kernels = [
            ("pytorch", "PyTorch", torch_gemm, "🔵"),
            ("naive", "CUDA Naive", cuda_naive_gemm, "🔴"),
            ("global_mem_coalesce", "CUDA Coalesced", cuda_coalesced_gemm, "🟢"),
        ]

        # Filter kernels based on user selection
        kernels = [
            (display_name, kernel_fn, emoji)
            for kernel_id, display_name, kernel_fn, emoji in all_kernels
            if kernel_id in kernels_to_run
        ]

        size_results = {}

        for kernel_name, kernel_fn, emoji in kernels:
            logger.info(f"\n{emoji} Benchmarking {kernel_name}...")
            try:
                avg_ms, min_ms, max_ms = benchmark_kernel(kernel_fn, a, b)
                tflops, bandwidth = calculate_metrics(M, N, K, avg_ms)

                size_results[kernel_name] = {
                    "avg_time_ms": avg_ms,
                    "min_time_ms": min_ms,
                    "max_time_ms": max_ms,
                    "tflops": tflops,
                    "bandwidth_gbps": bandwidth,
                }

                logger.info(
                    f"   ⏱️  Time: {avg_ms:.4f} ms (min: {min_ms:.4f}, max: {max_ms:.4f})"
                )
                logger.success(f"   💪 Performance: {tflops:.2f} TFLOPS")
                logger.success(f"   🌊 Bandwidth: {bandwidth:.2f} GB/s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"   ❌ Out of memory during {kernel_name} benchmark!")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        # Skip comparison if not all kernels completed
        if len(size_results) == 0:
            logger.warning("⚠️  No kernels completed for this size, skipping...\n")
            # Clean up and continue
            del a, b
            torch.cuda.empty_cache()
            continue

        # Calculate speedups
        if "PyTorch" not in size_results:
            logger.warning("⚠️  PyTorch baseline missing, skipping comparison\n")
            # Clean up and continue
            del a, b
            torch.cuda.empty_cache()
            continue

        baseline_time = size_results["PyTorch"]["avg_time_ms"]

        logger.info(f"\n{'─'*80}")
        logger.info("📊 Comparison (baseline: PyTorch)")
        logger.info(f"{'─'*80}")

        # Iterate over kernels that were actually run
        for kernel_name in size_results.keys():
            if kernel_name not in size_results:
                logger.warning(f"{kernel_name:20s}: ❌ Failed (OOM)")
                continue

            result = size_results[kernel_name]
            speedup = baseline_time / result["avg_time_ms"]
            result["speedup"] = speedup

            all_results.append(
                {
                    "size": size,
                    "kernel": kernel_name,
                    "avg_time_ms": result["avg_time_ms"],
                    "tflops": result["tflops"],
                    "bandwidth_gbps": result["bandwidth_gbps"],
                    "speedup": speedup,
                }
            )

            if kernel_name == "PyTorch":
                logger.info(f"{kernel_name:20s}: 1.00× (baseline) 🎯")
            else:
                emoji = "🏆" if speedup > 1 else "🐢"
                logger.info(f"{kernel_name:20s}: {speedup:.2f}× {emoji}")

        logger.info("\n")

        # Clean up GPU memory after each size
        del a, b
        torch.cuda.empty_cache()

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Create output directory
    output_dir = Path(__file__).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)

    # Save results to CSV
    csv_file = output_dir / "results.csv"
    results_df.to_csv(csv_file, index=False)
    logger.success(f"💾 Saved results to {csv_file}")

    # Create visualizations
    create_visualization(results_df, output_dir)

    # Create HTML report
    create_html_report(results_df, output_dir)

    # Open the report in browser
    report_path = output_dir / "index.html"
    logger.success(f"\n🎉 Benchmark complete! Opening {report_path} in browser...")

    # Open in default browser
    webbrowser.open(f"file://{report_path.absolute()}")

    logger.info(f"📂 Results saved to: {output_dir}")
    logger.info(f"   • HTML Report: {report_path}")
    logger.info(f"   • Interactive Charts: {output_dir / 'benchmark_results.html'}")
    logger.info(f"   • CSV Data: {output_dir / 'results.csv'}")


@click.command()
@click.option(
    "--kernels",
    "-k",
    multiple=True,
    type=click.Choice(
        ["pytorch", "naive", "global_mem_coalesce"], case_sensitive=False
    ),
    help="Specify which kernels to benchmark. Can be used multiple times. Choices: pytorch, naive, global_mem_coalesce",
)
def main(kernels):
    """Benchmark CUDA GEMM kernels against PyTorch.

    Examples:
        # Run all kernels (default)
        python benchmark.py

        # Run only PyTorch and naive kernels
        python benchmark.py -k pytorch -k naive

        # Run only coalesced kernel
        python benchmark.py -k global_mem_coalesce
    """
    # If no kernels specified, run all
    if not kernels:
        kernels_to_run = ["pytorch", "naive", "global_mem_coalesce"]
    else:
        kernels_to_run = list(kernels)

    run_benchmarks(kernels_to_run)


if __name__ == "__main__":
    main()
