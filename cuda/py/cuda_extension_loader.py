"""Shared CUDA extension loader for GEMM kernels.

This module provides a centralized way to build and load CUDA GEMM kernels
for both benchmark.py and kernel_runner.py, ensuring consistent build configuration.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline
from loguru import logger


def get_cuda_code(cuda_file: str, header_file: str, utils_header_file: str) -> Tuple[str, str, str]:
    """Load CUDA source and header files, removing #include and #pragma once directives.

    Args:
        cuda_file: Path to the CUDA source file
        header_file: Path to the main header file (gemm_kernels.cuh)
        utils_header_file: Path to utilities header file (utils.cuh)

    Returns:
        Tuple of (cuda_code, header_code, utils_code)
    """
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

    with open(utils_header_file) as f:
        utils_code = "".join(
            [
                line
                for line in f.readlines()
                if not line.startswith("#include")
                and not line.startswith("#pragma once")
            ]
        )

    return cuda_code, header_code, utils_code


def create_cuda_extension(verbose: bool = True):
    """Create PyTorch extension for CUDA GEMM kernels.

    This function handles all the complexity of building CUDA kernels with
    proper support for Tensor Cores (FP16/BF16 via WMMA API).

    Args:
        verbose: Whether to show verbose build output (default: True)

    Returns:
        The loaded CUDA extension module
    """
    file_dir = Path(__file__).parent.parent

    # Load all CUDA source files
    naive_cu = file_dir / "01_naive.cu"
    coalesce_cu = file_dir / "02_kernel_global_mem_coalesce.cu"
    shared_mem_cu = file_dir / "03_kernel_shared_mem.cu"
    blocktiling_1d_cu = file_dir / "04_kernel_blocktiling_1d.cu"
    blocktiling_2d_cu = file_dir / "05_kernel_blocktiling_2d.cu"
    vectorize_cu = file_dir / "06_kernel_vectorize.cu"
    warptiling_cu = file_dir / "07_kernel_warptiling.cu"
    warptiling_multidtype_cu = file_dir / "08_kernel_warptiling_all_dtypes.cu"
    tensorcore_cu = file_dir / "09_kernel_tensorcore.cu"
    tensorcore_double_buffered_cu = file_dir / "10_kernel_tensorcore_double_buffered.cu"
    cutlass_cu = file_dir / "11_kernel_cutlass.cu"
    header_file = file_dir / "gemm_kernels.cuh"
    utils_header_file = file_dir / "utils.cuh"

    if verbose:
        logger.info("📂 Loading CUDA sources:")
        logger.info(f"   • Naive: {naive_cu}")
        logger.info(f"   • Coalesced: {coalesce_cu}")
        logger.info(f"   • Shared Memory: {shared_mem_cu}")
        logger.info(f"   • 1D Block Tiling: {blocktiling_1d_cu}")
        logger.info(f"   • 2D Block Tiling: {blocktiling_2d_cu}")
        logger.info(f"   • Vectorize: {vectorize_cu}")
        logger.info(f"   • Warptiling (FP32): {warptiling_cu}")
        logger.info(f"   • Warptiling (Multi-Dtype): {warptiling_multidtype_cu}")
        logger.info(f"   • Tensor Core: {tensorcore_cu}")
        logger.info(
            f"   • Tensor Core Double Buffered: {tensorcore_double_buffered_cu}"
        )
        logger.info(f"   • CUTLASS: {cutlass_cu}")
        logger.info(f"   • Header: {header_file}")
        logger.info(f"   • Utils Header: {utils_header_file}")

    # Read all source files
    naive_code, _, _ = get_cuda_code(str(naive_cu), str(header_file), str(utils_header_file))
    coalesce_code, _, _ = get_cuda_code(str(coalesce_cu), str(header_file), str(utils_header_file))
    shared_mem_code, _, _ = get_cuda_code(str(shared_mem_cu), str(header_file), str(utils_header_file))
    blocktiling_1d_code, _, _ = get_cuda_code(str(blocktiling_1d_cu), str(header_file), str(utils_header_file))
    blocktiling_2d_code, _, _ = get_cuda_code(str(blocktiling_2d_cu), str(header_file), str(utils_header_file))
    vectorize_code, _, _ = get_cuda_code(str(vectorize_cu), str(header_file), str(utils_header_file))
    warptiling_code, _, _ = get_cuda_code(str(warptiling_cu), str(header_file), str(utils_header_file))
    warptiling_multidtype_code, _, _ = get_cuda_code(
        str(warptiling_multidtype_cu), str(header_file), str(utils_header_file)
    )
    tensorcore_code, _, _ = get_cuda_code(str(tensorcore_cu), str(header_file), str(utils_header_file))
    tensorcore_double_buffered_code, _, _ = get_cuda_code(
        str(tensorcore_double_buffered_cu), str(header_file), str(utils_header_file)
    )
    cutlass_code, header_code, utils_code = get_cuda_code(str(cutlass_cu), str(header_file), str(utils_header_file))

    # Combine CUDA sources
    # Add preprocessor directives to enable half-precision and WMMA for Tensor Cores
    # Must be at the top before any other includes
    cuda_header = """
// Undefine PyTorch's restrictive macros to enable Tensor Core operations
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

// Now include CUDA headers for half precision and WMMA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

// Include CUTLASS headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include <iostream>
#include <type_traits>

"""

    combined_cuda_code = (
        cuda_header
        + utils_code
        + "\n"
        + naive_code
        + "\n"
        + coalesce_code
        + "\n"
        + shared_mem_code
        + "\n"
        + blocktiling_1d_code
        + "\n"
        + blocktiling_2d_code
        + "\n"
        + vectorize_code
        + "\n"
        + warptiling_code
        + "\n"
        + warptiling_multidtype_code
        + "\n"
        + tensorcore_code
        + "\n"
        + tensorcore_double_buffered_code
        + "\n"
        + cutlass_code
    )

    # Create build directory
    build_dir = file_dir / "build" / "gemm_extension"
    build_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(f"🔨 Build directory: {build_dir}")

    # Determine CUTLASS include path from third-party directory
    cutlass_include_path = file_dir.parent / "third-party" / "cutlass" / "include"

    if cutlass_include_path.exists():
        if verbose:
            logger.info(f"📦 Found CUTLASS headers at: {cutlass_include_path}")
    else:
        if verbose:
            logger.warning(
                f"⚠️  CUTLASS headers not found at {cutlass_include_path}. CUTLASS kernels may fail to compile."
            )

    # Prepare extra compiler flags
    extra_cflags = ["-O3", "-std=c++17"]
    extra_include_paths = [str(cutlass_include_path)]

    # Load the extension
    # Note: PyTorch adds -D__CUDA_NO_HALF_* macros by default, but we handle
    # them with #undef in the source code itself (see cuda_header above)
    extension = load_inline(
        name="gemm_cuda_extension",
        cpp_sources=header_code,
        cuda_sources=combined_cuda_code,
        functions=[
            "sgemm_naive",
            "sgemm_global_mem_coalesce",
            "sgemm_shared_mem",
            "sgemm_blocktiling_1d",
            "sgemm_blocktiling_2d",
            "sgemm_vectorize",
            "sgemm_warptiling_default",
            "sgemm_warptiling_fp16",
            "sgemm_warptiling_bf16",
            "sgemm_tensorcore_fp16",
            "sgemm_tensorcore_bf16",
            "sgemm_tensorcore_double_buffered_fp16",
            "sgemm_tensorcore_double_buffered_bf16",
            "sgemm_cutlass_fp16",
            "sgemm_cutlass_bf16",
            "sgemm_cutlass_fp32",
        ],
        with_cuda=True,
        verbose=verbose,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        build_directory=str(build_dir),
    )

    if verbose:
        logger.success("✅ CUDA extension loaded successfully!")

    return extension
