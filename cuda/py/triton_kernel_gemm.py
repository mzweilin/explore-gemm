"""Triton GEMM Kernel Implementation

Based on the official Triton tutorial:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

Implements matrix multiplication with automatic kernel tuning that selects optimal
block sizes, pipeline stages, and number of warps for different matrix dimensions.

Supported dtypes: fp16, bf16, fp32
"""

import torch
import triton
import triton.language as tl


def get_configs():
    """
    Generate autotuning configurations for GEMM.

    Uses the same block size configurations as the CUTLASS autotuner,
    but adapted for Triton's num_warps and num_stages parameters.

    The configurations cover a range of matrix sizes and hardware capabilities:
    - Block sizes: M x N x K (32-256 x 32-256 x 32-64)
    - Pipeline stages: 3-5
    - Warps per block: 2-8 (calculated to match CUTLASS workload)
    """
    return [
        # Config 0: 128x256x64, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        # Config 1: 64x256x32, stages=4, warps=4
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # Config 2: 128x128x32, stages=4, warps=4
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # Config 3: 128x64x32, stages=4, warps=4
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # Config 4: 64x128x32, stages=4, warps=4
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # Config 5: 128x32x32, stages=4, warps=4
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        # Config 6: 64x32x32, stages=5, warps=2
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Config 7: 32x64x32, stages=5, warps=2
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Config 8: 128x128x64, stages=4, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Config 9: 128x64x64, stages=4, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Config 10: 64x128x64, stages=4, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Config 11: 256x256x32, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        # Config 12: 256x128x32, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        # Config 13: 128x256x32, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        # Config 14: 64x64x32, stages=5, warps=2
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Config 15: 256x256x64, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        # Config 16: 256x128x64, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        # Config 17: 128x256x64, stages=4, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Config 18: 256x256x64, stages=4, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        # Config 19: 128x128x64, stages=3, warps=8
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides (how to jump to next row/col)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters (compile-time constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton matrix multiplication kernel with automatic tuning.

    Computes C = A @ B where:
    - A is M×K
    - B is K×N
    - C is M×N

    Uses grouped ordering (GROUP_SIZE_M) to improve L2 cache locality.
    Accumulates in FP32 for numerical stability, then converts to output dtype.

    Args:
        a_ptr: Pointer to matrix A
        b_ptr: Pointer to matrix B
        c_ptr: Pointer to output matrix C
        M, N, K: Matrix dimensions
        stride_*: Memory strides for each matrix
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes (tunable)
        GROUP_SIZE_M: Grouping size for better cache locality
    """
    # Get the program ID and compute block coordinates
    pid = tl.program_id(axis=0)

    # Number of blocks in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grouped ordering: process blocks in groups for better L2 cache reuse
    # Instead of processing block (pid % num_pid_m, pid // num_pid_m),
    # we process blocks in a grouped manner
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Starting positions for this block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute the matrix multiplication with K loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Current K offset
        k_offset = k * BLOCK_SIZE_K

        # Create masks for boundary conditions
        mask_k = k_offset + offs_k < K

        # Load A and B with bounds checking
        # A is [M, K], we load [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

        # B is [K, N], we load [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_ptrs = b_ptr + ((k_offset + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # Accumulate using tensor cores (dot product)
        accumulator = tl.dot(a, b, accumulator)

    # Convert accumulator to output dtype and store
    c = accumulator.to(c_ptr.dtype.element_ty)

    # Create output mask for boundary conditions
    mask_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M
    mask_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < N

    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute matrix multiplication C = A @ B using Triton.

    Automatically selects optimal kernel configuration for the given matrix dimensions.
    Supports FP16, BF16, and FP32 data types.

    Args:
        a: Input matrix A (M×K), must be contiguous on CUDA device
        b: Input matrix B (K×N), must be contiguous on CUDA device

    Returns:
        Output matrix C (M×N) with same dtype as inputs

    Example:
        >>> a = torch.randn((256, 512), device='cuda', dtype=torch.float16)
        >>> b = torch.randn((512, 1024), device='cuda', dtype=torch.float16)
        >>> c = matmul(a, b)
        >>> assert c.shape == (256, 1024)
    """
    # Validate inputs
    assert a.is_cuda, "Input matrices must be on CUDA device"
    assert b.is_cuda, "Input matrices must be on CUDA device"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert a.dtype == b.dtype, f"Input dtypes must match: {a.dtype} vs {b.dtype}"
    assert a.shape[1] == b.shape[0], f"Incompatible shapes for matmul: {a.shape} @ {b.shape}"

    # Only support fp16, bf16, and fp32
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    assert a.dtype in supported_dtypes, (
        f"Unsupported dtype {a.dtype}. Only {supported_dtypes} are supported."
    )

    # Get matrix dimensions
    M, K = a.shape
    K_b, N = b.shape

    # Allocate output with same dtype as inputs
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define the grid: number of blocks to launch
    # We launch blocks in a 1D grid with one block per output tile
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c
