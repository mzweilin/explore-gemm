import os
from typing import Any, Callable
import torch
import triton
import triton.language as tl
from loguru import logger


# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def matmul_kernel_coalesced(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Flattened 1D block into 2D coordinates
    tid = tl.arange(0, BLOCK_SIZE * BLOCK_SIZE)
    offs_x = pid_m * BLOCK_SIZE + (tid // BLOCK_SIZE)
    offs_y = pid_n * BLOCK_SIZE + (tid % BLOCK_SIZE)

    # Valid mask
    mask = (offs_x < M) & (offs_y < N)

    # Accumulator (always FP32)
    acc = tl.zeros(tid.shape, dtype=tl.float32)

    # Loop over K
    for k in range(0, K):
        # Load A and B with bounds checking
        a = tl.load(A_ptr + offs_x * K + k, mask=offs_x < M, other=0.0)
        b = tl.load(B_ptr + k * N + offs_y, mask=offs_y < N, other=0.0)

        # Accumulate in FP32
        acc += a.to(tl.float32) * b.to(tl.float32)

    # Compute C pointer
    c_ptrs = C_ptr + offs_x * N + offs_y

    # Load old C
    c_old = tl.load(c_ptrs, mask=mask, other=0.0)

    # Final computation in FP32
    c_new = (alpha * acc + beta * c_old.to(tl.float32)).to(c_old.dtype)

    # Store back
    tl.store(c_ptrs, c_new, mask=mask)


def matmul_coalesced(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size: int = 2,
    print_ir: bool = False,
    print_ptx: bool = False,
) -> torch.Tensor:
    """
    Coalesced memory access matrix multiplication using Triton.
    Computes C = A @ B where A is MxK and B is KxN.

    Similar to CUDA coalesced kernel - uses a 1D grid where each program
    computes one output element with coalesced memory access pattern.

    Args:
        a: Input matrix A (MxK)
        b: Input matrix B (KxN)
        block_size: Block size (default: 32)
        print_ir: If True, print TTGIR (Triton IR)
        print_ptx: If True, print PTX assembly

    Returns:
        Output matrix C (MxN)
    """
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} and {b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    _, N = b.shape
    alpha = 1.0
    beta = 0.0

    c_out = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE: tl.constexpr = 32  # type: ignore

    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))  # type: ignore
    compiled = matmul_kernel_coalesced[grid](
        a, b, c_out, M, N, K, alpha, beta, BLOCK_SIZE=BLOCK_SIZE  # type: ignore
    )

    # Print IR/PTX if requested
    if print_ir or print_ptx:
        if print_ir:
            logger.info("=" * 80)
            logger.info("🔬 TTGIR (Triton IR)")
            logger.info("=" * 80)
            print(compiled.asm["ttgir"])
            logger.info("=" * 80)

        if print_ptx:
            logger.info("=" * 80)
            logger.info("🔧 PTX Assembly")
            logger.info("=" * 80)
            print(compiled.asm["ptx"])
            logger.info("=" * 80)

    return c_out
