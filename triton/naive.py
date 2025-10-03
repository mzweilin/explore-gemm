from typing import Any, Callable
import torch
import triton
import triton.language as tl
from loguru import logger


@triton.jit
def matmul_kernel_naive(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    # Each program/thread computes one element of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute C[pid_m, pid_n]
    # Load row from A and column from B
    offs_k = tl.arange(0, K)

    # For row-major contiguous tensors:
    # A[pid_m, k] = a_ptr + pid_m * K + k
    # B[k, pid_n] = b_ptr + k * N + pid_n
    a_ptrs = a_ptr + pid_m * K + offs_k
    b_ptrs = b_ptr + offs_k * N + pid_n

    a = tl.load(a_ptrs, mask=offs_k < K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k < K, other=0.0)

    # Compute dot product with float32 accumulation for better precision
    acc = tl.sum(a.to(tl.float32) * b.to(tl.float32)).to(a.dtype)

    # Write result: C[pid_m, pid_n] = c_ptr + pid_m * N + pid_n
    c_ptr_offset = c_ptr + pid_m * N + pid_n
    tl.store(c_ptr_offset, acc)


def matmul_naive(
    a: torch.Tensor,
    b: torch.Tensor,
    print_ir: bool = False,
    print_ptx: bool = False,
) -> torch.Tensor:
    """
    Naive matrix multiplication using Triton.
    Computes C = A @ B where A is MxK and B is KxN.
    Assumes row-major contiguous tensors.

    Args:
        a: Input matrix A (MxK)
        b: Input matrix B (KxN)
        print_ir: If True, print TTGIR (Triton IR)
        print_ptx: If True, print PTX assembly

    Returns:
        Output matrix C (MxN)
    """
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} and {b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K2, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid: Callable[[Any], tuple[int, int]] = lambda META: (M, N)

    kernel = matmul_kernel_naive[grid]

    # Execute kernel
    compiled = kernel(
        a,
        b,
        c,
        M,  # type: ignore
        N,  # type: ignore
        K,  # type: ignore
    )

    # Print IR/PTX if requested
    if print_ir or print_ptx:
        # Get the compiled kernel
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

    return c
