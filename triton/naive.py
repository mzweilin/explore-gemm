import torch
import triton
import triton.language as tl
from loguru import logger


@triton.jit
def matmul_naive_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs: each kernel instance computes a tile of C
    pid_m = tl.program_id(0)  # along M
    pid_n = tl.program_id(1)  # along N

    # Compute the row/col offsets of the element(s) in C
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulation
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K):
        # load A[:, k] and B[k, :]
        a = tl.load(A_ptr + offs_m * K + k, mask=offs_m < M, other=0.0)
        b = tl.load(B_ptr + k * N + offs_n, mask=offs_n < N, other=0.0)
        # outer product accumulate
        acc += a[:, None].to(tl.float32) * b[None, :].to(tl.float32)

    # Scale and write back
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]

    # Compute final result in fp32, then convert to output dtype
    c_old = tl.load(
        c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0
    )
    c_new = (alpha * acc + beta * c_old.to(tl.float32)).to(c_old.dtype)
    tl.store(c_ptrs, c_new, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
    _, N = b.shape
    alpha = 1.0
    beta = 0.0

    c_out = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M: tl.constexpr = 32  # type: ignore
    BLOCK_N: tl.constexpr = 32  # type: ignore

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # type: ignore

    compiled = matmul_naive_kernel[grid](
        a, b, c_out, M, N, K, alpha, beta, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N  # type: ignore
    )

    # # Print IR/PTX if requested
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

    return c_out
