# explore-gemm

CUDA implementations of General Matrix Multiply (GEMM) operations with PyTorch integration.

## Setup

To set up the project with libtorch, run:

```bash
./setup.sh
```

This will download and configure libtorch (PyTorch 2.7.1 with CUDA 12.8 by default) in the `third-party/` directory.

### Custom Versions

You can specify different PyTorch and CUDA versions:

```bash
# Use specific versions
./setup.sh -p 2.5.1 -c 121        # PyTorch 2.5.1 with CUDA 12.1
./setup.sh --pytorch 2.4.0 --cuda 118  # PyTorch 2.4.0 with CUDA 11.8

# See all options
./setup.sh --help
```

**Supported CUDA versions:** 118 (11.8), 121 (12.1), 124 (12.4), 128 (12.8)

## CUDA Kernels

This repository contains optimized SGEMM (Single-precision General Matrix Multiply) implementations:

- **Naive GEMM** ([cuda/01_naive.cu](cuda/naive.cu)) - Basic implementation
- **Global Memory Coalescing** ([cuda/02_kernel_global_mem_coalesce.cu](cuda/kernel_global_mem_coalesce.cu)) - Optimized for coalesced memory access

All kernels include PyTorch tensor wrappers for easy integration. See [cuda/gemm_kernels.cuh](cuda/gemm_kernels.cuh) for the API.

## Usage

```cpp
#include "cuda/gemm_kernels.cuh"

// Create PyTorch tensors on CUDA
auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

// Run GEMM: C = alpha * A @ B + beta * C
auto C = sgemm_naive(A, B, /*alpha=*/1.0f, /*beta=*/0.0f);
// or
auto C = sgemm_global_mem_coalesce(A, B, /*alpha=*/1.0f, /*beta=*/0.0f);
```

## License

See [LICENSE](LICENSE) for details.
