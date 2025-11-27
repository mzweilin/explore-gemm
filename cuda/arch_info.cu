#include "gemm_kernels.cuh"
#include "cuda/utils.cuh"
#include <string>

// Architecture information functions for Python access
std::string get_cutlass_arch_info()
{
    return std::string(get_configured_arch_name());
}

int get_cutlass_arch_sm()
{
    return get_configured_arch();
}
