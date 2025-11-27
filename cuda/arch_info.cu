#include "gemm_kernels.cuh"
#include <string>

// Compile-time architecture verification
// This will print during compilation to confirm the selected architecture
#if GPU_SM_ARCH == 90
#pragma message("Building with SM90 (Hopper) architecture")
#elif GPU_SM_ARCH == 89
#pragma message("Building with SM89 (Ada Lovelace) architecture")
#elif GPU_SM_ARCH == 80
#pragma message("Building with SM80 (Ampere) architecture")
#else
#error "Unsupported GPU_SM_ARCH value. Use 80, 89, or 90."
#endif

// Runtime architecture information
inline const char *get_configured_arch_name()
{
    if constexpr (GPU_SM_ARCH == 90)
        return "SM90 (Hopper)";
    else if constexpr (GPU_SM_ARCH == 89)
        return "SM89 (Ada Lovelace)";
    else if constexpr (GPU_SM_ARCH == 80)
        return "SM80 (Ampere)";
    else
        return "Unknown";
}

inline int get_configured_arch()
{
    return GPU_SM_ARCH;
}
// Architecture information functions for Python access
std::string get_cutlass_arch_info()
{
    return std::string(get_configured_arch_name());
}

int get_cutlass_arch_sm()
{
    return get_configured_arch();
}
