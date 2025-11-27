#pragma once

#include "cutlass/arch/arch.h"

constexpr int ceil_div(int m, int n)
{
    return (m + n - 1) / n;
}

constexpr int WARPSIZE = 32;

// GPU Architecture Configuration
// 80: Ampere (A100, RTX 3090, etc.)
// 89: Ada Lovelace (RTX 4090, etc.) - Default
// 90: Hopper (H100, etc.)
// Change this value to match your GPU architecture
constexpr int GPU_SM_ARCH = 89;

// Helper to select CUTLASS architecture type based on GPU_SM_ARCH
template <int SmArch>
struct CutlassArchSelector;

// Specialization for SM80 (Ampere)
template <>
struct CutlassArchSelector<80>
{
    using Arch = cutlass::arch::Sm80;
};

// Specialization for SM89 (Ada Lovelace) - uses SM80 CUTLASS ops
template <>
struct CutlassArchSelector<89>
{
    using Arch = cutlass::arch::Sm80;
};

// Specialization for SM90 (Hopper)
template <>
struct CutlassArchSelector<90>
{
    using Arch = cutlass::arch::Sm90;
};

// Convenience alias
using SelectedCutlassArch = typename CutlassArchSelector<GPU_SM_ARCH>::Arch;
