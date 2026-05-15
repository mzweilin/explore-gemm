#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

constexpr int ceil_div(int m, int n)
{
    return (m + n - 1) / n;
}

constexpr int WARPSIZE = 32;

template <typename T>
__device__ inline float output_to_float(T value);

template <>
__device__ inline float output_to_float<float>(float value)
{
    return value;
}

template <>
__device__ inline float output_to_float<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ inline float output_to_float<nv_bfloat16>(nv_bfloat16 value)
{
    return __bfloat162float(value);
}

template <typename T>
__device__ inline T float_to_output(float value);

template <>
__device__ inline float float_to_output<float>(float value)
{
    return value;
}

template <>
__device__ inline half float_to_output<half>(float value)
{
    return __float2half_rn(value);
}

template <>
__device__ inline nv_bfloat16 float_to_output<nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename OutputType, int TileRows, int TileCols>
__device__ inline void load_output_tile_to_smem(
    const OutputType *src, int ld_src, float *smem_tile)
{
    const int lane = threadIdx.x % WARPSIZE;
    for (int idx = lane; idx < TileRows * TileCols; idx += WARPSIZE)
    {
        const int row = idx / TileCols;
        const int col = idx % TileCols;
        smem_tile[idx] = output_to_float(src[row * ld_src + col]);
    }
}

template <typename OutputType, int TileRows, int TileCols>
__device__ inline void store_output_tile_from_smem(
    const float *smem_tile, OutputType *dst, int ld_dst)
{
    const int lane = threadIdx.x % WARPSIZE;
    for (int idx = lane; idx < TileRows * TileCols; idx += WARPSIZE)
    {
        const int row = idx / TileCols;
        const int col = idx % TileCols;
        dst[row * ld_dst + col] = float_to_output<OutputType>(smem_tile[idx]);
    }
}