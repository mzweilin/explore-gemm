#pragma once

constexpr int ceil_div(int m, int n)
{
    return (m + n - 1) / n;
}

constexpr int WARPSIZE = 32;