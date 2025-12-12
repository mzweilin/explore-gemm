// test_gemm_cutlass_hopper.cpp
// Tests for CUTLASS Hopper GEMM kernel (BF16 only)
// Uses CUTLASS 3.x Collective Builder API for SM90 (Hopper/H100)

#define CATCH_CONFIG_MAIN
#include "../../third-party/catch.hpp"
#include "../gemm_kernels.cuh"

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Tolerance for numerical comparison
constexpr float TOLERANCE = 1e-3f;

// Helper function to check if tensors are close
bool tensors_are_close(const torch::Tensor &a, const torch::Tensor &b, float tol = TOLERANCE)
{
    auto diff = (a - b).abs();
    auto max_diff = diff.max().item<float>();
    return max_diff < tol;
}

TEST_CASE("SGEMM CUTLASS Hopper BF16 - Architecture check", "[cutlass_hopper][bf16]")
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    REQUIRE(prop.major >= 9);
}

TEST_CASE("SGEMM CUTLASS Hopper BF16 - Basic functionality", "[cutlass_hopper][bf16]")
{
    torch::manual_seed(42);

    SECTION("Small matrix - 128x128")
    {
        int M = 128, N = 128, K = 128;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Medium matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Large matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Very large matrix - 1024x1024")
    {
        int M = 1024, N = 1024, K = 1024;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Non-square matrix - 384x768x256")
    {
        int M = 384, N = 768, K = 256;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_bf16);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }
}


