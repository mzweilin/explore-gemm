// test_gemm_cutlass_hopper.cpp
// Tests for CUTLASS Hopper GEMM kernel (FP16 and BF16)
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

TEST_CASE("SGEMM CUTLASS Hopper FP16 - Architecture check", "[cutlass_hopper][fp16]")
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    REQUIRE(prop.major >= 9);
}

TEST_CASE("SGEMM CUTLASS Hopper FP16 - Basic functionality", "[cutlass_hopper][fp16]")
{
    torch::manual_seed(42);

    SECTION("Small matrix - 128x128")
    {
        int M = 128, N = 128, K = 128;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        // Run CUTLASS Hopper kernel (alpha=1.0, beta=0.0)
        sgemm_cutlass_hopper_fp16(A, B, C);

        // Compare with PyTorch reference
        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kFloat16), ref, TOLERANCE));
    }

    SECTION("Medium matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kFloat16), ref, TOLERANCE));
    }

    SECTION("Large matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kFloat16), ref, TOLERANCE));
    }

    SECTION("Very large matrix - 1024x1024")
    {
        int M = 1024, N = 1024, K = 1024;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kFloat16), ref, TOLERANCE));
    }

    SECTION("Non-square matrix - 256x512x128")
    {
        int M = 256, N = 512, K = 128;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kFloat16), ref, TOLERANCE));
    }

    SECTION("Non-square matrix - 512x256x384")
    {
        int M = 512, N = 256, K = 384;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_fp16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kFloat16), ref, TOLERANCE));
    }
}


TEST_CASE("SGEMM CUTLASS Hopper BF16 - Basic functionality", "[cutlass_hopper][bf16]")
{
    torch::manual_seed(42);

    SECTION("Small matrix - 128x128")
    {
        int M = 128, N = 128, K = 128;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kBFloat16), ref, TOLERANCE));
    }

    SECTION("Medium matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kBFloat16), ref, TOLERANCE));
    }

    SECTION("Large matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kBFloat16), ref, TOLERANCE));
    }

    SECTION("Very large matrix - 1024x1024")
    {
        int M = 1024, N = 1024, K = 1024;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kBFloat16), ref, TOLERANCE));
    }

    SECTION("Non-square matrix - 384x768x256")
    {
        int M = 384, N = 768, K = 256;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        sgemm_cutlass_hopper_bf16(A, B, C);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C.to(torch::kBFloat16), ref, TOLERANCE));
    }
}


TEST_CASE("SGEMM CUTLASS Hopper - Comparison FP16 vs BF16", "[cutlass_hopper][comparison]")
{
    torch::manual_seed(42);

    int M = 512, N = 512, K = 512;
    auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Create input tensors
    auto A_fp16 = torch::rand({M, K}, options_fp16);
    auto B_fp16 = torch::rand({K, N}, options_fp16);

    auto A_bf16 = torch::rand({M, K}, options_bf16);
    auto B_bf16 = torch::rand({K, N}, options_bf16);

    auto C_fp16 = torch::zeros({M, N}, options_fp32);
    auto C_bf16 = torch::zeros({M, N}, options_fp32);

    // Run both kernels (alpha=1.0, beta=0.0)
    sgemm_cutlass_hopper_fp16(A_fp16, B_fp16, C_fp16);
    sgemm_cutlass_hopper_bf16(A_bf16, B_bf16, C_bf16);

    // Compare FP16: PyTorch FP16 matmul vs CUTLASS kernel output (converted to FP16)
    auto ref_fp16 = torch::matmul(A_fp16, B_fp16);
    REQUIRE(tensors_are_close(C_fp16.to(torch::kFloat16), ref_fp16, TOLERANCE));

    // Compare BF16: PyTorch BF16 matmul vs CUTLASS kernel output (converted to BF16)
    auto ref_bf16 = torch::matmul(A_bf16, B_bf16);
    REQUIRE(tensors_are_close(C_bf16.to(torch::kBFloat16), ref_bf16, TOLERANCE));
}
