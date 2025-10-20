// test_gemm_cutlass.cpp
// Tests for CUTLASS GEMM kernel (FP16, BF16, and FP32)

#define CATCH_CONFIG_MAIN
#include "../../third-party/catch.hpp"
#include "../gemm_kernels.cuh"

#include <torch/torch.h>
#include <iostream>
#include <cmath>

// Tolerance for numerical comparison
constexpr float TOLERANCE = 1e-2f;

// Helper function to check if tensors are close
bool tensors_are_close(const torch::Tensor &a, const torch::Tensor &b, float tol = TOLERANCE)
{
    auto diff = (a - b).abs();
    auto max_diff = diff.max().item<float>();
    return max_diff < tol;
}

TEST_CASE("SGEMM CUTLASS FP16 - Basic functionality", "[cutlass][fp16]")
{
    SECTION("Small matrix - 128x128")
    {
        int M = 128, N = 128, K = 128;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        // Run CUTLASS kernel
        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        // Compare with PyTorch reference
        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Medium matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Large matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Non-square matrix - 256x512x128")
    {
        int M = 256, N = 512, K = 128;
        auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp16);
        auto B = torch::rand({K, N}, options_fp16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }
}

TEST_CASE("SGEMM CUTLASS FP16 - Alpha/Beta scaling", "[cutlass][fp16][scaling]")
{
    int M = 128, N = 128, K = 128;
    auto options_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto A = torch::rand({M, K}, options_fp16);
    auto B = torch::rand({K, N}, options_fp16);

    SECTION("Alpha = 2.0, Beta = 0.0")
    {
        auto C = torch::zeros({M, N}, options_fp32);
        float alpha = 2.0f, beta = 0.0f;

        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Alpha = 1.0, Beta = 1.0")
    {
        auto C = torch::rand({M, N}, options_fp32);
        auto C_original = C.clone();
        float alpha = 1.0f, beta = 1.0f;

        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32)) + beta * C_original;

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Alpha = 0.5, Beta = 0.5")
    {
        auto C = torch::rand({M, N}, options_fp32);
        auto C_original = C.clone();
        float alpha = 0.5f, beta = 0.5f;

        sgemm_cutlass_fp16(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32)) + beta * C_original;

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }
}

TEST_CASE("SGEMM CUTLASS BF16 - Basic functionality", "[cutlass][bf16]")
{
    SECTION("Small matrix - 128x128")
    {
        int M = 128, N = 128, K = 128;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_bf16(A, B, C, alpha, beta);

        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Medium matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_bf16(A, B, C, alpha, beta);

        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }

    SECTION("Large matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_bf16);
        auto B = torch::rand({K, N}, options_bf16);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_bf16(A, B, C, alpha, beta);

        auto ref = torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32));

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }
}

TEST_CASE("SGEMM CUTLASS BF16 - Alpha/Beta scaling", "[cutlass][bf16][scaling]")
{
    int M = 128, N = 128, K = 128;
    auto options_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto A = torch::rand({M, K}, options_bf16);
    auto B = torch::rand({K, N}, options_bf16);

    SECTION("Alpha = 1.0, Beta = 1.0")
    {
        auto C = torch::rand({M, N}, options_fp32);
        auto C_original = C.clone();
        float alpha = 1.0f, beta = 1.0f;

        sgemm_cutlass_bf16(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A.to(torch::kFloat32), B.to(torch::kFloat32)) + beta * C_original;

        REQUIRE(tensors_are_close(C, ref, TOLERANCE));
    }
}

TEST_CASE("SGEMM CUTLASS FP32 - Basic functionality", "[cutlass][fp32]")
{
    SECTION("Small matrix - 128x128")
    {
        int M = 128, N = 128, K = 128;
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp32);
        auto B = torch::rand({K, N}, options_fp32);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        // Run CUTLASS kernel
        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        // Compare with PyTorch reference
        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, 1e-4f));
    }

    SECTION("Medium matrix - 256x256")
    {
        int M = 256, N = 256, K = 256;
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp32);
        auto B = torch::rand({K, N}, options_fp32);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, 1e-4f));
    }

    SECTION("Large matrix - 512x512")
    {
        int M = 512, N = 512, K = 512;
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp32);
        auto B = torch::rand({K, N}, options_fp32);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = torch::matmul(A, B);

        // Larger tolerance for bigger matrices due to accumulation
        REQUIRE(tensors_are_close(C, ref, 1e-3f));
    }

    SECTION("Non-square matrix - 256x512x128")
    {
        int M = 256, N = 512, K = 128;
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp32);
        auto B = torch::rand({K, N}, options_fp32);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = torch::matmul(A, B);

        // Relaxed tolerance for larger matrices
        REQUIRE(tensors_are_close(C, ref, 1e-3f));
    }

    SECTION("Edge case - non-multiple of block size")
    {
        int M = 33, N = 47, K = 29;
        auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        auto A = torch::rand({M, K}, options_fp32);
        auto B = torch::rand({K, N}, options_fp32);
        auto C = torch::zeros({M, N}, options_fp32);

        float alpha = 1.0f, beta = 0.0f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, 1e-4f));
    }
}

TEST_CASE("SGEMM CUTLASS FP32 - Alpha/Beta scaling", "[cutlass][fp32][scaling]")
{
    int M = 128, N = 128, K = 128;
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto A = torch::rand({M, K}, options_fp32);
    auto B = torch::rand({K, N}, options_fp32);

    SECTION("Alpha = 2.0, Beta = 0.0")
    {
        auto C = torch::zeros({M, N}, options_fp32);
        float alpha = 2.0f, beta = 0.0f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A, B);

        REQUIRE(tensors_are_close(C, ref, 1e-4f));
    }

    SECTION("Alpha = 1.0, Beta = 1.0")
    {
        auto C = torch::rand({M, N}, options_fp32);
        auto C_original = C.clone();
        float alpha = 1.0f, beta = 1.0f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A, B) + beta * C_original;

        REQUIRE(tensors_are_close(C, ref, 1e-4f));
    }

    SECTION("Alpha = 0.5, Beta = 0.5")
    {
        auto C = torch::rand({M, N}, options_fp32);
        auto C_original = C.clone();
        float alpha = 0.5f, beta = 0.5f;

        sgemm_cutlass_fp32(A, B, C, alpha, beta);

        auto ref = alpha * torch::matmul(A, B) + beta * C_original;

        REQUIRE(tensors_are_close(C, ref, 1e-4f));
    }
}
