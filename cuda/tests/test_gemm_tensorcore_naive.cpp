#define CATCH_CONFIG_MAIN
#include "../../third-party/catch.hpp"
#include "../gemm_kernels.cuh"
#include <torch/torch.h>
#include <cmath>

// Helper function to compute maximum absolute difference
float max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    auto diff = (a - b).abs();
    return diff.max().item<float>();
}

TEST_CASE("SGEMM Tensor Core Naive FP16 - Basic functionality", "[sgemm_tensorcore_naive_fp16]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    torch::manual_seed(42);

    SECTION("Small matrix - 256x256") {
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        // Compute expected result using PyTorch (convert to FP32 for matmul)
        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;

        // Compute using our naive Tensor Core kernel
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        // Check results match (relaxed tolerance for FP16)
        float diff = max_diff(c, expected);
        std::cout << "Naive 256x256 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 4e-2f);
    }

    SECTION("Medium matrix - 512x512") {
        const int M = 512;
        const int K = 512;
        const int N = 512;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive 512x512 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 8e-2f);
    }

    SECTION("Large matrix - 1024x1024") {
        const int M = 1024;
        const int K = 1024;
        const int N = 1024;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive 1024x1024 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1.6e-1f);
    }

    SECTION("Rectangular matrix - 512x1024x512") {
        const int M = 512;
        const int K = 1024;
        const int N = 512;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive 512x1024x512 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1.6e-1f);
    }
}

TEST_CASE("SGEMM Tensor Core Naive BF16 - Basic functionality", "[sgemm_tensorcore_naive_bf16]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    torch::manual_seed(42);

    SECTION("Small matrix - 256x256") {
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        // Compute expected result using PyTorch (convert to FP32 for matmul)
        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;

        // Compute using our naive Tensor Core kernel
        sgemm_tensorcore_naive_bf16(a, b, c, alpha, beta);

        // Check results match (relaxed tolerance for BF16)
        float diff = max_diff(c, expected);
        std::cout << "Naive 256x256 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 3e-1f);
    }

    SECTION("Medium matrix - 512x512") {
        const int M = 512;
        const int K = 512;
        const int N = 512;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive 512x512 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 6e-1f);
    }

    SECTION("Large matrix - 1024x1024") {
        const int M = 1024;
        const int K = 1024;
        const int N = 1024;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive 1024x1024 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1.2f);
    }

    SECTION("Rectangular matrix - 512x1024x512") {
        const int M = 512;
        const int K = 1024;
        const int N = 512;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive 512x1024x512 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1.2f);
    }
}

TEST_CASE("SGEMM Tensor Core Naive FP16 - Alpha/Beta scaling", "[sgemm_tensorcore_naive_fp16]") {
    const int M = 512;
    const int K = 512;
    const int N = 512;
    torch::manual_seed(42);

    SECTION("Alpha = 2.0, Beta = 0.0") {
        const float alpha = 2.0f;
        const float beta = 0.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, a.options());

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive Alpha=2.0, Beta=0.0 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1.6e-1f);
    }

    SECTION("Alpha = 1.0, Beta = 1.0") {
        const float alpha = 1.0f;
        const float beta = 1.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::rand({M, N}, a.options());
        auto c_orig = c.clone();

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c_orig;
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive Alpha=1.0, Beta=1.0 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 8e-2f);
    }

    SECTION("Alpha = 0.5, Beta = 1.5") {
        const float alpha = 0.5f;
        const float beta = 1.5f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::rand({M, N}, a.options());
        auto c_orig = c.clone();

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c_orig;
        sgemm_tensorcore_naive_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive Alpha=0.5, Beta=1.5 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 4e-2f);
    }
}

TEST_CASE("SGEMM Tensor Core Naive BF16 - Alpha/Beta scaling", "[sgemm_tensorcore_naive_bf16]") {
    const int M = 512;
    const int K = 512;
    const int N = 512;
    torch::manual_seed(42);

    SECTION("Alpha = 1.0, Beta = 1.0") {
        const float alpha = 1.0f;
        const float beta = 1.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::rand({M, N}, a.options());
        auto c_orig = c.clone();

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c_orig;
        sgemm_tensorcore_naive_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Naive Alpha=1.0, Beta=1.0 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 6e-1f);
    }
}
