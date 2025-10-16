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

TEST_CASE("SGEMM Warptiling FP32 - Basic functionality", "[sgemm_warptiling_fp32]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    SECTION("Small matrix - 128x128") {
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_warptiling_fp32(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "128x128 FP32 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }

    SECTION("Medium matrix - 256x256") {
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_warptiling_fp32(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "256x256 FP32 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }
}

TEST_CASE("SGEMM Warptiling FP16 - Basic functionality", "[sgemm_warptiling_fp16]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    SECTION("Small matrix - 128x128") {
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Compute expected result (convert to FP32 for matmul)
        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;

        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "128x128 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-2f);  // FP16 has lower precision
    }

    SECTION("Medium matrix - 256x256") {
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "256x256 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);
    }
}

TEST_CASE("SGEMM Warptiling BF16 - Basic functionality", "[sgemm_warptiling_bf16]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    SECTION("Small matrix - 128x128") {
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Compute expected result (convert to FP32 for matmul)
        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;

        sgemm_warptiling_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "128x128 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);  // BF16 has lower precision
    }

    SECTION("Medium matrix - 256x256") {
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_warptiling_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "256x256 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 3e-2f);
    }
}

TEST_CASE("SGEMM Warptiling FP16 - Alpha/Beta scaling", "[sgemm_warptiling_fp16]") {
    const int M = 128;
    const int K = 128;
    const int N = 128;

    SECTION("Alpha = 2.0, Beta = 0.0") {
        const float alpha = 2.0f;
        const float beta = 0.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c;
        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Alpha=2.0, Beta=0.0 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);
    }

    SECTION("Alpha = 1.0, Beta = 1.0") {
        const float alpha = 1.0f;
        const float beta = 1.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::rand({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c_orig = c.clone();

        auto a_fp32 = a.to(torch::kFloat32);
        auto b_fp32 = b.to(torch::kFloat32);
        auto expected = alpha * torch::matmul(a_fp32, b_fp32) + beta * c_orig;
        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Alpha=1.0, Beta=1.0 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);
    }
}
