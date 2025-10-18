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
        torch::manual_seed(42);
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        // Use the base warptiling kernel for FP32 (from 07_kernel_warptiling.cu)
        sgemm_warptiling_default(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "128x128 FP32 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }

    SECTION("Medium matrix - 256x256") {
        torch::manual_seed(42);
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        // Use the base warptiling kernel for FP32 (from 07_kernel_warptiling.cu)
        sgemm_warptiling_default(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "256x256 FP32 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }
}

TEST_CASE("SGEMM Warptiling FP16 - Basic functionality", "[sgemm_warptiling_fp16]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    SECTION("Small matrix - 128x128") {
        torch::manual_seed(42);
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

        // Compute expected result (use FP16 matmul for comparison)
        auto expected = alpha * torch::matmul(a, b) + beta * c;

        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        std::cout << "Expected tensor dtype: " << expected.dtype() << std::endl;
        std::cout << "Computed tensor dtype: " << c.dtype() << std::endl;
        auto diff_tensor = (c - expected).abs();
        std::cout << "Difference tensor dtype: " << diff_tensor.dtype() << std::endl;
        std::cout << diff_tensor << std::endl;
        std::cout << "Max difference value: " << diff_tensor.max().item<float>() << std::endl;
        std::cout << diff_tensor.min().item<float>() << std::endl;
        std::cout << torch::count_nonzero(diff_tensor ) << std::endl;

        float diff = max_diff(c.to(torch::kFloat32), expected.to(torch::kFloat32));
        std::cout << "128x128 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-2f);  // FP16 has lower precision
    }

    SECTION("Medium matrix - 256x256") {
        torch::manual_seed(42);
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c.to(torch::kFloat32), expected.to(torch::kFloat32));
        std::cout << "256x256 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);
    }
}

TEST_CASE("SGEMM Warptiling BF16 - Basic functionality", "[sgemm_warptiling_bf16]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    SECTION("Small matrix - 128x128") {
        torch::manual_seed(42);
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

        // Compute expected result (use BF16 matmul for comparison)
        auto expected = alpha * torch::matmul(a, b) + beta * c;

        sgemm_warptiling_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c.to(torch::kFloat32), expected.to(torch::kFloat32));
        std::cout << "128x128 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);  // BF16 has lower precision
    }

    SECTION("Medium matrix - 256x256") {
        torch::manual_seed(42);
        const int M = 256;
        const int K = 256;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_warptiling_bf16(a, b, c, alpha, beta);

        float diff = max_diff(c.to(torch::kFloat32), expected.to(torch::kFloat32));
        std::cout << "256x256 BF16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 3e-2f);
    }
}

TEST_CASE("SGEMM Warptiling FP16 - Alpha/Beta scaling", "[sgemm_warptiling_fp16]") {
    const int M = 128;
    const int K = 128;
    const int N = 128;

    SECTION("Alpha = 2.0, Beta = 0.0") {
        torch::manual_seed(42);
        const float alpha = 2.0f;
        const float beta = 0.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c.to(torch::kFloat32), expected.to(torch::kFloat32));
        std::cout << "Alpha=2.0, Beta=0.0 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);
    }

    SECTION("Alpha = 1.0, Beta = 1.0") {
        torch::manual_seed(42);
        const float alpha = 1.0f;
        const float beta = 1.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c = torch::rand({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
        auto c_orig = c.clone();

        auto expected = alpha * torch::matmul(a, b) + beta * c_orig;
        sgemm_warptiling_fp16(a, b, c, alpha, beta);

        float diff = max_diff(c.to(torch::kFloat32), expected.to(torch::kFloat32));
        std::cout << "Alpha=1.0, Beta=1.0 FP16 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-2f);
    }
}
