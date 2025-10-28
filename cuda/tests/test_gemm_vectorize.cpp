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

TEST_CASE("SGEMM Vectorize - Basic functionality", "[sgemm_vectorize]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    torch::manual_seed(42);

    SECTION("Small matrix - 128x128") {
        const int M = 128;
        const int K = 128;
        const int N = 128;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        // Compute expected result using PyTorch
        auto expected = alpha * torch::matmul(a, b) + beta * c;

        // Compute using our kernel
        sgemm_vectorize(a, b, c, alpha, beta);

        // Check results match
        float diff = max_diff(c, expected);
        std::cout << "128x128 max_diff: " << diff << std::endl;
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
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "256x256 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }

    SECTION("Large matrix - 512x512") {
        const int M = 512;
        const int K = 512;
        const int N = 512;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "512x512 max_diff: " << diff << std::endl;
        REQUIRE(diff < 5e-4f);  // Relaxed tolerance for larger matrices
    }

    SECTION("Rectangular matrix - 256x512x256") {
        const int M = 256;
        const int K = 512;
        const int N = 256;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "256x512x256 max_diff: " << diff << std::endl;
        REQUIRE(diff < 5e-4f);  // Relaxed tolerance for larger K dimension
    }

    SECTION("Rectangular matrix - 512x256x512") {
        const int M = 512;
        const int K = 256;
        const int N = 512;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "512x256x512 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }

    SECTION("Non-power-of-2 matrix - 384x384 (128*3)") {
        const int M = 384;
        const int K = 384;
        const int N = 384;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "384x384 max_diff: " << diff << std::endl;
        REQUIRE(diff < 5e-4f);  // Non-power-of-2 size
    }

    SECTION("Non-power-of-2 matrix - 640x640 (128*5)") {
        const int M = 640;
        const int K = 640;
        const int N = 640;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "640x640 max_diff: " << diff << std::endl;
        REQUIRE(diff < 5e-4f);
    }
}

TEST_CASE("SGEMM Vectorize - Alpha/Beta scaling", "[sgemm_vectorize]") {
    const int M = 256;
    const int K = 256;
    const int N = 256;
    torch::manual_seed(42);

    SECTION("Alpha = 2.0, Beta = 0.0") {
        const float alpha = 2.0f;
        const float beta = 0.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Alpha=2.0, Beta=0.0 max_diff: " << diff << std::endl;
        REQUIRE(diff < 5e-4f);  // Relaxed tolerance for scaled results
    }

    SECTION("Alpha = 1.0, Beta = 1.0") {
        const float alpha = 1.0f;
        const float beta = 1.0f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::rand({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c_orig = c.clone();

        auto expected = alpha * torch::matmul(a, b) + beta * c_orig;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Alpha=1.0, Beta=1.0 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }

    SECTION("Alpha = 0.5, Beta = 1.5") {
        const float alpha = 0.5f;
        const float beta = 1.5f;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::rand({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c_orig = c.clone();

        auto expected = alpha * torch::matmul(a, b) + beta * c_orig;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "Alpha=0.5, Beta=1.5 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-4f);
    }
}

TEST_CASE("SGEMM Vectorize - Very large power-of-2 matrices", "[sgemm_vectorize]") {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    torch::manual_seed(42);

    SECTION("1024x1024") {
        const int M = 1024;
        const int K = 1024;
        const int N = 1024;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "1024x1024 max_diff: " << diff << std::endl;
        REQUIRE(diff < 1e-3f);  // Slightly relaxed tolerance for larger matrices
    }

    SECTION("2048x2048") {
        const int M = 2048;
        const int K = 2048;
        const int N = 2048;

        auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        auto expected = alpha * torch::matmul(a, b) + beta * c;
        sgemm_vectorize(a, b, c, alpha, beta);

        float diff = max_diff(c, expected);
        std::cout << "2048x2048 max_diff: " << diff << std::endl;
        REQUIRE(diff < 2e-3f);  // Relaxed tolerance for very large matrices
    }
}
