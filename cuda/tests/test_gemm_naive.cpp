#define CATCH_CONFIG_MAIN
#include "../../third-party/catch.hpp"
#include "../gemm_kernels.cuh"
#include <torch/torch.h>

TEST_CASE("SGEMM Naive - Basic functionality", "[sgemm_naive]") {
    // Check if CUDA is available
    REQUIRE(torch::cuda::is_available());

    SECTION("Small square matrices") {
        const int M = 32, K = 32, N = 32;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        auto C = sgemm_naive(A, B, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);
        REQUIRE(C.device().is_cuda());

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-4f);
    }

    SECTION("Rectangular matrices") {
        const int M = 64, K = 48, N = 32;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        auto C = sgemm_naive(A, B, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-4f);
    }

    SECTION("Alpha and beta scaling") {
        const int M = 32, K = 32, N = 32;
        float alpha = 2.0f;
        float beta = 0.5f;

        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto C_init = torch::rand({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        auto C = sgemm_naive(A, B, alpha, beta, C_init);

        // Expected: C = alpha * (A @ B) + beta * C_init
        auto expected = alpha * torch::matmul(A, B) + beta * C_init;
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-3f);
    }

    SECTION("Large matrices") {
        const int M = 512, K = 512, N = 512;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        auto C = sgemm_naive(A, B, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-3f);
    }

    SECTION("Edge case - non-multiple of block size") {
        const int M = 33, K = 47, N = 29;
        auto A = torch::rand({M, K}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
        auto B = torch::rand({K, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

        auto C = sgemm_naive(A, B, 1.0f, 0.0f);

        REQUIRE(C.size(0) == M);
        REQUIRE(C.size(1) == N);

        // Compare with PyTorch's matmul
        auto expected = torch::matmul(A, B);
        auto diff = torch::abs(C - expected);
        auto max_diff = torch::max(diff).item<float>();

        REQUIRE(max_diff < 1e-4f);
    }
}