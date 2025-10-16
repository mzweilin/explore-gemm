#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"

// Helper function to compute maximum absolute difference
float max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    auto diff = (a - b).abs();
    return diff.max().item<float>();
}

// Benchmark configuration
struct BenchmarkConfig {
    int M;
    int N;
    int K;
    int warmup_iterations = 10;
    int benchmark_iterations = 100;
};

// Benchmark result
struct BenchmarkResult {
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float tflops;
    float bandwidth_gbps;
};

// Calculate TFLOPS and bandwidth
void calculate_metrics(int M, int N, int K, float avg_time_ms,
                      float &tflops, float &bandwidth_gbps) {
    // FLOPs: 2MNK for matrix multiplication
    double flops = 2.0 * M * N * K;
    tflops = (flops / (avg_time_ms * 1e-3)) * 1e-12;

    // Memory bandwidth: read A (MxK), read B (KxN), write C (MxN)
    const int bytes_per_element = 4;  // float32
    double bytes_total = (M * K + K * N + M * N) * bytes_per_element;
    bandwidth_gbps = (bytes_total * 1e-9) / (avg_time_ms * 1e-3);
}

// Benchmark a kernel function
template<typename KernelFunc>
BenchmarkResult benchmark_kernel(KernelFunc kernel_fn,
                                const torch::Tensor &a,
                                const torch::Tensor &b,
                                const BenchmarkConfig &config) {
    const int M = config.M;
    const int N = config.N;
    const int K = config.K;

    // Warmup
    for (int i = 0; i < config.warmup_iterations; ++i) {
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        kernel_fn(a, b, c, 1.0f, 0.0f);
    }

    // Create CUDA events for timing
    std::vector<cudaEvent_t> start_events(config.benchmark_iterations);
    std::vector<cudaEvent_t> end_events(config.benchmark_iterations);

    for (int i = 0; i < config.benchmark_iterations; ++i) {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
    }

    // Benchmark
    std::vector<float> times_ms;
    for (int i = 0; i < config.benchmark_iterations; ++i) {
        auto c = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        cudaEventRecord(start_events[i]);
        kernel_fn(a, b, c, 1.0f, 0.0f);
        cudaEventRecord(end_events[i]);
    }

    cudaDeviceSynchronize();

    // Calculate times
    for (int i = 0; i < config.benchmark_iterations; ++i) {
        float time_ms;
        cudaEventElapsedTime(&time_ms, start_events[i], end_events[i]);
        times_ms.push_back(time_ms);
    }

    // Clean up events
    for (int i = 0; i < config.benchmark_iterations; ++i) {
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }

    // Trim outliers (top and bottom 10%)
    std::sort(times_ms.begin(), times_ms.end());
    int trim_count = std::max(1, config.benchmark_iterations / 10);
    std::vector<float> times_trimmed(times_ms.begin() + trim_count,
                                    times_ms.end() - trim_count);

    float avg_time = 0.0f;
    for (float t : times_trimmed) {
        avg_time += t;
    }
    avg_time /= times_trimmed.size();

    BenchmarkResult result;
    result.avg_time_ms = avg_time;
    result.min_time_ms = *std::min_element(times_trimmed.begin(), times_trimmed.end());
    result.max_time_ms = *std::max_element(times_trimmed.begin(), times_trimmed.end());

    calculate_metrics(M, N, K, avg_time, result.tflops, result.bandwidth_gbps);

    return result;
}

void print_header() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             CUDA GEMM Autotunable Kernel Benchmark Suite                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_config_header(int M, int N, int K) {
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Matrix Configuration: " << std::setw(4) << M << " × " << std::setw(4) << K
              << " @ " << std::setw(4) << K << " × " << std::setw(4) << N
              << " = " << std::setw(4) << M << " × " << std::setw(4) << N << std::setw(27) << " ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
}

void print_result(const std::string &config_name, const BenchmarkResult &result,
                 const BenchmarkResult &baseline, bool is_baseline = false) {
    std::cout << std::fixed << std::setprecision(4);

    float speedup = is_baseline ? 1.0f : baseline.avg_time_ms / result.avg_time_ms;
    std::string speedup_indicator = speedup > 1.0f ? "🏆" : speedup < 1.0f ? "🐢" : "⚖️";

    std::cout << "  " << std::left << std::setw(30) << config_name
              << " │ " << std::right << std::setw(8) << result.avg_time_ms << " ms"
              << " │ " << std::setw(7) << result.tflops << " TFLOPS"
              << " │ " << std::setw(7) << result.bandwidth_gbps << " GB/s"
              << " │ " << std::setw(6) << std::setprecision(2) << speedup << "× " << speedup_indicator
              << "\n";
}

int main() {
    print_header();

    if (!torch::cuda::is_available()) {
        std::cerr << "❌ CUDA is not available. Exiting.\n";
        return 1;
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "🖥️  GPU: " << prop.name << "\n";
    std::cout << "💾 GPU Memory: " << (prop.totalGlobalMem / 1e9) << " GB\n";
    std::cout << "⚡ Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "\n";

    // Test configurations
    std::vector<BenchmarkConfig> configs = {
        {256, 256, 256, 10, 100},
        {512, 512, 512, 10, 100},
        {1024, 1024, 1024, 10, 100},
        {2048, 2048, 2048, 10, 50},
        {4096, 4096, 4096, 5, 20},
    };

    for (const auto &config : configs) {
        const int M = config.M;
        const int N = config.N;
        const int K = config.K;

        print_config_header(M, N, K);

        std::cout << "\n";
        std::cout << "  " << std::left << std::setw(30) << "Configuration"
                  << " │ " << std::right << std::setw(10) << "Avg Time"
                  << " │ " << std::setw(14) << "Performance"
                  << " │ " << std::setw(13) << "Bandwidth"
                  << " │ " << std::setw(9) << "Speedup"
                  << "\n";
        std::cout << "  " << std::string(30, '─') << "─┼─" << std::string(10, '─')
                  << "─┼─" << std::string(14, '─') << "─┼─" << std::string(13, '─')
                  << "─┼─" << std::string(9, '─') << "\n";

        try {
            // Create test matrices
            auto a = torch::rand({M, K}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto b = torch::rand({K, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            // Benchmark PyTorch (baseline)
            std::cout << "  🔵 Benchmarking PyTorch baseline..." << std::flush;
            auto baseline_result = benchmark_kernel(
                [](const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c, float alpha, float beta) {
                    c = torch::matmul(a, b);
                }, a, b, config);
            std::cout << "\r";
            print_result("PyTorch (baseline)", baseline_result, baseline_result, true);

            // Benchmark default warptiling configuration
            std::cout << "  💫 Benchmarking warptiling default..." << std::flush;
            auto warptiling_default = benchmark_kernel(sgemm_warptiling_default, a, b, config);
            std::cout << "\r";
            print_result("Warptiling (default)", warptiling_default, baseline_result);

            // Benchmark Config 2: BM=128, BN=128, WM=64, WN=32
            std::cout << "  🟢 Benchmarking warptiling Config 2..." << std::flush;
            auto warptiling_config2 = benchmark_kernel(
                [](const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c, float alpha, float beta) {
                    sgemm_warptiling<128, 128, 16, 64, 32, 2, 8, 4, 256>(a, b, c, alpha, beta);
                }, a, b, config);
            std::cout << "\r";
            print_result("Warptiling (128,128,16,64,32,2,8,4,256)", warptiling_config2, baseline_result);

            // Verify correctness of default configuration
            auto c_pytorch = torch::matmul(a, b);
            auto c_warptiling = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            sgemm_warptiling_default(a, b, c_warptiling, 1.0f, 0.0f);
            float diff = max_diff(c_pytorch, c_warptiling);

            std::cout << "\n  ✅ Correctness check: max_diff = " << diff;
            if (diff < 1e-3f) {
                std::cout << " (PASSED)\n";
            } else {
                std::cout << " (WARNING: High error!)\n";
            }

        } catch (const std::exception &e) {
            std::cout << "\n  ❌ Error: " << e.what() << "\n";
        }

        std::cout << "\n";
    }

    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         Benchmark Complete!                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";

    return 0;
}
