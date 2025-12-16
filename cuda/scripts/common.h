/***************************************************************************************************
 * Shared utilities for GEMM benchmarking
 **************************************************************************************************/

#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Helper for CUDA errors
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess)                                           \
    {                                                                   \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// GPU timer using CUDA events
class GpuTimer
{
public:
  GpuTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start()
  {
    cudaEventRecord(start_, 0);
  }

  void stop()
  {
    cudaEventRecord(stop_, 0);
  }

  float elapsed_millis()
  {
    float elapsed;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element>& block, uint64_t seed=2023) {
  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Result structure for benchmark results
struct BenchmarkResult
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  BenchmarkResult(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}
};

/// Helper to parse options from a mapping
template <class T>
bool parse_from_options_map(
    const std::string& val,
    const std::unordered_map<T, std::vector<std::string>>& options,
    T& result)
{
  for (const auto & [key, values] : options) {
    if (std::find(values.begin(), values.end(), val) != values.end()) {
      result = key;
      return true;
    }
  }
  return false;
}
