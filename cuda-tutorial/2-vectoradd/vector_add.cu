// Slide 53 "Student checkpoint: vector add" (and smoke.sbatch, slide 85).
// Build on the GPU node:  nvcc -O2 -std=c++17 -arch=native vector_add.cu -o vector_add
// Run:  ./vector_add 1003 256   then   ./vector_add 257 128
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { \
  cudaError_t lecture_status = (call); \
  if (lecture_status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, \
                 #call, cudaGetErrorString(lecture_status)); \
    std::exit(1); \
  } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  // Deliberately small inputs. This program checks correctness, not speed.
  int n = argc > 1 ? std::atoi(argv[1]) : 1003;
  int block = argc > 2 ? std::atoi(argv[2]) : 256;
  if (n <= 0 || n > 1000000 || block <= 0) {
    std::fprintf(stderr, "Usage: %s [n:1..1000000] [block:positive]\n", argv[0]);
    return 2;
  }
  int device = 0;
  CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CHECK(cudaGetDeviceProperties(&props, device));
  if (block > props.maxThreadsPerBlock || block > props.maxThreadsDim[0]) {
    std::fprintf(stderr, "Block size exceeds this device's limit\n");
    return 2;
  }
  std::printf("GPU=%s CC=%d.%d SMs=%d warp=%d\n", props.name, props.major,
              props.minor, props.multiProcessorCount, props.warpSize);
  std::vector<float> a(n), b(n), result(n);
  for (int i = 0; i < n; ++i) {
    a[i] = float(i % 19) * 0.25f;
    b[i] = float(i % 7) * 0.5f;
  }
  size_t bytes = size_t(n) * sizeof(float);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  int grid = (n + block - 1) / block;
  vector_add<<<grid, block>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(result.data(), dc, bytes, cudaMemcpyDeviceToHost));
  int failures = 0;
  for (int i = 0; i < n; ++i) {
    float ref = a[i] + b[i];
    if (std::fabs(result[i] - ref) > 1e-6f) ++failures;
  }
  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dc));
  std::printf("n=%d block=%d grid=%d\nchecked=%d failures=%d\n",
              n, block, grid, n, failures);
  if (failures) return 3;
  std::puts("PASS vector add");
  return 0;
}
