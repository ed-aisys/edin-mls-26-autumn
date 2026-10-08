// Example 2, slides 66-69: matrix-vector multiplication y = A x, A is M x K (row-major).
//   matvec         slide 67: one thread per output row
//   matvec_stride  slide 68: grid-stride loop over rows
//   matvec_shared  slide 69: tiles of x staged in shared memory (slides 75-76)
// Every version is checked against the CPU reference of slide 66.
#include <cmath>
#include <cstdio>
#include <vector>
#include "../common/check.h"

constexpr int TILE = 256;   // = blockDim.x for matvec_shared

// slide 67
__global__ void matvec(
    const float* A, const float* x,
    float* y, int M, int K) {
  int row = blockIdx.x * blockDim.x
          + threadIdx.x;
  if (row >= M) return;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k)
    sum += A[row*K+k] * x[k];
  y[row] = sum;
}

// slide 68
__global__ void matvec_stride(
    const float* A, const float* x,
    float* y, int M, int K) {
  // Grid-stride loop over rows
  int stride = blockDim.x * gridDim.x;
  int first  = blockIdx.x * blockDim.x
             + threadIdx.x;
  for (int row = first; row < M;
       row += stride) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
      sum += A[row*K+k] * x[k];
    y[row] = sum;
  }
}

// slide 69
__global__ void matvec_shared(
    const float* A, const float* x,
    float* y, int M, int K) {
  __shared__ float sx[TILE];
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  for (int t = 0; t < K; t += TILE) {
    // for each tile of x (all threads):
    int k = t + threadIdx.x;
    sx[threadIdx.x] = k < K ? x[k] : 0.0f;
    __syncthreads();
    if (row < M) {
      // accumulate the valid tile into sum
      for (int j = 0; j < TILE && t + j < K; ++j)
        sum += A[row*K + t + j] * sx[j];
    }
    __syncthreads();
    // next tile only after readers finish
  }
  if (row < M) y[row] = sum;
}

int main() {
  const int M = 4099, K = 1000;   // not multiples of the block size: edge rows and tiles
  std::vector<float> A((size_t)M * K), x(K), y(M), ref(M);
  for (size_t i = 0; i < A.size(); ++i) A[i] = (int(i % 13) - 6) * 0.125f;
  for (int k = 0; k < K; ++k) x[k] = (k % 7 - 3) * 0.25f;

  // slide 66: CPU reference
  for (int row = 0; row < M; ++row) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
      sum += A[row*K+k] * x[k];
    ref[row] = sum;
  }

  float *dA, *dx, *dy;
  checkCudaErrors(cudaMalloc(&dA, A.size() * sizeof(float)));
  checkCudaErrors(cudaMalloc(&dx, K * sizeof(float)));
  checkCudaErrors(cudaMalloc(&dy, M * sizeof(float)));
  checkCudaErrors(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));

  auto check = [&](const char *name) {
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(y.data(), dy, M * sizeof(float), cudaMemcpyDeviceToHost));
    int failures = 0;
    for (int row = 0; row < M; ++row)
      if (std::fabs(y[row] - ref[row]) > 1e-3f * (1 + std::fabs(ref[row]))) ++failures;
    printf("%-14s failures=%d  %s\n", name, failures, failures ? "FAIL" : "PASS");
    checkCudaErrors(cudaMemset(dy, 0, M * sizeof(float)));
    return failures;
  };

  int block = TILE, grid = (M + block - 1) / block, bad = 0;
  matvec<<<grid, block>>>(dA, dx, dy, M, K);
  bad += check("matvec");
  matvec_stride<<<4, block>>>(dA, dx, dy, M, K);   // fewer threads than rows on purpose
  bad += check("matvec_stride");
  matvec_shared<<<grid, block>>>(dA, dx, dy, M, K);
  bad += check("matvec_shared");

  checkCudaErrors(cudaFree(dA));
  checkCudaErrors(cudaFree(dx));
  checkCudaErrors(cudaFree(dy));
  return bad != 0;
}
