// Example 3, slides 77 and 81-83: tiled GEMM C = A x B with 16 x 16 shared-memory tiles,
// timed with CUDA events.  A is M x K, B is K x N, C is M x N, all row-major.
// Non-square sizes that are not multiples of TILE exercise the edge tiles and catch
// swapped row/column indices (slide 86, task 2). The result is checked on the CPU.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../common/check.h"

constexpr int TILE = 16;

__global__ void gemm_tiled(const float* a, const float* b, float* c, int M, int N, int K) {
  __shared__ float as[TILE][TILE], bs[TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
  float acc = 0;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int a_col = t * TILE + tx, b_row = t * TILE + ty;
    bool valid_a = row < M && a_col < K, valid_b = b_row < K && col < N;
    int a_index = row * K + a_col, b_index = b_row * N + col;
    // ---- slide 82: the two barriers ----
    // Every thread loads a valid value or zero.
    as[ty][tx] = valid_a ? a[a_index] : 0;
    bs[ty][tx] = valid_b ? b[b_index] : 0;
    __syncthreads();
    for (int q = 0; q < TILE; ++q)
      acc += as[ty][q] * bs[q][tx];
    __syncthreads();
  }
  // No early return above: edge threads load zeros and still reach both barriers.
  if (row < M && col < N) c[row*N+col] = acc;
}

int main(int argc, char** argv) {
  int M = argc > 3 ? atoi(argv[1]) : 1000;
  int N = argc > 3 ? atoi(argv[2]) : 777;
  int K = argc > 3 ? atoi(argv[3]) : 513;
  std::vector<float> a((size_t)M * K), b((size_t)K * N), c((size_t)M * N), ref((size_t)M * N);
  for (size_t i = 0; i < a.size(); ++i) a[i] = (int(i % 13) - 6) * 0.125f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = (int(i % 7) - 3) * 0.25f;
  for (int r = 0; r < M; ++r)
    for (int q = 0; q < K; ++q)
      for (int col = 0; col < N; ++col)
        ref[(size_t)r * N + col] += a[(size_t)r * K + q] * b[(size_t)q * N + col];

  float *da, *db, *dc;
  checkCudaErrors(cudaMalloc(&da, a.size() * sizeof(float)));
  checkCudaErrors(cudaMalloc(&db, b.size() * sizeof(float)));
  checkCudaErrors(cudaMalloc(&dc, c.size() * sizeof(float)));
  checkCudaErrors(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  printf("M=%d N=%d K=%d\n", M, N, K);

  gemm_tiled<<<grid, block, 0, stream>>>(da, db, dc, M, N, K);   // warm up
  checkCudaErrors(cudaGetLastError());
  const int repeats = 20;
  float ms = 0;
  // ---- slide 83: timing GPU work with events ----
  cudaEventRecord(start, stream);
  // Launch the measured work in this stream.
  for (int r = 0; r < repeats; ++r)
    gemm_tiled<<<grid, block, 0, stream>>>(da, db, dc, M, N, K);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  checkCudaErrors(cudaGetLastError());

  // Check results.
  checkCudaErrors(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  int failures = 0;
  for (size_t i = 0; i < c.size(); ++i)
    if (std::fabs(c[i] - ref[i]) > 1e-3f * (1 + std::fabs(ref[i]))) ++failures;
  printf("tiled %.3f ms per launch (kernel only)  failures=%d  %s\n",
         ms / repeats, failures, failures ? "FAIL" : "PASS");

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaFree(da));
  checkCudaErrors(cudaFree(db));
  checkCudaErrors(cudaFree(dc));
  return failures != 0;
}
