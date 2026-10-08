// Optional, slide 57: a race on a shared sum. The GPU value is wrong and changes per run.
// Build with --extended-lambda (device lambda).
#include <cmath>
#include <cstdio>

template <class F>
__global__ void kernel(int n, F f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  for (; i < n; i += step) f(i);
}

__device__ float sum = 0;  // device mem

int main() {
  int n = 65536, block = 128;
  int grid = (n + block - 1) / block;
  int *arr;
  cudaMallocManaged(&arr, n * sizeof(int));
  for (int i = 0; i < n; ++i) arr[i] = i;

  kernel<<<grid, block>>>(n,
    [=] __device__ (int i) {
      sum += sinf(arr[i]);  // race!
    });

  float result = 0;
  cudaMemcpyFromSymbol(&result, sum,
                       sizeof(float));
  printf("GPU %f\n", result);

  // CPU reference
  result = 0;
  for (int i = 0; i < n; ++i)
    result += sinf(i);
  printf("CPU %f\n", result);

  cudaFree(arr);
  return 0;
}
