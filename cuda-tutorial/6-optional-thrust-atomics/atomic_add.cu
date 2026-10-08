// Optional, slide 60: fix the race with atomicAdd, or build the same atomic from atomicCAS.
// Both GPU values now differ from the CPU only by floating-point summation order.
// Build with --extended-lambda (device lambda).
#include <cmath>
#include <cstdio>

template <class F>
__global__ void kernel(int n, F f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  for (; i < n; i += step) f(i);
}

__device__ float sum = 0;
__device__ float sum_cas = 0;

// Build your own with atomicCAS
__device__ float
my_atom_add(float *dst, float src) {
  int old = __float_as_int(*dst), expect;
  do {
    expect = old;
    old = atomicCAS((int *)dst, expect,
      __float_as_int(
        __int_as_float(expect) + src));
  } while (expect != old);  // retry
  return __int_as_float(old);
}

int main() {
  int n = 65536, block = 128;
  int grid = (n + block - 1) / block;
  int *arr;
  cudaMallocManaged(&arr, n * sizeof(int));
  for (int i = 0; i < n; ++i) arr[i] = i;

  // Fix: a hardware read-modify-write
  kernel<<<grid, block>>>(n,
    [=] __device__ (int i) {
      atomicAdd(&sum, sinf(arr[i]));
    });
  kernel<<<grid, block>>>(n,
    [=] __device__ (int i) {
      my_atom_add(&sum_cas, sinf(arr[i]));
    });

  float result = 0, result_cas = 0;
  cudaMemcpyFromSymbol(&result, sum, sizeof(float));
  cudaMemcpyFromSymbol(&result_cas, sum_cas, sizeof(float));
  printf("GPU atomicAdd   %f\n", result);
  printf("GPU my_atom_add %f\n", result_cas);

  result = 0;
  for (int i = 0; i < n; ++i)
    result += sinf(i);
  printf("CPU             %f\n", result);

  cudaFree(arr);
  return 0;
}
