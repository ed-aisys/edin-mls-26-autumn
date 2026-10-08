// Optional, slides 57-58: thrust containers, vector add. Prints 752851072.000000.
// Build with --extended-lambda (device lambda).
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>

template <class F>
__global__ void kernel(int n, F f) {
  int i = blockIdx.x * blockDim.x
        + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  for (; i < n; i += step) f(i);
}

int main() {
  int n = 65536, block = 128;
  int grid = (n + block - 1) / block;
  thrust::host_vector<float> x_h(n);
  thrust::host_vector<float> y_h(n);
  // fill both with thrust::generate
  thrust::generate(x_h.begin(), x_h.end(), [] { return std::rand() / 3.0; });
  thrust::generate(y_h.begin(), y_h.end(), [] { return std::rand() / 11.0; });

  thrust::device_vector<float>
      x_d = x_h, y_d = y_h;  // H -> D
  kernel<<<grid, block>>>(n,
    [x = x_d.data(), y = y_d.data()]
    __device__ (int i) {
      x[i] = x[i] + y[i];
    });
  cudaDeviceSynchronize();
  x_h = x_d;                 // D -> H
  printf("%f\n", x_h[0]);
  return 0;
}
