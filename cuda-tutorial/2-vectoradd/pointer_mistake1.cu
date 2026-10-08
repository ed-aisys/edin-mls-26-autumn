// Slide 47: a host (malloc) pointer handed to a kernel.
// Expected on the teaching cluster:
//   CUDA error at pointer_mistake1.cu:21 code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()"
// Remove the checkCudaErrors(...) wrapper and the program silently prints 0 instead.
// On HMM/ATS systems (nvidia-smi -q | grep "Addressing Mode") the GPU can read malloc memory
// and the program prints 55.
#include <cstdio>
#include <cstdlib>
#include "../common/check.h"

__global__ void kernel(int *arr) {
  arr[0] = 0;
  for (int i = 1; arr[i] != 0; ++i)
    arr[0] += arr[i];
}

int main() {
  int *a = (int*)calloc(12, sizeof(int));
  for (int i = 1; i <= 10; ++i) a[i] = i;
  kernel<<<1, 1>>>(a);   // host pointer!
  checkCudaErrors(
      cudaDeviceSynchronize());
  printf("%d\n", a[0]);
  free(a);
  return 0;
}
