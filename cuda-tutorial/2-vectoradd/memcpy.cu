// Slide 49: explicit copies with cudaMemcpy. Prints 55.
#include <cstdio>
#include <cstdlib>

__global__ void kernel(int *arr) {
  arr[0] = 0;
  for (int i = 1; arr[i] != 0; ++i)   // a[11] == 0 ends the loop
    arr[0] += arr[i];
}

int main() {
  size_t bytes = 12 * sizeof(int);
  int *a = (int *)calloc(1, bytes);
  for (int i=1; i<=10; ++i) a[i]=i;
  int *d_a;
  cudaMalloc(&d_a, bytes);
  cudaMemcpy(d_a, a, bytes,
             cudaMemcpyHostToDevice);
  kernel<<<1, 1>>>(d_a);
  cudaMemcpy(a, d_a, bytes,
             cudaMemcpyDeviceToHost);
  printf("%d\n", a[0]);
  free(a); cudaFree(d_a);
  return 0;
}
