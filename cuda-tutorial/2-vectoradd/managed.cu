// Slide 48, right: managed memory with cudaMallocManaged. Prints 55.
#include <cstdio>

__global__ void kernel(int *arr) {
  arr[0] = 0;
  for (int i = 1; arr[i] != 0; ++i)   // a[11] == 0 ends the loop
    arr[0] += arr[i];
}

int main() {
  size_t bytes = 12 * sizeof(int);
  int *a;
  cudaMallocManaged(&a, bytes);
  for (int i=1; i<=10; ++i) a[i]=i;
  a[11] = 0;          // end marker
  kernel<<<1, 1>>>(a);
  cudaDeviceSynchronize();
  printf("%d\n", a[0]);   // 55
  cudaFree(a);
  return 0;
}
