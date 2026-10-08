// Slide 48: a device (cudaMalloc) pointer dereferenced on the host.
// Expected: Segmentation fault (core dumped), at the first host write.
#include <cstdio>

__global__ void kernel(int *arr) {
  arr[0] = 0;
  for (int i = 1; arr[i] != 0; ++i)
    arr[0] += arr[i];
}

int main() {
  int *a;
  cudaMalloc(&a, 12 * sizeof(int));
  // a now points into GPU memory
  for (int i = 1; i <= 10; ++i)
    a[i] = i;           // host write
  kernel<<<1, 1>>>(a);
  printf("%d\n", a[0]);  // host read
  return 0;
}
