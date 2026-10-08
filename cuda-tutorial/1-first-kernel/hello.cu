// Slides 19 and 34: "A first CUDA kernel".
#include <cstdio>

__global__ void kernel() {
    printf("Block %d of %d, Thread %d of %d\n",
        blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main() {
    kernel<<<4, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
