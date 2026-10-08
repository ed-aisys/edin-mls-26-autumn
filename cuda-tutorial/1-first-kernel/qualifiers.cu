// Slide 38: "Function qualifiers: one program".
#include <cstdio>

__device__ void gpu_hello() {
    printf("gpu hello!\n");
}

__host__ void cpu_hello() {
    printf("cpu hello!\n");
}

__global__ void kernel() {
    gpu_hello();
}

int main() {
    kernel<<<1, 2>>>();
    cudaDeviceSynchronize();
    cpu_hello();
}
