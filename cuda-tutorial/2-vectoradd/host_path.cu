// Slides 38 and 49: Example 0, vector add, the complete host path.
// The code below is the slide code; only n and size are added (the slide leaves them out).
// As the slide notes say: real code should also check the launch (see vector_add.cu),
// and the cudaDeviceSynchronize() after the blocking cudaMemcpy is redundant.
#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    // Host memory
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize host data
    for(int i = 0; i < n; i++){
        h_a[i] = 1.0f; h_b[i] = 2.0f;
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy Result device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // Sync
    cudaDeviceSynchronize();

    // print first 10 elements of h_c
    for(int i = 0; i < 10; i++){
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
