// Example 1, slides 60-62: two independent chains in two streams, joined by an event.
//   step 1: x = x1 * x1        chain A, stream s1
//   step 2: y = x2 * x2        chain B, stream s2
//   step 3: z = x * sin(x1)    chain A
//   step 4: w = y * cos(x2)    chain B
//   step 5: z = z + w          waits for both chains
#include <cmath>
#include <cstdio>
#include <vector>
#include "../common/check.h"

__global__ void square(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * in[i];
}
__global__ void left(const float *x1, const float *x, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] * sinf(x1[i]);
}
__global__ void right(const float *x2, const float *y, float *w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] = y[i] * cosf(x2[i]);
}
__global__ void combine(float *z, const float *w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = z[i] + w[i];
}

int main() {
    const int n = 1 << 20, b = 256, g = (n + b - 1) / b;
    const size_t bytes = n * sizeof(float);
    std::vector<float> h_x1(n), h_x2(n), h_z(n);
    for (int i = 0; i < n; ++i) {
        h_x1[i] = (i % 100) * 0.01f;
        h_x2[i] = (i % 37) * 0.02f;
    }
    float *x1, *x2, *x, *y, *z, *w;
    for (float **p : {&x1, &x2, &x, &y, &z, &w})
        checkCudaErrors(cudaMalloc(p, bytes));
    checkCudaErrors(cudaMemcpy(x1, h_x1.data(), bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(x2, h_x2.data(), bytes, cudaMemcpyHostToDevice));

    // ---- slide 61: independent branches ----
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    square<<<g,b,0,s1>>>(x1, x, n);
    square<<<g,b,0,s2>>>(x2, y, n);
    left<<<g,b,0,s1>>>(x1, x, z, n);
    right<<<g,b,0,s2>>>(x2, y, w, n);

    // ---- slide 62: joining the branches ----
    cudaEvent_t ready;
    cudaEventCreate(&ready);
    cudaEventRecord(ready, s2);
    cudaStreamWaitEvent(s1, ready, 0);
    combine<<<g,b,0,s1>>>(z, w, n);
    cudaStreamSynchronize(s1);
    checkCudaErrors(cudaGetLastError());
    // Check every API return value.

    checkCudaErrors(cudaMemcpy(h_z.data(), z, bytes, cudaMemcpyDeviceToHost));
    int failures = 0;
    for (int i = 0; i < n; ++i) {
        float x_ = h_x1[i] * h_x1[i], y_ = h_x2[i] * h_x2[i];
        float ref = x_ * sinf(h_x1[i]) + y_ * cosf(h_x2[i]);
        if (std::fabs(h_z[i] - ref) > 1e-5f * (1 + std::fabs(ref))) ++failures;
    }
    printf("n=%d failures=%d\n%s\n", n, failures, failures ? "FAIL streams" : "PASS streams");

    checkCudaErrors(cudaEventDestroy(ready));
    checkCudaErrors(cudaStreamDestroy(s1));
    checkCudaErrors(cudaStreamDestroy(s2));
    for (float *p : {x1, x2, x, y, z, w})
        checkCudaErrors(cudaFree(p));
    return failures != 0;
}
