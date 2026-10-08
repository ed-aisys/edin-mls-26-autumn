// Example 1, slides 65-67: copies and computation across batches.
// Each batch runs the slide-65 sequence (H2D copy, kernel, D2H copy) in its own stream,
// from pinned host memory, so batch k+1's copy can overlap batch k's kernel.
// The same work in one stream is timed for comparison. The gain depends on the
// copy/compute ratio and on how many copy engines the GPU has.
#include <cmath>
#include <cstdio>
#include "../common/check.h"

__global__ void kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        for (int k = 0; k < 200; ++k)   // enough work to be comparable to the copies
            v = sinf(v) + 1.0f;
        out[i] = v;
    }
}

// Issue all batches; batch b goes to streams[b % num_streams].
void run(const float *h_in_all, float *h_out_all, float *d_in_all, float *d_out_all,
         int total, int batches, cudaStream_t *streams, int num_streams) {
    int n = total / batches, block = 256, grid = (n + block - 1) / block;
    size_t bytes = n * sizeof(float);
    for (int b = 0; b < batches; ++b) {
        cudaStream_t s = streams[b % num_streams];
        const float *h_in = h_in_all + (size_t)b * n;
        float *h_out = h_out_all + (size_t)b * n;
        float *d_in = d_in_all + (size_t)b * n, *d_out = d_out_all + (size_t)b * n;
        // ---- slide 67: one batch ----
        cudaMemcpyAsync(d_in, h_in, bytes,
            cudaMemcpyHostToDevice, s);
        kernel<<<grid, block, 0, s>>>(
            d_in, d_out, n);
        cudaMemcpyAsync(h_out, d_out, bytes,
            cudaMemcpyDeviceToHost, s);
    }
    for (int k = 0; k < num_streams; ++k) {   // host waits once per stream
        cudaStream_t s = streams[k];
        cudaStreamSynchronize(s);
    }
    checkCudaErrors(cudaGetLastError());
}

int main() {
    const int total = 1 << 24, batches = 8;
    const size_t bytes = total * sizeof(float);
    float *h_in, *h_out, *d_in, *d_out;
    checkCudaErrors(cudaMallocHost(&h_in, bytes));    // pinned host memory
    checkCudaErrors(cudaMallocHost(&h_out, bytes));
    checkCudaErrors(cudaMalloc(&d_in, bytes));
    checkCudaErrors(cudaMalloc(&d_out, bytes));
    for (int i = 0; i < total; ++i)
        h_in[i] = (i % 1000) * 0.001f;

    cudaStream_t streams[batches];
    for (auto &s : streams)
        checkCudaErrors(cudaStreamCreate(&s));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    run(h_in, h_out, d_in, d_out, total, batches, streams, batches);   // warm-up
    for (int num_streams : {1, batches}) {
        checkCudaErrors(cudaEventRecord(start));
        run(h_in, h_out, d_in, d_out, total, batches, streams, num_streams);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float ms;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        printf("%d batches in %d stream(s): %7.2f ms\n", batches, num_streams, ms);
    }

    int failures = 0;
    for (int i = 0; i < total; i += 997) {
        float v = h_in[i];
        for (int k = 0; k < 200; ++k)
            v = sinf(v) + 1.0f;
        if (std::fabs(h_out[i] - v) > 1e-4f) ++failures;
    }
    printf("%s\n", failures ? "FAIL batches" : "PASS batches");

    for (auto s : streams)
        checkCudaErrors(cudaStreamDestroy(s));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFreeHost(h_in));
    checkCudaErrors(cudaFreeHost(h_out));
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    return failures != 0;
}
