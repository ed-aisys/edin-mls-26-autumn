// Environment check: which GPU did the allocation give you, and what are its limits?
// The limits are the ones on slide 44 ("Grid size, block size and residency").
#include <cstdio>
#include "../common/check.h"

__global__ void ping(int *out) { *out = 42; }

int main() {
    int runtime = 0, driver = 0, device = 0;
    checkCudaErrors(cudaRuntimeGetVersion(&runtime));
    checkCudaErrors(cudaDriverGetVersion(&driver));
    checkCudaErrors(cudaGetDevice(&device));

    cudaDeviceProp p;
    checkCudaErrors(cudaGetDeviceProperties(&p, device));
    printf("GPU             %s\n", p.name);
    printf("compute cap.    %d.%d (sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
    printf("CUDA runtime    %d.%d, driver supports up to %d.%d\n",
           runtime / 1000, runtime % 1000 / 10, driver / 1000, driver % 1000 / 10);
    printf("SMs             %d\n", p.multiProcessorCount);
    printf("memory          %.1f GiB\n", p.totalGlobalMem / double(1 << 30));
    printf("threads / SM    %d\n", p.maxThreadsPerMultiProcessor);
    printf("threads / block %d\n", p.maxThreadsPerBlock);
    printf("block (x,y,z)   %d, %d, %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("grid  (x,y,z)   %d, %d, %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

    // A kernel that actually runs: compiled for this GPU and launched successfully.
    int *out;
    checkCudaErrors(cudaMallocManaged(&out, sizeof(int)));
    ping<<<1, 1>>>(out);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("kernel launch   %s\n", *out == 42 ? "OK" : "FAILED");
    checkCudaErrors(cudaFree(out));
    return 0;
}
