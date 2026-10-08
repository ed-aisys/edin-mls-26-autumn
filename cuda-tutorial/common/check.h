// checkCudaErrors(call): the error-check macro used on the lecture slides.
// Same message format as helper_cuda.h from NVIDIA/cuda-samples:
//   CUDA error at file.cu:LINE code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()"
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define checkCudaErrors(call) do {                                              \
    cudaError_t check_err_ = (call);                                            \
    if (check_err_ != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",        \
                     __FILE__, __LINE__, (int)check_err_,                       \
                     cudaGetErrorName(check_err_), #call);                      \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)
