// Task 2, part B: row-wise kernels built on the helpers in task2_reduce.cuh.
// Read docs/task2-reduce.md first.
//
// All three kernels work on a [rows, cols] matrix and use one block per row.
// The threads of a block walk along the row with a block-stride loop:
//
//     for (int c = threadIdx.x; c < cols; c += blockDim.x) ...
//
// so thread 0 handles columns 0, 256, 512, ..., thread 1 handles 1, 257, ...
// Neighbouring threads read neighbouring addresses, and any cols works.
#include "asr_kernels.h"
#include "task2_reduce.cuh"

constexpr int REDUCE_THREADS = 256;  // threads per block: a multiple of 32, at most 1024

// Worked example. LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias,
                                 float* y, int cols, float eps) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;  // this block's row
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  float sum = 0.0f;  // 1. mean
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += xr[c];
  const float mean = block_reduce_sum(sum) / cols;

  float sq = 0.0f;   // 2. variance
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float d = xr[c] - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_reduce_sum(sq) / cols + eps);  // 1 / sqrt(var + eps)

  for (int c = threadIdx.x; c < cols; c += blockDim.x)  // 3. normalise and write
    yr[c] = (xr[c] - mean) * rstd * weight[c] + bias[c];
}

// RMSNorm (text decoder): y = x / sqrt(mean(x^2) + eps) * weight
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y, int cols,
                               float eps) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  // ======================== BLANK 2.5 =========================
  // RMSNorm of this row: y = x / sqrt(mean(x^2) + eps) * weight.
  // YOUR CODE HERE
  float sum_sq = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      float v = xr[c];
      sum_sq += v * v;
  }
  float rms = rsqrtf(block_reduce_sum(sum_sq) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      yr[c] = xr[c] * rms * weight[c];
  }
  // ============================================================
}

// Softmax over each row: y = exp(x - max) / sum(exp(x - max)).
// The textbook version reads the row three times (max, sum, write). With the
// online-softmax MaxSum from part A, the first two passes become one.
__global__ void softmax_kernel(const float* x, float* y, int cols) {
  const float* xr = x + static_cast<size_t>(blockIdx.x) * cols;
  float* yr = y + static_cast<size_t>(blockIdx.x) * cols;

  float row_max = 0.0f;  // max of the whole row
  float row_sum = 1.0f;  // sum of exp(x - row_max) over the whole row
  // ======================== BLANK 2.6 =========================
  // Pass 1: set row_max and row_sum (the sum of exp(x - row_max) over the row),
  // reading x only once, with MaxSum.
  // YOUR CODE HERE
  MaxSum local = {-INFINITY, 0.0f};
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      MaxSum single = {xr[c], 1.0f};
      local = maxsum_merge(local, single);
  }
  MaxSum global = block_reduce_maxsum(local);
  row_max = global.m;
  row_sum = global.l;
  // ============================================================

  // ======================== BLANK 2.7 =========================
  // Pass 2: y = exp(x - row_max) / row_sum.
  // YOUR CODE HERE
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      yr[c] = expf(xr[c] - row_max) / row_sum;
  }
  // ============================================================
}

ASR_API int launch_rmsnorm(const float* x, const float* weight, float* y, int rows,
                           int cols, float eps, cudaStream_t stream) {
  rmsnorm_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, weight, y, cols, eps);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_layernorm(const float* x, const float* weight, const float* bias,
                             float* y, int rows, int cols, float eps, cudaStream_t stream) {
  layernorm_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, weight, bias, y, cols, eps);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_softmax(const float* x, float* y, int rows, int cols,
                           cudaStream_t stream) {
  softmax_kernel<<<rows, REDUCE_THREADS, 0, stream>>>(x, y, cols);
  RETURN_LAUNCH_ERROR();
}
