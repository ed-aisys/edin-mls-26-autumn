// Task 2, part A: reduction building blocks. Read docs/task2-reduce.md first.
//
// A reduction turns many numbers into one (a sum, a max, ...). On a GPU we do it
// in two levels:
//   1. inside a warp: the 32 lanes swap registers with __shfl_xor_sync, no memory
//   2. across the warps of a block: each warp leaves one value in shared memory
//
// These helpers are used by task2_reduce.cu (RMSNorm, LayerNorm, softmax) and by
// task4_attention.cu (the online softmax). Every lane of the warp, or every thread
// of the block, must call them; they all get the result back.
#pragma once
#include "common.cuh"

// Sum of v over the 32 lanes of a warp. Every lane gets the total.
__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    // ======================== BLANK 2.1 =========================
    // One step of the butterfly sum (see the guide).
    // YOUR CODE HERE
    v += __shfl_xor_sync(FULL_MASK, v, offset);
    // ============================================================
  }
  return v;
}

// Max of v over the 32 lanes of a warp. Every lane gets the result.
__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    // ======================== BLANK 2.2 =========================
    // The same for the maximum.
    // YOUR CODE HERE
    v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, offset));
    // ============================================================
  }
  return v;
}

// Sum of v over all threads of the block. Every thread gets the total.
// blockDim.x must be a multiple of 32 and at most 1024.
// (Worked example: block_reduce_max below follows the same four steps.)
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float partial[WARP_SIZE];  // one slot per warp (at most 32 warps)
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  v = warp_reduce_sum(v);            // 1. each warp adds up its own 32 values
  if (lane == 0) partial[warp] = v;  // 2. lane 0 of each warp stores the warp's sum
  __syncthreads();                   //    wait until every warp has stored
  v = (lane < num_warps) ? partial[lane] : 0.0f;  // 3. every warp loads all warp sums
  v = warp_reduce_sum(v);                         //    and adds them up again
  __syncthreads();  // 4. a second call reuses partial[]; nobody may write it until all have read
  return v;
}

// Max of v over all threads of the block. Every thread gets the result.
__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float partial[WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  // ======================== BLANK 2.3 =========================
  // Block-wide maximum, built like block_reduce_sum above.
  // YOUR CODE HERE
  v = warp_reduce_max(v);                // 1. each warp reduces its 32 values
  if (lane == 0) partial[warp] = v;     // 2. lane 0 stores warp's max
  __syncthreads();                       //    wait for all warps
  v = (lane < num_warps) ? partial[lane] : -INFINITY;  // 3. load warp maxima (others get -inf)
  v = warp_reduce_max(v);                               //    reduce again
  __syncthreads();                       // 4. barrier for next use of partial[]
  return v;
  // ============================================================
  return v;
}

// Online softmax.
// A MaxSum summarises a set of numbers x_1..x_n without keeping them:
//     m = max(x_i)          l = sum(exp(x_i - m))
// The summary of a single number x is {x, 1}. The empty set is {-INFINITY, 0}.
// Two summaries can be merged without looking at the numbers again, because
// exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new).
struct MaxSum {
  float m;
  float l;
};

__device__ __forceinline__ MaxSum maxsum_merge(MaxSum a, MaxSum b) {
  MaxSum r = {-INFINITY, 0.0f};
  // ======================== BLANK 2.4 =========================
  // Merge a and b with the rule in the guide. Merging two empty summaries
  // ({-INFINITY, 0}) must give an empty summary, not NaN.
  // YOUR CODE HERE
  if (a.m == -INFINITY && a.l == 0.0f) return b;
  if (b.m == -INFINITY && b.l == 0.0f) return a;
  float new_m = fmaxf(a.m, b.m);
  float new_l = a.l * expf(a.m - new_m) + b.l * expf(b.m - new_m);
  r = {new_m, new_l};
  // ============================================================
  return r;
}
