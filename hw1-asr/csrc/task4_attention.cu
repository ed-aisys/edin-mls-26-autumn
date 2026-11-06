// Task 4: attention with an online softmax (the idea behind FlashAttention).
// Read docs/task4-attention.md first.
//
// For one query vector q (one token, one head) and the keys/values of the same
// head:
//     score_j = (q . k_j) * scale
//     p_j     = softmax(score)_j
//     out     = sum_j p_j * v_j
//
// The plain way writes the whole [seq_q, seq_k] score matrix to GPU memory and
// reads it back for the softmax and again for p * V. Here nothing but the output
// goes back to memory. Keys and values are streamed through shared memory one
// tile of 32 at a time, and each query keeps a running max m, a running sum l
// and a running output acc (the MaxSum idea from task 2).
//
// Work split:
//   block  = ATTN_WARPS consecutive queries of one head of one batch element
//   warp   = one query
//   lane   = during scoring: one key of the current tile (KV_TILE = 32 = warp size)
//            during p * V:   output dimensions lane, lane + 32, lane + 64, lane + 96
//
// Layouts (as produced by the projections, no transposes):
//   q, out: [batch, seq_q, num_heads,    head_dim]
//   k, v:   [batch, seq_k, num_kv_heads, head_dim]
// The text decoder uses grouped-query attention (GQA): 16 query heads share 4
// K/V heads. The audio encoder has num_kv_heads == num_heads.
#include "asr_kernels.h"
#include "task2_reduce.cuh"  // warp_reduce_max, warp_reduce_sum from task 2

constexpr int ATTN_WARPS = 4;          // queries per block; try 2, 8 or 16
constexpr int KV_TILE = WARP_SIZE;     // keys per tile: one per lane
constexpr int MAX_HEAD_DIM = 128;      // head_dim must be a multiple of 32, at most this
constexpr int MAX_DIMS_PER_LANE = MAX_HEAD_DIM / WARP_SIZE;

__global__ void attention_kernel(const float* q, const float* k, const float* v, float* out,
                                 int num_heads, int num_kv_heads, int seq_q, int seq_k,
                                 int head_dim, float scale, int causal) {
  // Dynamic shared memory, sized by the launcher:
  extern __shared__ float smem[];
  const int k_pitch = head_dim + 1;              // +1 avoids bank conflicts when lanes read rows
  float* k_tile = smem;                          // [KV_TILE][k_pitch]
  float* v_tile = k_tile + KV_TILE * k_pitch;    // [KV_TILE][head_dim]
  float* q_all = v_tile + KV_TILE * head_dim;    // [ATTN_WARPS][head_dim]

  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int h = blockIdx.y;                      // query head
  const int b = blockIdx.z;                      // batch element
  const int qi = blockIdx.x * ATTN_WARPS + warp; // this warp's query
  const bool active = qi < seq_q;                // the last block may have spare warps

  int kv_h = 0;  // the K/V head this query head reads
  // ======================== BLANK 4.1 =========================
  // Set kv_h, the K/V head that query head h reads (GQA: groups of consecutive
  // query heads share one K/V head).
  // YOUR CODE HERE
  kv_h = h * num_kv_heads / num_heads;
  // ============================================================

  const size_t q_stride = static_cast<size_t>(num_heads) * head_dim;      // token to token in q, out
  const size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;  // token to token in k, v
  const float* k_head = k + static_cast<size_t>(b) * seq_k * kv_stride + kv_h * head_dim;
  const float* v_head = v + static_cast<size_t>(b) * seq_k * kv_stride + kv_h * head_dim;
  // key j of this head: k_head + j * kv_stride, value j: v_head + j * kv_stride

  // Each warp copies its query, already multiplied by scale, into shared memory.
  float* qv = q_all + warp * head_dim;
  if (active) {
    const float* q_row = q + (static_cast<size_t>(b) * seq_q + qi) * q_stride + h * head_dim;
    for (int d = lane; d < head_dim; d += WARP_SIZE) qv[d] = q_row[d] * scale;
  }

  // Causal masking: query qi sits at position qi + (seq_k - seq_q) of the key
  // sequence and may only see keys up to that position.
  const int shift = seq_k - seq_q;
  int last_key = causal ? min(seq_k - 1, qi + shift) : seq_k - 1;
  if (!active) last_key = -1;  // spare warps see nothing
  // The block loops until the last key any of its queries needs. Every thread
  // runs the same number of iterations, so the __syncthreads() inside are safe.
  const int block_last_q = min(seq_q - 1, (blockIdx.x + 1) * ATTN_WARPS - 1);
  const int block_last_key = causal ? min(seq_k - 1, block_last_q + shift) : seq_k - 1;

  const int dims = head_dim / WARP_SIZE;  // output dims per lane (2 or 4 here)
  float m = -INFINITY;                    // running max of the scores seen so far
  float l = 0.0f;                         // running sum of exp(score - m)
  float acc[MAX_DIMS_PER_LANE] = {0.0f};  // running sum of exp(score - m) * v, per lane dim

  for (int j0 = 0; j0 <= block_last_key; j0 += KV_TILE) {
    // ======================== BLANK 4.2 =========================
    // Copy keys and values j0 .. j0+KV_TILE-1 of this head into k_tile (rows of
    // k_pitch floats) and v_tile (rows of head_dim floats), with 0 for keys past
    // seq_k. All threads of the block take part.
    // YOUR CODE HERE
    for (int idx = threadIdx.x; idx < KV_TILE * k_pitch; idx += blockDim.x) {
        int r = idx / k_pitch;
        int c = idx % k_pitch;
        if (c < head_dim) {
            int key_idx = j0 + r;
            if (key_idx < seq_k)
                k_tile[r * k_pitch + c] = k_head[key_idx * kv_stride + c];
            else
                k_tile[r * k_pitch + c] = 0.0f;
        } else {
            k_tile[r * k_pitch + c] = 0.0f;
        }
    }
    for (int idx = threadIdx.x; idx < KV_TILE * head_dim; idx += blockDim.x) {
        int r = idx / head_dim;
        int c = idx % head_dim;
        int key_idx = j0 + r;
        if (key_idx < seq_k)
            v_tile[r * head_dim + c] = v_head[key_idx * kv_stride + c];
        else
            v_tile[r * head_dim + c] = 0.0f;
    }
    __syncthreads();
    // ============================================================
    float s = -INFINITY;  // this lane's score for key j0 + lane
    // ======================== BLANK 4.3 =========================
    // Set s to this query's score for key j0 + lane, if the query may see that
    // key (last_key).
    // YOUR CODE HERE
    if (active && j0 + lane <= last_key) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            sum += qv[d] * k_tile[lane * k_pitch + d];
        s = sum;
    } else {
        s = -INFINITY;
    }
    // ============================================================

    float p = 0.0f;  // this lane's exp(s - m_new): unnormalised probability of key j0 + lane
    // ======================== BLANK 4.4 =========================
    // Update m, l and acc with this tile (guide, section 3) and set p.
    // YOUR CODE HERE
    float tile_max = warp_reduce_max(s);
    float m_new = fmaxf(m, tile_max);
    float corr = (m_new == -INFINITY) ? 1.0f : expf(m - m_new);
    p = (m_new == -INFINITY) ? 0.0f : expf(s - m_new);
    float sum_exp = warp_reduce_sum(p);
    l = l * corr + sum_exp;
    for (int d = 0; d < dims; ++d) acc[d] *= corr;
    m = m_new;
    // ============================================================

    // ======================== BLANK 4.5 =========================
    // Add this tile's values, weighted by p, to acc.
    // YOUR CODE HERE
    for (int r = 0; r < KV_TILE; ++r) {
        float pr = __shfl_sync(FULL_MASK, p, r);
        for (int d = 0; d < dims; ++d) {
            int dim = lane + d * WARP_SIZE;
            acc[d] += pr * v_tile[r * head_dim + dim];
        }
    }
    // ============================================================

    __syncthreads();  // all warps are done with this tile before the next one overwrites it
  }

  // ======================== BLANK 4.6 =========================
  // Write acc / l to out for this warp's query.
  // YOUR CODE HERE
  if (active) {
      float inv_l = 1.0f / l;
      float* out_row = out + (static_cast<size_t>(b) * seq_q + qi) * q_stride + h * head_dim;
      for (int d = 0; d < dims; ++d) {
          int dim = lane + d * WARP_SIZE;
          out_row[dim] = acc[d] * inv_l;
      }
  }
  // ============================================================
}

ASR_API int launch_attention(const float* q, const float* k, const float* v, float* out,
                             int batch, int num_heads, int num_kv_heads, int seq_q,
                             int seq_k, int head_dim, float scale, int causal,
                             cudaStream_t stream) {
  if (head_dim % WARP_SIZE != 0 || head_dim > MAX_HEAD_DIM || num_heads % num_kv_heads != 0)
    return static_cast<int>(cudaErrorInvalidValue);
  size_t smem = sizeof(float) *
                (KV_TILE * (head_dim + 1) + KV_TILE * head_dim + ATTN_WARPS * head_dim);
  if (smem > 48 * 1024) {  // above 48 KB a kernel must ask for more shared memory
    cudaError_t err = cudaFuncSetAttribute(
        attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (err != cudaSuccess) return static_cast<int>(err);
  }
  dim3 grid(ceil_div(seq_q, ATTN_WARPS), num_heads, batch);
  attention_kernel<<<grid, ATTN_WARPS * WARP_SIZE, smem, stream>>>(
      q, k, v, out, num_heads, num_kv_heads, seq_q, seq_k, head_dim, scale, causal);
  RETURN_LAUNCH_ERROR();
}
