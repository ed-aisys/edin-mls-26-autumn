# 4. Example 2: Matrix-Vector Multiplication and Shared Memory

Slides 67–77. `y = A x` with `A: M × K`, `x: K`, `y: M`. One output is one row of A dotted with x:

```cuda
for (int row = 0; row < M; ++row) {      // slide 67: CPU reference
  float sum = 0.0f;
  for (int k = 0; k < K; ++k)
    sum += A[row*K+k] * x[k];
  y[row] = sum;
}
```

`matvec.cu` runs three GPU versions and checks each against this reference. M = 4099 and K = 1000 are not multiples of the block size, so edge rows and edge tiles are exercised.

| Kernel | Slide | Idea |
|--------|-------|------|
| `matvec` | 68 | One thread computes one output row; spare threads in the last block exit |
| `matvec_stride` | 69 | Grid-stride loop: rows `first, first + stride, …`, so any grid size covers all M rows |
| `matvec_shared` | 70 | Tiles of `x` are loaded into `__shared__ float sx[TILE]` once per block and reused by all its rows |

Every row reads all of `x`, the input that is reused. A new work assignment (slide 69) is not data reuse. Shared memory must be loaded explicitly:

```cuda
__shared__ float sx[TILE];
// for each tile of x (all threads):
sx[threadIdx.x] = k < K ? x[k] : 0.0f;
__syncthreads();
if (row < M) {
  // accumulate the valid tile into sum
}
__syncthreads();
// next tile only after readers finish
```

1. **Load.** Every thread loads one element of x, even if `row ≥ M`.
2. **Sync.** Wait until the whole tile is in shared memory.
3. **Use.** Only valid rows accumulate. There is no early return, because every thread of the block must reach both `__syncthreads()`.
4. **Sync again.** Do not overwrite the tile while other threads still read it.

```
$ ./matvec
matvec         failures=0  PASS
matvec_stride  failures=0  PASS
matvec_shared  failures=0  PASS
```

Slides 71–76 compare this with a cache. A cache manages reuse in hardware and may evict `x`; shared memory keeps it until the program overwrites it.
