# 5. Example 3: Tiled GEMM

Slides 78–83 and 86. `C = A × B` with `A: M × K`, `B: K × N`, row-major. `gemm.cu` runs the tiled kernel from the slides (16 × 16 tiles of A and B in shared memory), checks it against a CPU reference and times it with CUDA events.

## Tiling and the two barriers (slides 81–82)

Each block loads an A tile and a B tile cooperatively, synchronizes, computes, and moves along K. For one 16 × 16 tile pair, 512 input values serve 4,096 multiply-adds.

```cuda
// Every thread loads a valid value or zero.
as[ty][tx] = valid_a ? a[a_index] : 0;
bs[ty][tx] = valid_b ? b[b_index] : 0;
__syncthreads();                 // 1: data is ready before any thread consumes it
for (int q = 0; q < TILE; ++q)
  acc += as[ty][q] * bs[q][tx];
__syncthreads();                 // 2: consumers finish before the tile is overwritten
```

Edge tiles load zeros. Never return early: every thread must reach both barriers.

## Timing GPU work with events (slide 83)

```cuda
cudaEventRecord(start, stream);
// Launch the measured work in this stream.
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

A CPU timer around an asynchronous launch mostly measures the submission. Measure honestly: warm up, repeat, check the results, and state what the interval includes (here: the kernel only, no copies).

```
$ ./gemm
M=1000 N=777 K=513
tiled 0.596 ms per launch (kernel only)  failures=0  PASS
```

`./gemm M N K` picks other sizes. Exercise (slide 86, task 2): a non-square GEMM exposes swapped row and column indices. Swap them on purpose and watch `failures`. A handwritten kernel teaches indexing and reuse; it does not generally beat cuBLAS, which is the baseline to compare with (slides 78–79).
