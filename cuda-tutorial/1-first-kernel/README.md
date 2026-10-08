# 1. A First CUDA Kernel

Slides 19 and 36–37.

## `hello.cu` (slide 19)

```cuda
__global__ void kernel() {
    printf("Block %d of %d, Thread %d of %d\n",
        blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main() {
    kernel<<<4, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```bash
make 1-first-kernel/hello && ./1-first-kernel/hello
```

`<<<4, 3>>>` launches 4 blocks × 3 threads = 12 threads, so there are 12 lines. Blocks run in any order:

```
Block 3 of 4, Thread 0 of 3
Block 3 of 4, Thread 1 of 3
Block 3 of 4, Thread 2 of 3
Block 1 of 4, Thread 0 of 3
...
```

Questions from the slide:
1. What is a block? What is a thread?
2. What does `<<<4, 3>>>` launch?
3. Why `cudaDeviceSynchronize()`?

## `qualifiers.cu` (slides 36–37)

| Qualifier | Runs on | Called from |
|-----------|---------|-------------|
| `__global__` | GPU | host, with `<<<grid, block>>>`; returns `void`; cannot recurse |
| `__device__` | GPU | device code only (may recurse) |
| `__host__` | CPU | host (the default) |
| `__host__ __device__` | both | both; `__CUDA_ARCH__` is set only in device code |

```bash
make 1-first-kernel/qualifiers && ./1-first-kernel/qualifiers
```

```
gpu hello!
gpu hello!
cpu hello!
```

- `<<<1, 2>>>` = 1 block × 2 threads → two GPU lines.
- The launch is asynchronous. `cudaDeviceSynchronize()` waits for the kernel, and that is also when the device `printf` buffer is printed, so the GPU lines come first.
- Try deleting `cudaDeviceSynchronize()`. The GPU lines may then come after the cpu line, or not be printed at all.
- Device code uses `printf`, not `std::cout`.
