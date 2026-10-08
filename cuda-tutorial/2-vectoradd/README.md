# 2. Example 0: Vector Add

Slides 38–53 (and 83 for the batch job). One thread per element, `c[idx] = a[idx] + b[idx]`.

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one global index per thread
    if (idx < n) {                                     // spare threads skip the work
        c[idx] = a[idx] + b[idx];
    }
}
// gridSize × blockSize ≥ n threads
vector_add<<<gridSize, blockSize>>>(a, b, c, n);
```

## Files

| File | Slide | Run | Expected |
|------|-------|-----|----------|
| `host_path.cu` | 49 | `./host_path` | `3.000000` ten times |
| `pointer_mistake1.cu` | 47 (1) | `./pointer_mistake1` | `CUDA error at pointer_mistake1.cu:21 code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()"` |
| `pointer_mistake2.cu` | 47 (2) | `./pointer_mistake2` | `Segmentation fault (core dumped)` |
| `memcpy.cu` | 48 (left) | `./memcpy` | `55` |
| `managed.cu` | 48 (right) | `./managed` | `55` |
| `vector_add.cu` | 51 | `./vector_add 1003 256` | `PASS vector add` |
| `smoke.sbatch` | 83 | `sbatch smoke.sbatch` | job `COMPLETED`, `PASS` twice in `smoke-JOB_ID.out` |

Build everything with `make` in `cuda-tutorial/`, then run from this directory.

## The complete host path (`host_path.cu`, slides 38 and 49)

1. Allocate host and device buffers (`malloc`, `cudaMalloc`).
2. Copy inputs to the GPU (`cudaMemcpy ... cudaMemcpyHostToDevice`).
3. Launch ⌈n/256⌉ blocks of 256 threads.
4. Copy the results back and check them.
5. Free both sides.

The program is the slide code. As the slide notes say, real code should also check the launch, and the `cudaDeviceSynchronize()` after the blocking `cudaMemcpy` is redundant. `vector_add.cu` does both properly.

## Host and device memory (slides 45–48)

- `pointer_mistake1.cu`: a `malloc` pointer handed to a kernel. The kernel faults; `checkCudaErrors` reports error 700. Without the check, the program silently prints 0, and a wrong answer is worse than a crash. On systems with HMM or ATS (`nvidia-smi -q | grep "Addressing Mode"`), the GPU can read `malloc` memory and the program prints 55. The teaching cluster GPUs report `None`.
- `pointer_mistake2.cu`: `cudaMalloc` fills in a device address. Dereferencing it on the host crashes.
- `memcpy.cu`: explicit copies. `cudaMemcpy` waits for the kernel in the same stream, so no extra sync is needed.
- `managed.cu`: one pointer for both sides. Pages migrate on demand, at a hidden time cost; the host must `cudaDeviceSynchronize()` before reading.

The kernel adds `arr[1]`, `arr[2]`, ... until it reads a 0, so `a[11]` must be 0. `malloc` and `cudaMallocManaged` do not zero memory; the files use `calloc` or set `a[11] = 0`.

## Student checkpoint (`vector_add.cu`, slide 51)

On the GPU node, in this directory:

```bash
nvcc -O2 -std=c++17 -arch=native \
  vector_add.cu -o vector_add
./vector_add 1003 256
./vector_add 257 128
```

Expected output (the first line names your GPU):

```
GPU=NVIDIA GeForce RTX 2080 Ti CC=7.5 SMs=68 warp=32
n=1003 block=256 grid=4
checked=1003 failures=0
PASS vector add
```

Then:
1. Record the node and GPU.
2. Explain the bounds check: 1003 and 257 are not multiples of the block size.
3. Exit the GPU shell.

Why do both configurations produce the same answer?

## Batch job (`smoke.sbatch`, slide 83)

From the head node, in this directory:

```bash
sbatch smoke.sbatch
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode
cat smoke-JOB_ID.out
```

Success means `State COMPLETED`, `ExitCode 0:0` and `PASS` in the output file. "Submitted" alone is not success. `scancel JOB_ID` cancels a job you no longer need.
