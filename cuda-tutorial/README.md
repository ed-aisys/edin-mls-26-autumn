# CUDA Tutorial

The CUDA C++ examples from the lecture *GPU Architecture and CUDA Programming*. Each file is the slide code, completed into a program that builds and runs. Where a slide shows only a fragment, the file contains that fragment unchanged, plus the setup and a CPU check around it.

## Build and Run

Build on a GPU compute node (see [`0-environment`](0-environment/README.md)), not on the head node:

```bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
make                       # every example, nvcc -O2 -std=c++17 -arch=native
make 2-vectoradd/memcpy    # one example
make ARCH=-arch=sm_75      # a fixed GPU target instead of the local GPU
make clean
```

Each binary is built next to its source; run it from that directory, e.g. `cd 2-vectoradd && ./memcpy`. You can also compile a single file by hand, as on slide 51:

```bash
nvcc -O2 -std=c++17 -arch=native vector_add.cu -o vector_add
```

The thrust and atomics examples pass device lambdas to kernels and need `--extended-lambda`.

## Examples and Slides

| Directory | File | Slides | What it shows |
|-----------|------|--------|---------------|
| `0-environment` | `README.md` | 28–33 | Login and Slurm commands from the slides (no code) |
| `1-first-kernel` | `hello.cu` | 19 | `<<<4, 3>>>`: 4 blocks × 3 threads |
| | `qualifiers.cu` | 36–37 | `__global__`, `__device__`, `__host__`; asynchronous launch |
| `2-vectoradd` (Example 0) | `host_path.cu` | 38, 49 | The complete host path: allocate, copy, launch, copy back, free |
| | `pointer_mistake1.cu` | 47 | Host pointer in a kernel → `cudaErrorIllegalAddress` |
| | `pointer_mistake2.cu` | 47 | Device pointer on the host → segmentation fault |
| | `memcpy.cu`, `managed.cu` | 48 | Explicit copies vs managed memory |
| | `vector_add.cu`, `smoke.sbatch` | 51, 83 | Student checkpoint: checked vector add, batch job |
| `3-streams` (Example 1) | `branches.cu` | 60–62 | Two chains in two streams, joined by an event |
| | `batches.cu` | 63–65 | Batched copies and kernels overlapping across streams |
| `4-shared-memory` (Example 2) | `matvec.cu` | 66–69, 75–76 | Matrix-vector multiply: per-row, grid-stride, shared tiles of x |
| `5-tiled-gemm` (Example 3) | `gemm.cu` | 79–82, 85 | Tiled GEMM, two barriers, event timing |
| `6-optional-thrust-atomics` | `thrust_vector_add.cu` | 55–56 | thrust `host_vector` / `device_vector` |
| | `atomic_race.cu` | 57 | A data race on one `__device__` variable |
| | `atomic_add.cu` | 58 | `atomicAdd` and an `atomicCAS` loop |

`common/check.h` defines `checkCudaErrors(call)`, the check used on the slides. It prints `CUDA error at file.cu:LINE code=N(name) "call"` and exits.

## Tested On

Teaching partition, CUDA 12.8.93: NVIDIA GeForce RTX 2080 Ti (opencast, sm_75, driver 580) and NVIDIA H200 MIG 1g.18gb (saxa, sm_90, driver 595). All examples build, and every check passes. Timings differ between GPUs and runs; the timing printouts are illustrations, not benchmarks.
