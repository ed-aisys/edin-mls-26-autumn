# 6. Optional: thrust and Atomics

Slides 56–59. All three programs pass a device lambda to a grid-stride kernel and are built with `--extended-lambda` (the Makefile adds it).

```cuda
template <class F>
__global__ void kernel(int n, F f) {
  int i = blockIdx.x * blockDim.x
        + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  for (; i < n; i += step) f(i);
}
```

## `thrust_vector_add.cu` (slides 56–57)

thrust is the STL-style template library that ships with the CUDA toolkit (part of CCCL). `host_vector` and `device_vector` hold the data; an assignment such as `x_d = x_h` runs the `cudaMemcpy` for you. The same vector add as Example 0, two copies and one kernel, without a single `cudaMalloc`.

```
$ ./thrust_vector_add
752851072.000000
```

## `atomic_race.cu` (slide 58)

Every thread does `sum += sinf(arr[i])` on one `__device__ float`. `sum += x` is load, add, store: threads interleave and overwrite each other. Which thread wins is undefined. This is a data race, not a rounding problem. A `__device__` variable is read back with `cudaMemcpyFromSymbol`.

```
$ ./atomic_race                  # different on every run
GPU -0.999825
CPU 1.229885
```

## `atomic_add.cu` (slide 59)

`atomicAdd` does the read-modify-write in one step. `my_atom_add` builds the same operation from `atomicCAS(addr, expect, new)`, which stores `new` only if `*addr == expect`. `atomicCAS` works on integer types only, hence `__float_as_int`. Both results now differ from the CPU only by summation order:

```
$ ./atomic_add
GPU atomicAdd   1.229948
GPU my_atom_add 1.229817
CPU             1.229885
```

Atomics on one address serialise. For a large reduction, sum inside each block first (shared memory), then do one `atomicAdd` per block.
