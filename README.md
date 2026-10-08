# EDIN MLS 2026 Autumn

Welcome to the Machine Learning Systems course at the University of Edinburgh (Autumn 2026). This repository holds the course's code. The first part is the CUDA C++ examples from the lecture *GPU Architecture and CUDA Programming*. Every example is the code shown on the slides, as a complete program you can build and run on the teaching cluster.

More material (tutorials, assignments) will be added during the term.

## Quick Start

The commands are the ones on slides 29–33 and 53.

```bash
# 1. Log in: two ssh hops (campus network or approved VPN)
ssh YOUR_UUN@student.ssh.inf.ed.ac.uk
ssh icf.inf.ed.ac.uk

# 2. Clone the repository on the head node
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
cd edin-mls-26-autumn/cuda-tutorial

# 3. Ask Slurm for a GPU shell (the head node has no GPU)
srun -p Teaching --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:40:00 --pty bash

# 4. On the compute node: put nvcc on PATH, build and run
export PATH=/opt/cuda-12.8.0/bin:$PATH
make
cd 1-first-kernel && ./hello
```

Leave the GPU shell with `exit` when you are done, so the GPU goes back to the pool.

## Repository Structure

```
edin-mls-26-autumn/
└── cuda-tutorial/                 # CUDA C++ examples from the lecture (slide numbers in each README)
    ├── 0-environment/             # Teaching cluster: login and Slurm allocation (commands from the slides)
    ├── 1-first-kernel/            # First kernel, function qualifiers
    ├── 2-vectoradd/               # Example 0: vector add, host/device memory, student checkpoint
    ├── 3-streams/                 # Example 1: CUDA streams and event dependencies
    ├── 4-shared-memory/           # Example 2: matrix-vector multiply, explicit shared storage
    ├── 5-tiled-gemm/              # Example 3: tiled GEMM, block barriers, event timing
    ├── 6-optional-thrust-atomics/ # Optional: thrust containers, atomics
    ├── common/check.h             # checkCudaErrors(), as used on the slides
    └── Makefile                   # builds every example with nvcc -arch=native
```

## Learning Path

| Phase | What to do | Slides | Goal |
|-------|------------|--------|------|
| **1. Setup** | `0-environment`: log in, get a GPU shell, check node, GPU and `nvcc` | 28–33 | A working GPU allocation |
| **2. First kernel** | `1-first-kernel` | 19, 36–37 | Grid, block, thread; host vs device code |
| **3. Example 0** | `2-vectoradd`, then the checkpoint `vector_add.cu` | 38–55 | Indexing, memory, launch configuration |
| **4. Example 1** | `3-streams` | 61–67 | Streams, events, overlapping copies |
| **5. Example 2** | `4-shared-memory` | 68–78 | Data reuse and `__shared__` |
| **6. Example 3** | `5-tiled-gemm` | 79–84, 87 | Tiling, barriers, timing against a baseline |
| **Optional** | `6-optional-thrust-atomics` | 57–60 | thrust, atomics |

## GPU Compatibility

The examples are plain CUDA C++ and need CUDA 11.6 or newer (for `-arch=native`). They were tested on the Teaching partition with CUDA 12.8 on NVIDIA GeForce RTX 2080 Ti (sm_75) and NVIDIA H200 MIG 1g.18gb (sm_90). Your allocation may give you a different GPU. Build on the compute node, so that `-arch=native` matches the GPU you were given.

## Resources

- **Lecture slides**: *GPU Architecture and CUDA Programming* (MLSys 2026, CUDA part). Each example README lists the slides it belongs to.
- **Teaching cluster**: [computing.help.inf.ed.ac.uk/teaching-cluster](https://computing.help.inf.ed.ac.uk/teaching-cluster), [Slurm](https://computing.help.inf.ed.ac.uk/slurm), [cluster tips](https://computing.help.inf.ed.ac.uk/cluster-tips)
- **CUDA**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvidia-smi` or the program finds no GPU | You are on the head node. Run `srun ... --pty bash` first. |
| `nvcc: command not found` | `export PATH=/opt/cuda-12.8.0/bin:$PATH` on the compute node. |
| `nvcc fatal: Unsupported gpu architecture 'native'` / no GPU at compile time | Build on the compute node, or pass a target: `make ARCH=-arch=sm_75`. |
| `srun` waits for a long time | The partition is busy; `sinfo -p Teaching` shows its state but does not reserve a slot. |
| `cudaErrorUnsupportedPtxVersion` | The binary was built for another GPU or toolkit. Run `make clean && make` on the node you run on. |

## License

CC0 1.0 Universal. See [LICENSE](LICENSE).
