# Homework 1: speech recognition on your own CUDA kernels

In this homework you write the GPU kernels that a real speech recognition model runs on. The model is GLM-ASR-Nano, a network with about 2.3 billion parameters. It listens to a 3.5-second recording (`test_audio.wav`) and writes down what was said:

```
Transcription: Concord returned to its place amidst the tents.
```

Every heavy computation in the model goes through four CUDA files that you complete: all the matrix multiplications, the normalisations, and attention. PyTorch only holds the arrays in GPU memory and glues the steps together. When the four tasks are done, the model transcribes the recording correctly on your code.

You do not need to know anything about speech recognition or neural networks; [docs/model-tour.md](docs/model-tour.md) explains the parts of the model you will meet. You do need the CUDA from the lecture, lessons 1 to 5 of [cuda-tutorial](../cuda-tutorial): grids, blocks and threads, vector add, shared memory, and the tiled GEMM. If those feel shaky, run through them again first.

## The four tasks

The kernels are fill-in-the-blank. Each task file contains numbered blanks such as:

```cuda
for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
  // ======================== BLANK 2.1 =========================
  // Add the value held by the lane whose id differs from yours in bit `offset`.
  // __shfl_xor_sync(FULL_MASK, v, offset) returns that lane's v.
  // YOUR CODE HERE
  // ============================================================
}
```

Replace `// YOUR CODE HERE` with your code. A blank needs between one and about fifteen lines. The code around the blanks, the Python side and the tests are already written.

| Task | Files you edit | What you write | Guide | Test |
|------|----------------|----------------|-------|------|
| 1. RoPE | `csrc/task1_rope.cu` | an element-wise kernel that rotates the query and key vectors | [task1-rope.md](docs/task1-rope.md) | `make test1` |
| 2. Reductions | `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu` | warp and block reductions, then RMSNorm and a one-pass softmax built on them | [task2-reduce.md](docs/task2-reduce.md) | `make test2` |
| 3. Tiled GEMM | `csrc/task3_gemm.cu` | the linear layer: a shared-memory tiled matrix multiply with bias and GELU | [task3-gemm.md](docs/task3-gemm.md) | `make test3` |
| 4. Attention | `csrc/task4_attention.cu` | attention that streams keys and values through shared memory with an online softmax | [task4-attention.md](docs/task4-attention.md) | `make test4` |

Do the tasks in order. Each one builds on the one before, and task 4 calls the warp reductions you write in task 2.

After every change to a file in `csrc/`, run `make` on a GPU node. It compiles your kernels and the tests, and it is where you see your compiler errors.

Recall the memory hierarchy from the lecture. Global memory (the GPU's main memory) is large and slow. Shared memory is small and fast and belongs to one block, and registers are the fastest and belong to one thread. In all four tasks, a fast kernel loads each value from global memory as few times as it can and does the repeated work in shared memory and registers. The tests print how close each kernel gets to the GPU's limits.

## What is in this folder

| Folder or file | What it is for |
|----------------|----------------|
| `csrc/` | The CUDA code. You fill in the blanks of the five `task*` files. `common.cuh` has small helpers, `asr_kernels.h` lists the C functions that Python calls, and `provided.cu` holds kernels that are already finished. |
| `tests/` | One test program per task (`make test1` to `make test4`). Each runs your kernel on the model's shapes, compares it with a CPU version and times it. |
| `docs/` | A guide for each task, and a [tour of the model](docs/model-tour.md). |
| `glm_asr_cuda/` | The Python side. `model.py` is the model; `ops.py` runs each operation on your kernel or on PyTorch. You do not need to change them. |
| `build/` | Created by `make`: the compiled library and tests, one folder per kind of GPU. Safe to delete. |
| `Makefile` | Builds `csrc/` and `tests/` into `build/`. |
| `run.sh`, `benchmark.py` | Transcribe `test_audio.wav` with your kernels and check the result. |
| `setup_env.sh` | One-time setup: the conda environment and the model download. |
| `run_asr.sbatch` | The tests and the model as a batch job. |
| `test_audio.wav`, `test_audio.txt` | The recording and what is said in it. |

## Step 1: set up, once

Log in to the cluster's head node, `mlp.inf.ed.ac.uk`, and clone the course repository there. If you cloned it earlier, run `git pull` inside it instead.

```bash
ssh YOUR_UUN@mlp.inf.ed.ac.uk
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
cd edin-mls-26-autumn/hw1-asr
bash setup_env.sh
```

`mlp` only accepts connections from the Informatics VPN or a DICE machine. From anywhere else, first `ssh YOUR_UUN@ssh.inf.ed.ac.uk`, then run the `ssh` to `mlp` from there.

Run `setup_env.sh` on the head node, because it downloads files and the GPU nodes may have no internet access. It installs Miniconda if you do not have conda yet, creates a conda environment called `mls-asr` with PyTorch and Hugging Face `transformers`, and downloads the model (4.5 GB). It needs about 20 GB of space in your home directory, and most of its time goes into downloading about 8 GB. You can run it again if it stops halfway.

## Step 2: get a GPU shell

Do this every time you work on the homework:

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
cd ~/edin-mls-26-autumn/hw1-asr      # or wherever you cloned it
nvidia-smi -L                         # which GPU did you get?
```

The model needs more than the tutorial examples did: 16 GB of CPU memory (`--mem=16G`) and about 9 GB of GPU memory. When you are done, leave with `exit` so the GPU goes back to the pool.

If you prefer batch jobs, `sbatch run_asr.sbatch` runs all the tests and the model and writes the output to `asr-JOB_ID.out`. Options on the command line override the ones in the script, so `sbatch --gres=gpu:nvidia_rtx_a6000:1 run_asr.sbatch` runs it on an A6000.

### Which GPU you get

The Teaching partition has several kinds of GPU, and `--gres=gpu:1` gives you whichever is free. `nvidia-smi -L` tells you which one you got:

| `nvidia-smi -L` shows | SMs | GPU memory | To ask for this one |
|-----------------------|----:|-----------:|---------------------|
| `NVIDIA GeForce RTX 2080 Ti` | 68 | 11 GB | `--gres=gpu:nvidia_geforce_rtx_2080_ti:1` |
| `NVIDIA RTX A6000` | 84 | 48 GB | `--gres=gpu:nvidia_rtx_a6000:1` |
| `NVIDIA H200` with `MIG 1g.18gb` below it | 16 | 18 GB | `--gres=gpu:h200_1g.18gb:1` |
| `NVIDIA H200` with `MIG 3g.71gb` below it | 60 | 71 GB | `--gres=gpu:h200_3g.71gb:1` |

There is also one whole H200 (132 SMs, 141 GB), but it is rarely free.

An SM (streaming multiprocessor) is one of the GPU's processors; the blocks of a kernel are spread over the SMs. A MIG slice is a part of an H200 that works like a separate, smaller GPU: a `1g.18gb` slice has 16 of the H200's 132 SMs and 18 GB of its memory. The H200 is a newer and faster GPU than the other two, but a small slice of it is slower than a whole 2080 Ti.

Your code works on all of them, and every one has enough memory for the model. The speed is very different, though, so when you compare two versions of a kernel, run both on the same kind of GPU. To get the same kind every time, put its name in `--gres` as in the table, for example:

```bash
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
```

Asking for one kind can mean a longer wait when all of them are busy.

## Step 3: check your setup

```bash
make                     # compile the kernels and the tests for this GPU
./run.sh --cuda none     # run the model on PyTorch only
```

`make` has to finish without errors. The blanks are still empty, so the kernels compile but do not compute anything yet. `./run.sh --cuda none` then runs the whole model with PyTorch's own operations instead of your kernels. Before it starts, Python loads PyTorch and the model's weights from your home directory, which takes from half a minute to a few minutes, depending on the node and on how busy the file system is. It should end like this (this is from an RTX 2080 Ti; the times depend on the GPU, and for PyTorch also on the node's CPU):

```
Time: 992.9 ms per transcription (+/- 4.8 ms), 76.4 ms per token
Peak GPU memory: 8.88 GiB
Transcription: Concord returned to its place amidst the tents.
Expected:      Concord returned to its place amidst the tents
Accuracy: 100.0%
Status: PASS
```

You have not changed any code yet, so if this fails, something in the setup is wrong. [When something goes wrong](#when-something-goes-wrong) lists the usual causes. Fix it before you start the tasks.

## Step 4: do the tasks

For each task:

1. Read its guide in `docs/`. It explains what the kernel computes, how the work is split between threads, and each blank.
2. Fill in the blanks.
3. Compile, test, and run the model with your kernel:

```bash
make                     # compile; fix every error it reports
make test1               # the task's test (make test2, test3, test4 for the others)
./run.sh --cuda task1    # the model with your task 1 kernel switched on
```

The test runs your kernel on the shapes the model uses, compares the results with a CPU version, and prints the time. When it says `ALL PASS`, run the model: `./run.sh --cuda task1` after task 1, `./run.sh --cuda task1,task2` after task 2, and so on.

The tests print one line per case. While a blank is still empty you will see failures like:

```
text decoder (GQA 16/4, full)        FAIL  max_err=2.39e+00
    160556 of 163840 checked values are wrong. First: index 2048, got -0.0918819, expected -0.22623
```

When the kernel is right, the line says `PASS` instead and also shows the time of one call and the speed of your kernel.

`--cuda` takes a list of tasks: those use your kernels, and the rest use PyTorch. So you can check each task inside the real model before the others are done. A few kernels in `csrc/provided.cu` are already complete; they are always on unless you pass `--cuda none`.

The tests are also the place to experiment. Every task file has a tuning constant at the top (`ROPE_THREADS`, `REDUCE_THREADS`, `TILE`, `ATTN_WARPS`), and every guide ends with a "Try this" list of experiments. Change something, rerun the test, and see what happens to the time.

## Step 5: run everything on your kernels

```bash
./run.sh --profile
```

With all four tasks done, this transcribes the recording on your kernels and then shows where the GPU time goes: one line per operation, with where it ran (`cuda` for your kernels, `torch` for PyTorch), how many times it was called, its total time and its share of the total.

The matrix multiplications of task 3 take most of the time. `./run.sh --cuda none --profile` shows the same table for PyTorch, whose matrix multiply is NVIDIA's cuBLAS library. `./run.sh --tokens` prints every generated token with the probability your softmax gave it.

## How the pieces fit together

```
benchmark.py                 reads the audio, runs the model, checks the transcription
  glm_asr_cuda/model.py      the model: which operation runs when, on which arrays
    glm_asr_cuda/ops.py      one Python function per operation; calls your kernel or PyTorch
      build/sm_XX/libasr_kernels.so      built by `make` from csrc/
        csrc/task*.cu        your kernels
tests/test_task*.cu          the tests: the same kernels on fixed inputs, with a CPU reference
```

Python cannot call CUDA C++ directly. `make` compiles `csrc/` into a shared library, and `ops.py` calls the plain C launch functions in it (`launch_rope`, `launch_linear`, ...) through Python's `ctypes` module, passing the GPU addresses of PyTorch tensors. The launch functions are already written; they sit at the bottom of each task file and show the grid and block sizes your kernel is started with.

`make` compiles for the GPU of the node you are on and keeps each GPU type in its own directory: `build/sm_75` for an RTX 2080 Ti, `build/sm_86` for an RTX A6000, `build/sm_90` for an H200. So always run `make` (or `./run.sh`, which runs it for you) on a GPU node. Other useful targets: `make test` runs all four tests, `make clean` deletes `build/`.

## Rules

- Your work goes in the five task files in `csrc/`. You can change anything in them, including the tuning constants and the structure of the kernels, as long as the tests and the transcription still pass.
- The kernels must be your own CUDA code. Do not call cuBLAS, cuDNN, CUTLASS, Thrust or PyTorch to do the work of a task.
- Hand in with `ops.py`, `model.py`, `benchmark.py`, `provided.cu` and the tests unchanged. You are welcome to change them while you experiment.

## What to hand in

- The five files you edited: `csrc/task1_rope.cu`, `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu`, `csrc/task3_gemm.cu` and `csrc/task4_attention.cu`. Put these files into the final code zip file you hand in.
- The output of `make test` and of `./run.sh --profile`, both from the same GPU node, and the GPU name (`nvidia-smi -L`). (If this is too long, put things you consider the most important into the main body of the report, and the remaining into the appendix.)
- Choose at least one question from each guide to answer in the final report, preferably together in a same place (e.g., a subsection). If you want to answer more, or you think it would be helpful to demonstrate your whole system, you can put answers to more than the required four questions into the report. You can also put answers beyond the four required ones in the appendix, as long as there are no less than four questions, each from one guide, in the main body of the report.
-  If you choose to focus your extra exploration point on the ASR part, please develop your system and demonstrate in the report as described in the related requirement files.

## When something goes wrong

| What you see | Usual cause | What to do |
|--------------|-------------|------------|
| `No GPU found. Build on a GPU node` | You are on the head node | Get a GPU shell first (step 2) |
| `nvcc: command not found` | CUDA is not on your PATH | `export PATH=/opt/cuda-12.8.0/bin:$PATH` |
| `ModuleNotFoundError: No module named 'torch'` | The conda environment is not active | `conda activate mls-asr` |
| A test case says `FAIL` and `got 0` | A blank is still empty, or the kernel writes to the wrong place | Check the indices of that blank against the guide |
| `CUDA error ... an illegal memory access` | An index runs past the end of an array | Run the test under `compute-sanitizer` (below) |
| The test hangs | A `__syncthreads()` that only some threads of the block reach | Keep every barrier outside `if`s and early `return`s that depend on the thread |
| `nan` in the output | `exp` of a large number, a division by zero, or `-inf - (-inf)` | See the notes on `-INFINITY` in the task 2 and task 4 guides |
| Tests pass but the transcription is wrong | Often a kernel that only fails on shapes the tests miss | `./run.sh --cuda task1`, then `task1,task2`, ... to find the task that breaks it |
| `CUDA out of memory` | Something else is using the GPU, or the GPU is too small | `nvidia-smi`; ask for another GPU |
| `no kernel image is available for execution` | The library was built for a different GPU | Run `make` on the node you run on |

Two tools help with the harder bugs:

- `ASR_DEBUG=1 ./run.sh` waits for every kernel to finish before starting the next, so an error is reported by the operation that caused it, not by a later one.
- `compute-sanitizer build/sm_XX/test_task3_gemm small` (with your GPU's `sm_XX`) reports the exact line of every out-of-bounds access. `compute-sanitizer --tool racecheck build/sm_XX/test_task3_gemm small` reports shared-memory races, such as a missing `__syncthreads()`. The sanitizer makes every kernel many times slower, so use it on the tests rather than the whole model, and add `small`: the test then runs only its small cases and skips the timing, which takes seconds instead of many minutes.
