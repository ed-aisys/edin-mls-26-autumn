# 0. Environment: the Teaching Cluster

Slides 28–33 and 44. Logging in and receiving a GPU allocation are separate steps. The head node is for preparing files and submitting jobs, so seeing no GPU there is normal. GPU work runs on a compute node that Slurm allocates to you.

## 1. Log in: two ssh hops (slide 29)

```bash
# From an approved campus/VPN network:
ssh YOUR_UUN@student.ssh.inf.ed.ac.uk
# On the gateway:
ssh icf.inf.ed.ac.uk
```

Before class, check three things separately: your own DICE account, Teaching course access, and a campus or approved VPN network.

Slides 30–32 add the macOS Kerberos route (`kinit -f YOUR_UUN@INF.ED.AC.UK`, then ssh with GSSAPI), the difference between your DICE/AFS home on the gateway and your cluster home on `icf`, and links to the official DICE and cluster documentation. Compute nodes use the cluster home, so clone this repository there:

```bash
# On icf (the head node)
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
```

## 2. A Slurm allocation for a GPU shell (slide 33)

```bash
sinfo -p Teaching
srun -p Teaching --gres=gpu:1 \
     --cpus-per-task=1 --mem=2G \
     --time=00:40:00 --pty bash
hostname
nvidia-smi -L
export PATH=/opt/cuda-12.8.0/bin:$PATH
nvcc --version
```

1. `sinfo` shows the partition state. It does not reserve a slot.
2. `srun` queues for 1 GPU and a 40-minute shell.
3. `hostname` and `nvidia-smi -L` show which node and which GPU you got. It may be an RTX 2080 Ti, an A6000, or a MIG slice of an H200.
4. `nvcc --version` confirms the CUDA 12.8 compiler is on PATH.

## 3. Check the GPU

```bash
cd edin-mls-26-autumn/cuda-tutorial
make 0-environment/check
./0-environment/check
```

Output on an RTX 2080 Ti:

```
GPU             NVIDIA GeForce RTX 2080 Ti
compute cap.    7.5 (sm_75)
CUDA runtime    12.8, driver supports up to 13.0
SMs             68
memory          10.6 GiB
threads / SM    1024
threads / block 1024
block (x,y,z)   1024, 1024, 64
grid  (x,y,z)   2147483647, 65535, 65535
kernel launch   OK
```

These are the limits on slide 44. Threads per SM differ between GPU generations: 1024 on the 2080 Ti (Turing), 2048 on the GTX 1080 Ti in the slide screenshot and on the H200.

## 4. Leave the shell

`exit` releases the allocation. Do not keep a GPU shell open when you are not using it.
