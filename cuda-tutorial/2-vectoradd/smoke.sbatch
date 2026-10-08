#!/bin/bash
#SBATCH --job-name=gpu-lecture-smoke
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=smoke-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
printf 'JOB=%s NODE=%s\n' "$SLURM_JOB_ID" "$(hostname)"
export PATH=/opt/cuda-12.8.0/bin:$PATH
nvcc --version
# Compile on the allocated GPU node so that native matches the allocated device.
nvcc -O2 -std=c++17 -arch=native vector_add.cu -o vector_add
./vector_add 1003 256
./vector_add 257 128
