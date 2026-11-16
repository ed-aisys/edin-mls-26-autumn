#!/usr/bin/env bash
#SBATCH --job-name=mls-qwen06-retrieval
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=rag/results/qwen06-retrieval-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch on an allocated GPU}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -r /home/htang2/toolchain-20251006/toolchain.rc ]]; then
  source /home/htang2/toolchain-20251006/toolchain.rc
fi
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
RUN_PYTHON="$HOME/mls-asr-env/bin/python"
RANKINGS="${1:?Supply frozen full-practice rankings.jsonl}"
RUN_DIR="rag/results/qwen06-retrieval-$SLURM_JOB_ID"
date -Is
hostname
git rev-parse HEAD
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$RUN_PYTHON" -m unittest discover -s rag/tests -p test_qwen_retrieval.py -v
"$RUN_PYTHON" -u rag/extensions/evaluate_qwen_reranker.py run \
  --rankings "$RANKINGS" --output "$RUN_DIR/smoke" --limit 2 --batch-size 8
"$RUN_PYTHON" -u rag/extensions/evaluate_qwen_reranker.py run \
  --rankings "$RANKINGS" --output "$RUN_DIR/full" --batch-size 8
date -Is
echo QWEN_RETRIEVAL_JOB_COMPLETE
