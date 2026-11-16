#!/usr/bin/env bash
#SBATCH --job-name=mls-qwen4-answers
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=rag/results/qwen4-answers-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
# All model generation/judging happens at ELM; no local GPU is requested.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
RUN_PYTHON="$HOME/mls-rag-env/bin/python"
RUN_DIR="${1:-rag/results/qwen4-answers-$SLURM_JOB_ID}"
RANKINGS=rag/results/qwen4-retrieval-3724174/full/rankings.jsonl
date -Is
hostname
"$RUN_PYTHON" -m unittest discover -s rag/tests -p test_qwen_answers.py -v
"$RUN_PYTHON" -m unittest discover -s rag/tests -p test_judge_protocol.py -v
if [[ ! -f "$RUN_DIR/prepared.json" ]]; then
  "$RUN_PYTHON" -u rag/extensions/run_qwen_answer_experiment.py prepare \
    --rankings "$RANKINGS" --run "$RUN_DIR"
fi
RESUME=()
if [[ -f "$RUN_DIR/answers/generation_config.json" ]]; then
  RESUME=(--resume)
fi
"$RUN_PYTHON" -u rag/extensions/run_qwen_answer_experiment.py generate \
  --run "$RUN_DIR" --max-new-queries 2 "${RESUME[@]}"
"$RUN_PYTHON" -u rag/extensions/run_qwen_answer_experiment.py smoke --run "$RUN_DIR"
for ARM in dense5 qwen4_matched5; do
  "$RUN_PYTHON" -u rag/scripts/evaluate_answers.py --split practice --allow-subset --dry-run \
    --candidates "$RUN_DIR/answers/smoke_$ARM.jsonl"
  "$RUN_PYTHON" -u rag/scripts/evaluate_answers.py --split practice --allow-subset \
    --model utter-project/EuroLLM-22B-Instruct-2512 --json-mode --api-retries 1 \
    --candidates "$RUN_DIR/answers/smoke_$ARM.jsonl" --out-dir "$RUN_DIR/judge_json_$ARM"
done
echo QWEN4_ANSWER_SMOKE_PASSED
"$RUN_PYTHON" -u rag/extensions/run_qwen_answer_experiment.py generate --run "$RUN_DIR" --resume
for ARM in dense5 qwen4_matched5; do
  "$RUN_PYTHON" -u rag/scripts/evaluate_answers.py --split practice --dry-run \
    --candidates "$RUN_DIR/answers/$ARM.jsonl"
  "$RUN_PYTHON" -u rag/scripts/evaluate_answers.py --split practice \
    --model utter-project/EuroLLM-22B-Instruct-2512 --json-mode \
    --candidates "$RUN_DIR/answers/$ARM.jsonl" --out-dir "$RUN_DIR/judge_json_$ARM"
done
"$RUN_PYTHON" -u rag/extensions/run_qwen_answer_experiment.py summarise --run "$RUN_DIR"
date -Is
echo QWEN4_ANSWER_JOB_COMPLETE
