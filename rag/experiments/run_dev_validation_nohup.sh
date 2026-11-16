#!/usr/bin/env bash
# Dev held-out validation: prepare (GPU rerank) + ELM generate + judge + summary.
# Usage: nohup bash rag/experiments/run_dev_validation_nohup.sh > rag/results/dev-validation/nohup.master.log 2>&1 &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RUN=rag/results/dev-validation
PLAN="$RUN/plan"
ANS="$RUN/answers"
RANK=rag/results/bm25-compare-dev/dense_rankings.jsonl
SAMPLE=rag/experiments/dev125.jsonl
mkdir -p "$RUN"

source "$HOME/mls-rag-env/bin/activate" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PY="${RAG_PYTHON:-$HOME/mls-rag-env/bin/python}"

echo "=== dev validation start $(date -Is) host=$(hostname) ==="

if [[ ! -d "$PLAN" ]]; then
  echo "=== prepare (125 x 5 arms, cuda rerank) ==="
  "$PY" -u rag/extensions/run_selection_experiment.py prepare \
    --sample "$SAMPLE" \
    --rankings "$RANK" \
    --rerank-device cuda \
    --seed "2026-11-13-dev-validation-v1" \
    --output "$PLAN"
fi

if [[ ! -f rag/.env ]]; then
  echo "ABORT: rag/.env missing"
  exit 1
fi

GEN_EXTRA=()
if [[ -f "$ANS/generation_config.json" ]]; then
  GEN_EXTRA+=(--resume)
fi
echo "=== ELM generate dev ==="
"$PY" -u rag/extensions/run_selection_experiment.py generate \
  --plan "$PLAN" \
  --output "$ANS" \
  "${GEN_EXTRA[@]}"

echo "=== judge dev (json-mode) ==="
for arm in dense5 dense20 rerank5 dense_budget rerank_budget; do
  "$PY" -u rag/scripts/evaluate_answers.py --split dev --json-mode \
    --candidates "$ANS/$arm.jsonl" \
    --out-dir "$ANS/judge_json_$arm"
done

if [[ -f "$RUN/comparison.json" ]]; then
  rm -f "$RUN/comparison.json"
fi
"$PY" -u rag/extensions/summarize_selection_experiment.py \
  --run "$RUN" --judge-prefix judge_json_ --split dev \
  --output "$RUN/comparison.json"

echo "=== dev validation complete $(date -Is) ==="
