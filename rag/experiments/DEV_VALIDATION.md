# Dev split validation (125 queries)

The completed experiment uses five frozen arms: `dense5`, `dense20`, `rerank5`,
`dense_budget`, and `rerank_budget`. Do not tune methods from dev scores.
The report revision only recomputes summaries from existing outputs.
**Do not rerun dev generation or judging for this revision.**

## Existing-results audit (safe: no model/API calls)

From the repository root:

```bash
~/mls-rag-env/bin/python rag/extensions/summarize_selection_experiment.py \
  --run rag/results/dev-validation --judge-prefix judge_json_ \
  --split dev --output rag/results/dev-validation/comparison.json
python3 rag/extensions/audit_report_evidence.py
```

The summariser replaces the specified summary, validates matched query coverage,
and hashes its answer/judge inputs. `comparison.json` and the report-evidence
summary are committed; raw answers and judge outputs remain local.
`limitations` now follows the split. Generation times in the report use
`median_generation_ms`, not the mean.

## Historical reproduction procedure (not executed during report revision)

The following requires course assets, model caches and local API credentials.
It documents how the original run was made, not an instruction to rerun dev.

## 1. Transcripts

```bash
conda activate mls-asr
export PATH=/usr/local/cuda/bin:$PATH
python rag/scripts/evaluate_rag.py --split dev \
  --transcript-cache rag/results/dev_transcripts_cuda.jsonl
```

Build `rag/experiments/dev125.jsonl` from the cache:

```bash
python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("rag/results/dev_transcripts_cuda.jsonl").read_text().splitlines() if l.strip()]
for r in rows:
    r["split"] = "dev"
Path("rag/experiments/dev125.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
PY
```

## 2. Retrieval rankings

```bash
python rag/extensions/compare_retrieval_routes.py \
  --split dev \
  --transcript-cache rag/experiments/dev125.jsonl \
  --output rag/results/bm25-compare-dev
```

## 3. Frozen five-arm run

```bash
python rag/extensions/run_selection_experiment.py prepare \
  --sample rag/experiments/dev125.jsonl \
  --rankings rag/results/bm25-compare-dev/dense_rankings.jsonl \
  --rerank-device cuda \
  --seed "2026-11-13-dev-validation-v1" \
  --output rag/results/dev-validation/plan

python rag/extensions/run_selection_experiment.py generate \
  --plan rag/results/dev-validation/plan \
  --output rag/results/dev-validation/answers --resume
```

Judge and summarize as in `extensions/README.md` (`--split dev` for evaluate_answers).
The original launch script is `rag/experiments/run_dev_validation_nohup.sh`;
it invokes generation and judge services and must not be run as an audit command.

Report dev as **held-out confirmation**; do not tune on dev after seeing scores.
