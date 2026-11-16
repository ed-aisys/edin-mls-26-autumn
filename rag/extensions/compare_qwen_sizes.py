"""Post-run paired comparisons; no inference or parameter selection."""
import hashlib
import json
from pathlib import Path
import random
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "rag"))
from src.data import load_jsonl

small_dir = root / "rag/results/qwen06-retrieval-3724137/full"
large_dir = root / "rag/results/qwen4-retrieval-3724174/full"
small = {r["query_id"]: r for r in load_jsonl(small_dir / "rankings.jsonl")}
large = {r["query_id"]: r for r in load_jsonl(large_dir / "rankings.jsonl")}
small_config = json.loads((small_dir / "config.json").read_text())
large_config = json.loads((large_dir / "config.json").read_text())
assert len(small) == len(large) == 350 and set(small) == set(large)
for key in ("rankings", "corpus", "qrels", "metrics"):
    assert small_config["input_sha256"][key] == large_config["input_sha256"][key]
for key in ("instruction", "prefix", "suffix", "input_policy", "batch_size", "qwen_dtype", "device"):
    assert small_config[key] == large_config[key]
for directory, config in ((small_dir, small_config), (large_dir, large_config)):
    assert hashlib.sha256((directory.parent / "runner_at_submission.py").read_bytes()).hexdigest() == config["input_sha256"]["script"]
    assert hashlib.sha256((directory.parent / "qrels_practice.tsv").read_bytes()).hexdigest() == config["input_sha256"]["qrels"]
for qid in small:
    assert small[qid]["transcript"] == large[qid]["transcript"]
    assert small[qid]["dense"] == large[qid]["dense"]
    assert small[qid]["minilm_512"]["record_ids"] == large[qid]["minilm_512"]["record_ids"]
ids = list(small)
rng = random.Random(20261116)
samples = [rng.choices(range(350), k=350) for _ in range(2000)]


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


result = {"queries": 350, "matched_input_protocol_verified": True,
          "dense_and_minilm_orderings_identical_between_runs": True,
          "bootstrap": {"replicates": 2000, "seed": 20261116, "rng": "stdlib random.Random",
                        "scope": "descriptive query-paired intervals, no multiple-comparison adjustment"},
          "comparisons": {}}
for name, left, arm_left, right, arm_right in (
    ("4b_minus_06b_matched", large, "qwen_matched", small, "qwen_matched"),
    ("4b_minus_06b_native2048", large, "qwen_native2048", small, "qwen_native2048"),
    ("4b_native_minus_matched", large, "qwen_native2048", large, "qwen_matched"),
    ("06b_native_minus_matched", small, "qwen_native2048", small, "qwen_matched"),
):
    result["comparisons"][name] = {}
    for metric in small[ids[0]]["dense"]["metrics"]:
        deltas = [left[q][arm_left]["metrics"][metric] - right[q][arm_right]["metrics"][metric] for q in ids]
        means = sorted(sum(deltas[i] for i in sample) / 350 for sample in samples)
        result["comparisons"][name][metric] = {"mean_delta": sum(deltas) / 350,
            "bootstrap_95pct_interval": [percentile(means, 0.025), percentile(means, 0.975)],
            "improved": sum(v > 1e-12 for v in deltas), "worse": sum(v < -1e-12 for v in deltas),
            "tied": sum(abs(v) <= 1e-12 for v in deltas)}
output = root / "rag/results/qwen4-retrieval-3724174/paired-size-comparison.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
for name, values in result["comparisons"].items():
    print(name)
    for metric in ("Recall@5", "MRR@10", "NDCG@10"):
        print(metric, json.dumps(values[metric]))
