"""Exploratory practice-only L6/L12 retrieval control; no generation or judging."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

REVISIONS = {
    "l6": ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
    "l12": ("cross-encoder/ms-marco-MiniLM-L12-v2", "7b0235231ca2674cb8ca8f022859a6eba2b1c968"),
}
METRICS = ("recall_at_5", "mrr_at_10", "ndcg_at_10")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(order, relevance):
    return dict(zip(METRICS, (recall_at_k(order, relevance, 5),
                              mrr_at_k(order, relevance, 10),
                              ndcg_at_k(order, relevance, 10))))


def transition(first, second, relevant):
    """Second replaces first; count qrel-positive query/document pairs."""
    before, after = set(first[:5]) & set(relevant), set(second[:5]) & set(relevant)
    return {"removed_relevant_pairs": len(before - after),
            "added_relevant_pairs": len(after - before),
            "recall_direction": "improved" if len(after) > len(before) else
                                "worsened" if len(after) < len(before) else "unchanged"}


def bootstrap(deltas, seed=20261116, repeats=5000):
    import numpy as np
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rankings", type=Path, default=ROOT / "rag/results/selection-rankings-practice-full/rankings.jsonl")
    ap.add_argument("--l6", type=Path, default=ROOT / "rag/models/selection/reranker")
    ap.add_argument("--l12", type=Path, default=ROOT / "rag/models/rerank-model-control/l12")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output directory; no implicit resume/overwrite")
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not torch.cuda.is_available():
        raise ValueError("Use the same idle CUDA device for both models")
    torch.set_num_threads(2)
    torch.manual_seed(20261116)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    corpus_path = ROOT / "rag/data/rag_corpus/corpus.jsonl"
    qrels_path = ROOT / "rag/data/rag_corpus/qrels_practice.tsv"
    inputs = rows(args.rankings)
    qids = [r["query_id"] for r in inputs]
    assert len(qids) == len(set(qids)) == 350
    corpus = {r["record_id"]: r["text"] for r in rows(corpus_path)}
    for row in inputs:
        assert len(row["record_ids"]) == len(set(row["record_ids"])) == 100
        assert all(rid in corpus for rid in row["record_ids"])
    asset_paths = {"l6": args.l6, "l12": args.l12}
    assets = {
        name: {"repo": REVISIONS[name][0], "revision": REVISIONS[name][1],
               "files_sha256": {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file()}}
        for name, path in asset_paths.items()
    }
    original_manifest = json.loads((ROOT / "rag/models/selection/manifest.json").read_text())
    for name, expected in original_manifest["reranker"]["files_sha256"].items():
        assert assets["l6"]["files_sha256"][name] == expected
    args.output.mkdir(parents=True)
    config = {
        "purpose": "Post-hoc exploratory model replacement, not a depth-only causal ablation",
        "split": "practice", "queries": 350, "query_ids": qids,
        "candidates": 100, "selected": 5, "max_length": 512,
        "truncation": "only_second", "input_order": "query, document",
        "batch_size": 32, "dtype": "float32", "tf32": False, "cpu_threads": 2,
        "timing": "3 full-query warmups/model; one timed pass/model/query; CUDA synchronize before/after; includes tokenization, transfer, forward and score materialization; excludes sorting and model loading",
        "model_order": "L6 first on even query indices, L12 first on odd; same dense document order",
        "bootstrap": {"unit": "query", "paired": True, "seed": 20261116, "resamples": 5000,
                      "interval": "percentile 2.5/97.5; no multiplicity adjustment"},
        "planned_contrasts": ["l6 minus dense", "l12 minus dense", "l12 minus l6"],
        "planned_case_query_ids": ["q_b7390fb1b4a9ad79", "q_65e52727cacd73bb"],
        "assets": assets, "input_sha256": {str(p): sha(p) for p in (args.rankings, corpus_path, qrels_path)},
        "source_sha256": sha(Path(__file__)), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(), "generation_or_judge_calls": 0,
        "limitations": ["Exploratory: motivated by observed L6 failures, not held-out confirmation",
                        "No repeated-run uncertainty, judge scores or answer-quality comparison",
                        "Qrels may be incomplete; qrel-negative does not imply irrelevant",
                        "Changing checkpoints also changes learned weights/training; not capacity identification",
                        "Shared GPU/host conditions and single-pass timing do not establish stable throughput"],
    }
    save(args.output / "config.json", config)  # Freeze protocol before inference.
    gpu_before = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv"], text=True)
    models, tokenizers = {}, {}
    for name, path in asset_paths.items():
        tokenizers[name] = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        models[name] = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float32).to(device).eval()
        assert not models[name].training
    configs = {name: model.config.to_dict() for name, model in models.items()}
    assert configs["l6"]["num_hidden_layers"] == 6 and configs["l12"]["num_hidden_layers"] == 12
    assert all(model.config.num_labels == 1 for model in models.values())

    def score(name, row):
        scores = []
        ids = row["record_ids"]
        with torch.inference_mode():
            for start in range(0, len(ids), 32):
                batch = ids[start:start + 32]
                features = tokenizers[name]([row["transcript"]] * len(batch),
                                            [corpus[rid] for rid in batch],
                                            padding=True, truncation="only_second",
                                            max_length=512, return_tensors="pt").to(device)
                scores.extend(models[name](**features).logits.flatten().tolist())
        assert len(scores) == 100 and all(math.isfinite(s) for s in scores)
        return scores

    for _ in range(3):
        for name in ("l6", "l12"):
            score(name, inputs[0])
    torch.cuda.synchronize()
    measured = []
    with (args.output / "scores.jsonl").open("x") as out:
        for index, row in enumerate(inputs):
            record = {"query_id": row["query_id"], "models": {}}
            for name in (("l6", "l12") if index % 2 == 0 else ("l12", "l6")):
                torch.cuda.synchronize()
                start = time.perf_counter()
                scores = score(name, row)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                order = sorted(range(100), key=lambda j: (-scores[j], j))
                record["models"][name] = {"record_ids": [row["record_ids"][j] for j in order],
                                          "scores": [scores[j] for j in order], "reranking_ms": elapsed}
            measured.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            if (index + 1) % 25 == 0:
                print(f"Scored {index + 1}/350 practice questions", flush=True)
    # Qrels are used only after inference, never in preparation or scoring.
    qrels = {}
    for row in csv.DictReader(qrels_path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"):
        if float(row.get("score", 1)) > 0:
            qrels.setdefault(row["query_id"], {})[row["record_id"]] = float(row.get("score", 1))
    assert set(qids) == set(qrels)
    details, cases = [], []
    saved = {r["query_id"]: r for r in rows(ROOT / "rag/results/selection-run-full/plan/rerankings.jsonl")}
    original_l6_matches = 0
    max_l6_score_error = 0.0
    for row, result in zip(inputs, measured, strict=True):
        qid, ids = row["query_id"], row["record_ids"]
        orders = {"dense": ids, **{name: result["models"][name]["record_ids"] for name in models}}
        vals = {name: metrics(order, qrels[qid]) for name, order in orders.items()}
        transitions = {f"{second} minus {first}": transition(orders[first], orders[second], qrels[qid])
                       for first, second in (("dense", "l6"), ("dense", "l12"), ("l6", "l12"))}
        original_l6_matches += orders["l6"] == saved[qid]["record_ids"]
        old_scores = dict(zip(saved[qid]["record_ids"], saved[qid]["scores"]))
        max_l6_score_error = max(max_l6_score_error, max(
            abs(s - old_scores[rid]) for rid, s in zip(
                result["models"]["l6"]["record_ids"], result["models"]["l6"]["scores"], strict=True)))
        details.append({"query_id": qid, "metrics": vals, "transitions": transitions,
                        "timing_ms": {name: result["models"][name]["reranking_ms"] for name in models}})
        if qid in config["planned_case_query_ids"]:
            score_maps = {name: dict(zip(result["models"][name]["record_ids"],
                                        result["models"][name]["scores"])) for name in models}
            inspect_ids = list(dict.fromkeys(ids[:5] + orders["l6"][:5] + orders["l12"][:5]))
            cases.append({"query_id": qid, "transcript": row["transcript"], "metrics": vals,
                          "documents": [{"record_id": rid, "text": corpus[rid],
                                         "qrel_positive": rid in qrels[qid],
                                         "ranks": {name: order.index(rid) + 1 for name, order in orders.items()},
                                         "scores": {name: score_maps[name][rid] for name in models}}
                                        for rid in inspect_ids]})
    per_arm = {}
    for name in ("dense", "l6", "l12"):
        per_arm[name] = {metric: statistics.mean(r["metrics"][name][metric] for r in details)
                         for metric in METRICS}
        if name != "dense":
            timings = [r["timing_ms"][name] for r in details]
            per_arm[name].update(mean_reranking_ms=statistics.mean(timings),
                                 median_reranking_ms=statistics.median(timings),
                                 min_reranking_ms=min(timings), max_reranking_ms=max(timings))
    paired = {}
    for first, second in (("dense", "l6"), ("dense", "l12"), ("l6", "l12")):
        key = f"{second} minus {first}"
        paired[key] = {}
        for metric in METRICS:
            deltas = [r["metrics"][second][metric] - r["metrics"][first][metric] for r in details]
            paired[key][metric] = {"mean_delta": statistics.mean(deltas),
                                    "query_bootstrap_95_interval": bootstrap(deltas),
                                    "improved": sum(d > 0 for d in deltas),
                                    "worsened": sum(d < 0 for d in deltas),
                                    "unchanged": sum(d == 0 for d in deltas)}
        paired[key]["evidence_replacement"] = {
            field: sum(r["transitions"][key][field] for r in details)
            for field in ("removed_relevant_pairs", "added_relevant_pairs")}
    save(args.output / "per_query.json", details)
    save(args.output / "cases.json", cases)
    summary = {"split": "practice", "queries": 350, "exploratory": True,
               "per_arm": per_arm, "paired": paired,
               "l6_reproduction": {"identical_full_rankings": original_l6_matches,
                                   "max_absolute_score_difference": max_l6_score_error},
               "gpu_before": gpu_before,
               "model_architecture": {name: {"layers": c["num_hidden_layers"], "hidden_size": c["hidden_size"],
                                             "attention_heads": c["num_attention_heads"]} for name, c in configs.items()},
               "tokenizer_file_hash_equal": {f: assets["l6"]["files_sha256"].get(f) == assets["l12"]["files_sha256"].get(f)
                                             for f in ("tokenizer.json", "vocab.txt", "tokenizer_config.json")},
               "limitations": config["limitations"],
               "output_sha256": {name: sha(args.output / name)
                                 for name in ("config.json", "scores.jsonl", "per_query.json", "cases.json")}}
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
