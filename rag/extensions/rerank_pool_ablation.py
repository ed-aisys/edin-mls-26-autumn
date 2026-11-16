"""Rerank top-5 from dense pools of size 20/50/100; report selected-ID recall vs qrels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rankings", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=ROOT / "rag/models/selection")
    ap.add_argument("--pools", type=int, nargs="+", default=(20, 50, 100))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = CourseData(ROOT / "rag/data/rag_corpus")
    qrels = data.load_qrels("practice")
    records = {r["record_id"]: r["text"] for r in load_jsonl(ROOT / "rag/data/rag_corpus/corpus.jsonl")}
    rows = load_jsonl(args.rankings)

    device = torch.device("cpu" if args.device == "cpu" else "cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    tok = AutoTokenizer.from_pretrained(str(args.assets / "reranker"), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.assets / "reranker"), local_files_only=True).to(device).eval()

    def score_pairs(question: str, ids: list[str]) -> list[float]:
        out = []
        with torch.inference_mode():
            for start in range(0, len(ids), args.batch_size):
                batch = ids[start : start + args.batch_size]
                inputs = tok([question] * len(batch), [records[i] for i in batch],
                               padding=True, truncation="only_second", max_length=512,
                               return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out.extend(model(**inputs).logits.reshape(-1).tolist())
        return out

    per_pool_times = {p: [] for p in args.pools}
    per_pool_recall = {p: [] for p in args.pools}
    details = []

    for i, row in enumerate(rows, 1):
        qid, question = row["query_id"], row["transcript"]
        ids = row["record_ids"]
        rel = set(qrels[qid])
        item = {"query_id": qid}
        for pool in args.pools:
            sub = ids[:pool]
            t0 = time.perf_counter()
            scores = score_pairs(question, sub)
            ms = (time.perf_counter() - t0) * 1000
            per_pool_times[pool].append(ms)
            order = sorted(range(len(sub)), key=lambda j: (-scores[j], j))
            top5 = [sub[j] for j in order[:5]]
            r5 = recall_at_k(top5, rel, 5)
            per_pool_recall[pool].append(r5)
            item[f"pool_{pool}_rerank_ms"] = ms
            item[f"pool_{pool}_recall_at_5"] = r5
        details.append(item)
        if i % 25 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] pools done", flush=True)

    summary = {
        "queries": len(rows),
        "device": args.device,
        "pools": args.pools,
        "mean_rerank_ms": {str(p): sum(per_pool_times[p]) / len(per_pool_times[p]) for p in args.pools},
        "mean_selected_id_recall_at_5": {str(p): sum(per_pool_recall[p]) / len(per_pool_recall[p]) for p in args.pools},
        "dense_top5_baseline_recall_at_5": sum(recall_at_k(r["record_ids"][:5], set(qrels[r["query_id"]]), 5) for r in rows) / len(rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_query.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in details), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
