"""Compare dense, BM25 and RRF retrieval on frozen transcripts (no ASR/ELM)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))

from extensions.bm25_retriever import BM25Retriever, reciprocal_rank_fusion
from src.data import CourseData, load_jsonl
from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.retriever import ExactDenseRetriever


def load_transcript_cache(path: Path) -> dict[str, str]:
    cache = {row["query_id"]: row["transcript"] for row in load_jsonl(path)}
    if len(cache) != len(load_jsonl(path)):
        raise ValueError("Duplicate query_id in transcript cache")
    return cache


def aggregate(rows: list[dict], prefix: str) -> dict:
    keys = ("Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10")
    out = {f"{prefix}_{k}": float(np.mean([r[f"{prefix}_{k}"] for r in rows])) for k in keys}
    out[f"{prefix}_mean_ms"] = float(np.mean([r[f"{prefix}_ms"] for r in rows]))
    return out


def complementarity(dense_ids: list[str], bm25_ids: list[str], rel: dict, k: int = 100) -> dict:
    relevant = {rid for rid, score in rel.items() if score > 0}
    d_set = set(dense_ids[:k])
    b_set = set(bm25_ids[:k])
    only_bm25 = relevant & b_set - d_set
    only_dense = relevant & d_set - b_set
    both = relevant & d_set & b_set
    neither = relevant - d_set - b_set
    return {
        f"relevant_in_both_top{k}": len(both),
        f"relevant_only_dense_top{k}": len(only_dense),
        f"relevant_only_bm25_top{k}": len(only_bm25),
        f"relevant_in_neither_top{k}": len(neither),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rrf-c", type=float, default=60.0)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new output directory")
    if args.top_k < 100:
        ap.error("--top-k must be >= 100 for Recall@100")

    data = CourseData(ROOT / "rag/data/rag_corpus")
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split)
    transcripts = load_transcript_cache(args.transcript_cache)
    missing = [it["query_id"] for it in items if it["query_id"] not in transcripts]
    if missing:
        raise ValueError(f"Cache missing {len(missing)} {args.split} query ids (first: {missing[0]})")

    record_ids = list(data.records.keys())
    texts = [data.records[rid] for rid in record_ids]
    t0 = time.perf_counter()
    bm25 = BM25Retriever(record_ids, texts)
    build_s = time.perf_counter() - t0

    from src.encoder import LocalEncoder

    encoder = LocalEncoder(ROOT / "rag/models/enc_a", device="cpu")
    dense = ExactDenseRetriever(
        ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a.npy",
        ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a_ids.json",
    )

    args.output.mkdir(parents=True)
    per_query = []
    for i, item in enumerate(items, 1):
        qid = item["query_id"]
        transcript = transcripts[qid]
        rel = qrels[qid]

        t_enc = time.perf_counter()
        vector = encoder.encode(transcript)
        enc_ms = (time.perf_counter() - t_enc) * 1000.0
        d_ret = dense.retrieve(vector, k=args.top_k)
        b_ret = bm25.retrieve(transcript, k=args.top_k)
        rrf_ids = reciprocal_rank_fusion([d_ret["record_ids"], b_ret["record_ids"]], k_out=args.top_k, c=args.rrf_c)

        row = {
            "query_id": qid,
            "split": args.split,
            "dense_ms": enc_ms + d_ret["latency_ms"],
            "bm25_ms": b_ret["latency_ms"],
            "rrf_ms": d_ret["latency_ms"] + b_ret["latency_ms"],
            "dense_record_ids": d_ret["record_ids"],
            "bm25_record_ids": b_ret["record_ids"],
            "rrf_record_ids": rrf_ids,
            **complementarity(d_ret["record_ids"], b_ret["record_ids"], rel, k=args.top_k),
        }
        for route, ids in (("dense", d_ret["record_ids"]), ("bm25", b_ret["record_ids"]), ("rrf", rrf_ids)):
            for k in (5, 10, 20, 100):
                row[f"{route}_Recall@{k}"] = recall_at_k(ids, rel, k)
            row[f"{route}_MRR@10"] = mrr_at_k(ids, rel, 10)
            row[f"{route}_NDCG@10"] = ndcg_at_k(ids, rel, 10)
        per_query.append(row)
        with (args.output / "per_query.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        ranking_row = {
            "query_id": qid,
            "transcript": transcript,
            "record_ids": d_ret["record_ids"],
            "scores": d_ret["scores"],
            "latency_ms": d_ret["latency_ms"],
        }
        with (args.output / "dense_rankings.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(ranking_row) + "\n")
        print(
            f"[{i}/{len(items)}] {qid} "
            f"dR@5={row['dense_Recall@5']:.3f} bR@5={row['bm25_Recall@5']:.3f} "
            f"rrfR@5={row['rrf_Recall@5']:.3f}",
            flush=True,
        )

    summary = {
        "split": args.split,
        "queries": len(per_query),
        "bm25_index_build_seconds": build_s,
        "rrf_c": args.rrf_c,
        "dense": aggregate(per_query, "dense"),
        "bm25": aggregate(per_query, "bm25"),
        "rrf": aggregate(per_query, "rrf"),
        "complementarity_totals": {
            "relevant_only_bm25_top100_queries": sum(1 for r in per_query if r["relevant_only_bm25_top100"] > 0),
            "relevant_only_dense_top100_queries": sum(1 for r in per_query if r["relevant_only_dense_top100"] > 0),
            "bm25_rescues_dense_miss_top5": sum(
                1
                for r in per_query
                if r["dense_Recall@5"] < 1.0 and r["bm25_Recall@5"] > r["dense_Recall@5"]
            ),
            "dense_beats_bm25_top5": sum(1 for r in per_query if r["dense_Recall@5"] > r["bm25_Recall@5"]),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
