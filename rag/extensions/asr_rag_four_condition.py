"""2×2 diagnostic: CUDA vs reference transcript × dense top-5 vs budget selection (retrieval layer)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))

from extensions.evidence_selection import select_context
from src.data import CourseData, load_jsonl
from src.encoder import LocalEncoder
from src.metrics import recall_at_k
from src.retriever import ExactDenseRetriever


def load_cache(path: Path) -> dict[str, str]:
    return {r["query_id"]: r["transcript"] for r in load_jsonl(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--cuda-cache", type=Path, required=True, help="Transcripts from ASR_CUDA=all")
    ap.add_argument("--reference-cache", type=Path, required=True, help="Oracle/reference transcripts (e.g. ASR_CUDA=none)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--budget-tokens", type=int, default=4096)
    ap.add_argument("--query-ids", type=Path, default=None, help="Optional jsonl subset")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new output directory")

    data = CourseData(ROOT / "rag/data/rag_corpus")
    qrels = data.load_qrels(args.split)
    records = {rid: data.record_text(rid) for rid in data.records}
    cuda_t = load_cache(args.cuda_cache)
    ref_t = load_cache(args.reference_cache)

    if args.query_ids:
        qids = [r["query_id"] for r in load_jsonl(args.query_ids)]
    else:
        qids = [it["query_id"] for it in data.audio_items(args.split)]

    encoder = LocalEncoder(ROOT / "rag/models/enc_a", device="cpu")
    retriever = ExactDenseRetriever(
        ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a.npy",
        ROOT / "rag/data/rag_corpus/embeddings/corpus_enc_a_ids.json",
    )
    from transformers import AutoTokenizer

    from extensions.evidence_selection import chat_input_tokens
    from src.generator import SYSTEM_PROMPT

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "rag/models/selection/tokenizer"), local_files_only=True
    )
    count_tokens = lambda prompt: chat_input_tokens(tokenizer, SYSTEM_PROMPT, prompt)

    args.output.mkdir(parents=True)
    rows = []
    for i, qid in enumerate(qids, 1):
        if qid not in cuda_t or qid not in ref_t:
            continue
        rel = qrels.get(qid, {})
        block = {"query_id": qid, "split": args.split, "conditions": {}}
        for label, transcript in (("cuda_asr", cuda_t[qid]), ("reference_asr", ref_t[qid])):
            vector = encoder.encode(transcript)
            ids = retriever.retrieve(vector, k=100)["record_ids"]
            dense5 = ids[:5]
            budget = select_context(transcript, ids, records, count_tokens, budget=args.budget_tokens)
            budget_ids = budget["retrieved_record_ids"]
            block["conditions"][label] = {
                "transcript": transcript,
                "dense5_recall": recall_at_k(dense5, rel, 5),
                "budget_recall_id": recall_at_k(budget_ids, rel, max(len(budget_ids), 1)),
                "budget_docs": len(budget_ids),
                "budget_tokens": budget.get("local_input_tokens"),
            }
        rows.append(block)
        with (args.output / "per_query.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(block) + "\n")
        if i % 10 == 0 or i == len(qids):
            print(f"[{i}/{len(qids)}] {qid}", flush=True)

    n = len(rows) or 1
    summary = {
        "queries": len(rows),
        "budget_tokens": args.budget_tokens,
        "transcript_drift_queries": sum(
            1
            for r in rows
            if r["conditions"]["cuda_asr"]["transcript"].strip() != r["conditions"]["reference_asr"]["transcript"].strip()
        ),
    }
    for key in ("cuda_asr", "reference_asr"):
        summary[f"{key}_mean_dense5_recall"] = sum(r["conditions"][key]["dense5_recall"] for r in rows) / n
        summary[f"{key}_mean_budget_recall"] = sum(r["conditions"][key]["budget_recall_id"] for r in rows) / n

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
