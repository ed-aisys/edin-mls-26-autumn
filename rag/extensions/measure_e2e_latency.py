"""Wall-clock breakdown for audio → answer on a query sample (GPU node for ASR)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))

from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData, load_jsonl
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.retriever import ExactDenseRetriever


def run_once(pipeline_parts, audio, k: int):
    asr, encoder, retriever, data, ctx_builder, generator = pipeline_parts
    timing = {}

    t0 = time.perf_counter()
    transcript = asr.transcribe(audio)
    timing["asr_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    vector = encoder.encode(transcript)
    timing["embedding_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    retrieval = retriever.retrieve(vector, k=k)
    timing["retrieval_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    built = ctx_builder.build(transcript, retrieval["record_ids"], data)
    timing["context_ms"] = (time.perf_counter() - t0) * 1000.0

    if generator:
        t0 = time.perf_counter()
        generation = generator.generate(built["prompt"])
        timing["generation_ms"] = generation["latency_ms"]
        timing["generation_wall_ms"] = (time.perf_counter() - t0) * 1000.0
        answer = generation["answer"]
    else:
        timing["generation_ms"] = 0.0
        timing["generation_wall_ms"] = 0.0
        answer = ""

    timing["end_to_end_ms"] = sum(
        timing[k]
        for k in ("asr_ms", "embedding_ms", "retrieval_ms", "context_ms", "generation_wall_ms")
    )
    return {"transcript": transcript, "answer_preview": answer[:200], "timing": timing}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--sample-jsonl", type=Path, default=None, help="query_id list; default first N manifest items")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--skip-elm", action="store_true", help="Measure through context build only")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new output file or directory")

    data_dir = ROOT / "rag/data/rag_corpus"
    data = CourseData(data_dir)
    if args.sample_jsonl:
        qids = [r["query_id"] for r in load_jsonl(args.sample_jsonl)][: args.limit]
    else:
        qids = [it["query_id"] for it in data.audio_items(args.split)][: args.limit]

    items = {it["query_id"]: it for it in data.audio_items(args.split)}
    asr = ASRAdapter()
    asr.prepare()
    encoder = LocalEncoder(ROOT / "rag/models/enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    generator = None if args.skip_elm else ELMGenerator()
    parts = (asr, encoder, retriever, data, SimpleContextBuilder(), generator)

    if args.warmup and qids:
        audio = data_dir / items[qids[0]]["audio_file"]
        run_once(parts, audio, args.k)

    rows = []
    for i, qid in enumerate(qids, 1):
        audio = data_dir / items[qid]["audio_file"]
        row = {"query_id": qid, "split": args.split, **run_once(parts, audio, args.k)}
        rows.append(row)
        print(f"[{i}/{len(qids)}] {qid} e2e={row['timing']['end_to_end_ms']:.0f}ms", flush=True)

    keys = ("asr_ms", "embedding_ms", "retrieval_ms", "context_ms", "generation_wall_ms", "end_to_end_ms")
    summary = {
        "split": args.split,
        "queries": len(rows),
        "skip_elm": args.skip_elm,
        "mean_timing_ms": {k: statistics.mean(r["timing"][k] for r in rows) for k in keys},
        "median_timing_ms": {k: statistics.median(r["timing"][k] for r in rows) for k in keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {"summary": summary, "per_query": rows}
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
