"""Collect CUDA vs PyTorch-reference ASR transcripts for the same audio queries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))

from src.asr_adapter import ASRAdapter
from src.data import CourseData, load_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {r["query_id"]: r["transcript"] for r in load_jsonl(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cuda-out", type=Path, required=True)
    ap.add_argument("--reference-out", type=Path, required=True)
    ap.add_argument("--cuda-groups", default="all")
    ap.add_argument("--reference-groups", default="none", help="PyTorch reference ASR per hw1-asr")
    args = ap.parse_args()

    data = CourseData(ROOT / "rag/data/rag_corpus")
    items = data.audio_items(args.split)
    if args.limit:
        items = items[: args.limit]

    cuda_cache = load_existing(args.cuda_out)
    ref_cache = load_existing(args.reference_out)

    def transcribe_block(cuda_groups: str, cache: dict) -> list[dict]:
        adapter = ASRAdapter(cuda_groups=cuda_groups)
        adapter.prepare()
        rows = []
        for i, item in enumerate(items, 1):
            qid = item["query_id"]
            if qid in cache:
                transcript = cache[qid]
            else:
                audio = data.data_dir / item["audio_file"]
                transcript = adapter.transcribe(audio)
            rows.append({"query_id": qid, "transcript": transcript, "split": args.split})
            print(f"[{cuda_groups}] [{i}/{len(items)}] {qid}", flush=True)
        adapter.close()
        return rows

    if not cuda_cache or len(cuda_cache) < len(items):
        write_jsonl(args.cuda_out, transcribe_block(args.cuda_groups, cuda_cache))
    if not ref_cache or len(ref_cache) < len(items):
        write_jsonl(args.reference_out, transcribe_block(args.reference_groups, ref_cache))

    print(f"Wrote {args.cuda_out} and {args.reference_out}")


if __name__ == "__main__":
    main()
