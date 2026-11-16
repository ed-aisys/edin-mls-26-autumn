"""Export anonymised answer packets for pairwise blind review (no method labels in review file)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))

from src.data import CourseData, load_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers-dir", type=Path, required=True, help="Directory with arm.jsonl files")
    ap.add_argument("--arms", nargs="+", default=["dense5", "dense_budget", "rerank5"])
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--key-output", type=Path, required=True, help="Keep labels separate for debrief")
    args = ap.parse_args()

    by_qid: dict[str, list[dict]] = {}
    for arm in args.arms:
        path = args.answers_dir / f"{arm}.jsonl"
        for row in load_jsonl(path):
            by_qid.setdefault(row["query_id"], []).append(
                {"arm": arm, "transcript": row["transcript"], "answer": row["answer"]}
            )

    qids = [q for q in by_qid if len(by_qid[q]) == len(args.arms)]
    random.Random(args.seed).shuffle(qids)
    qids = qids[: args.sample]

    data = CourseData(ROOT / "rag/data/rag_corpus")
    review = []
    key = []
    for idx, qid in enumerate(qids, 1):
        cands = by_qid[qid]
        random.Random(args.seed + idx).shuffle(cands)
        for j, cand in enumerate(cands, 1):
            label = f"Q{idx:02d}-{'ABCD'[j-1]}"
            review.append(
                {
                    "label": label,
                    "query_id_hidden": f"hidden_{idx}",
                    "transcript": cand["transcript"],
                    "answer": cand["answer"],
                    "reviewer_fields": {
                        "evidence_sufficient_yes_no": "",
                        "claims_supported_yes_no": "",
                        "conditions_confused_yes_no": "",
                        "judge_seems_fair_yes_no": "",
                        "notes": "",
                    },
                }
            )
            key.append({"label": label, "query_id": qid, "arm": cand["arm"]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.key_output.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(review)} blind packets ({len(qids)} queries) to {args.output}")


if __name__ == "__main__":
    main()
