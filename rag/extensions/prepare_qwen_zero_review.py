"""Triage all saved answers and freeze blinded diagnostic review packets.

Labels are used only for offline diagnosis. No evaluation rows are removed,
no scores are changed, and this program makes no model or API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("dense5", "qwen4_matched5")


def indexed(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["query_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate query IDs: {path}")
    return result


def coverage_group(relevant, selected):
    hits = len(set(relevant) & set(selected))
    if not relevant:
        raise ValueError("A diagnostic question must have positive qrels")
    return "none" if hits == 0 else "full" if hits == len(set(relevant)) else "partial"


def triage(ids, answers, scores, qrels, retrieval):
    expected = set(ids)
    if len(expected) != len(ids) or set(retrieval) != expected:
        raise ValueError("Require exact query coverage without duplicates")
    for arm in ARMS:
        if set(answers[arm]) != expected or set(scores[arm]) != expected:
            raise ValueError(f"Incomplete or mismatched answer/score coverage: {arm}")
    rows = []
    for qid in ids:
        row = {"query_id": qid, "arms": {}}
        for arm, rank_name in zip(ARMS, ("dense", "qwen_matched")):
            answer, score = answers[arm][qid], scores[arm][qid]
            selected = answer["retrieved_record_ids"]
            if selected != retrieval[qid][rank_name]["record_ids"][:5]:
                raise ValueError(f"Selected evidence differs from frozen rankings: {qid}/{arm}")
            dims = [score[d] for d in ("correctness", "grounding", "completeness")]
            if any(d < 0 or d > 4 for d in dims) or abs(sum(dims) / 12 - score["normalised_answer_score"]) > 1e-12:
                raise ValueError(f"Inconsistent score: {qid}/{arm}")
            row["arms"][arm] = {
                "all_zero": all(d == 0 for d in dims),
                "normalised_answer_score": score["normalised_answer_score"],
                "labelled_top5_coverage": coverage_group(qrels[qid], selected),
                "positive_qrels": len(qrels[qid]),
                "retrieved_positive_qrels": len(set(selected) & set(qrels[qid])),
                "pool100_positive_qrels": len(set(retrieval[qid]["dense"]["record_ids"]) & set(qrels[qid])),
            }
        current = row["arms"][ARMS[1]]
        row["review_stratum"] = "zero_" + current["labelled_top5_coverage"] if current["all_zero"] else "positive_control"
        rows.append(row)
    return rows


def choose_sample(rows, per_stratum, seed):
    if per_stratum < 1:
        raise ValueError("per-stratum must be positive")
    rng = random.Random(seed)
    chosen = []
    for name in ("zero_none", "zero_partial", "zero_full", "positive_control"):
        candidates = sorted(r["query_id"] for r in rows if r["review_stratum"] == name)
        chosen.extend(rng.sample(candidates, min(per_stratum, len(candidates))))
    return chosen


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "rag/results/qwen4-answers-3724290")
    parser.add_argument("--rankings", type=Path, default=ROOT / "rag/results/qwen4-retrieval-3724174/full/rankings.jsonl")
    parser.add_argument("--assets", type=Path, default=ROOT / "rag-assets.zip")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; never overwrites reviewer work")
    parser.add_argument("--per-stratum", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh output directory to preserve frozen reviews")
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    answers = {arm: indexed(args.run / "answers" / (arm + ".jsonl")) for arm in ARMS}
    scores = {arm: indexed(args.run / ("judge_json_" + arm) / "answer_judge_per_query.jsonl") for arm in ARMS}
    retrieval = indexed(args.rankings)
    with zipfile.ZipFile(args.assets) as archive:
        corpus_bytes = archive.read("data/rag_corpus/corpus.jsonl")
        if hashlib.sha256(corpus_bytes).hexdigest() != config["input_sha256"]["corpus"]:
            raise ValueError("Corpus differs from the frozen answer experiment")
        corpus = {r["record_id"]: r["text"] for r in map(json.loads, corpus_bytes.decode().splitlines())}
        qrels = {}
        for r in csv.DictReader(io.StringIO(archive.read("data/rag_corpus/qrels_practice.tsv").decode("utf-8-sig")), delimiter="\t"):
            if float(r["score"]) > 0:
                qrels.setdefault(r["query_id"], set()).add(r["record_id"])
    rows = triage(config["query_ids"], answers, scores, qrels, retrieval)
    chosen = choose_sample(rows, args.per_stratum, args.seed)
    by_qid = {r["query_id"]: r for r in rows}
    rng = random.Random(args.seed)
    packets, key = [], []
    for number, qid in enumerate(chosen, 1):
        order = list(ARMS)
        rng.shuffle(order)
        for position, arm in enumerate(order):
            label = f"Q{number:02d}-{'AB'[position]}"
            a = answers[arm][qid]
            packets.append({
                "label": label, "question": a["transcript"], "answer": a["answer"],
                "selected_evidence": [{"text": corpus[rid]} for rid in a["retrieved_record_ids"]],
                "gold_evidence": [{"text": corpus[rid]} for rid in sorted(qrels[qid])],
                "reviewer_fields": {name: None for name in (
                    "answerable_from_selected_evidence", "core_answer_correct", "unsupported_claims",
                    "conditions_confused", "important_points_missing", "notes")},
            })
            key.append({"label": label, "query_id": qid, "arm": arm,
                        "review_stratum": by_qid[qid]["review_stratum"], "original_judge": scores[arm][qid]})
    sources = [args.run / "config.json", args.rankings]
    sources += [args.run / "answers" / (a + ".jsonl") for a in ARMS]
    sources += [args.run / ("judge_json_" + a) / "answer_judge_per_query.jsonl" for a in ARMS]
    summary = {
        "status": "awaiting_manual_review", "queries_retained": len(rows), "seed": args.seed,
        "sampled_queries": len(chosen), "review_packets": len(packets),
        "strata": dict(Counter(r["review_stratum"] for r in rows)),
        "sampled_strata": dict(Counter(by_qid[q]["review_stratum"] for q in chosen)),
        "arms": {arm: {"all_zero": sum(r["arms"][arm]["all_zero"] for r in rows),
                       "zeros_by_labelled_top5_coverage": dict(Counter(
                           r["arms"][arm]["labelled_top5_coverage"] for r in rows if r["arms"][arm]["all_zero"]))}
                 for arm in ARMS},
        "limitations": ["Qrel coverage is not proof that selected text answers the question.",
                        "Balanced diagnostic sampling does not estimate population error rates.",
                        "Do not use this score-conditioned sample as a benchmark improvement result.",
                        "Review packets hide method labels and judge scores; keep the key separate."],
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    args.output.mkdir(parents=True)
    save(args.output / "triage.json", rows)
    save(args.output / "review-packets.json", packets)
    save(args.output / "review-key.json", key)
    save(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "queries_retained", "sampled_queries", "strata", "arms")}, indent=2))


if __name__ == "__main__":
    main()
