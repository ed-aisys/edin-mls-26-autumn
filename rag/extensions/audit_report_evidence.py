"""Audit saved outputs only; never calls generation or judge services."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("dense5", "dense20", "rerank5", "dense_budget", "rerank_budget")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    output = {"definition": "Positive opening: first explanation sentence contains the whole word correctly or accurately; not incorrectly.",
              "splits": {}, "input_files_sha256": {}}
    for split, run in (("dev", "dev-validation"), ("practice", "selection-run-full")):
        base = ROOT / "rag/results" / run / "answers"
        qrels = {}
        path = ROOT / "rag/data/rag_corpus" / f"qrels_{split}.tsv"
        for row in csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"):
            if float(row.get("score", 1)) > 0:
                qrels.setdefault(row["query_id"], set()).add(row["record_id"])
        answers, judges, stats = {}, {}, {}
        for arm in ARMS:
            ap = base / f"{arm}.jsonl"
            jp = base / f"judge_json_{arm}/answer_judge_per_query.jsonl"
            for path in (ap, jp):
                output["input_files_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            answers[arm] = {r["query_id"]: r for r in rows(ap)}
            judges[arm] = {r["query_id"]: r for r in rows(jp)}
            assert answers[arm].keys() == judges[arm].keys()
            scores, recalls = [], []
            zero_positive = 0
            for qid, row in answers[arm].items():
                judge = judges[arm][qid]
                scores.append(judge["normalised_answer_score"])
                recalls.append(len(set(row["retrieved_record_ids"]) & qrels[qid]) / len(qrels[qid]))
                first_sentence = " ".join(judge["judge_explanations"]).split(".")[0]
                zero_positive += judge["normalised_answer_score"] == 0 and bool(
                    re.search(r"\b(correctly|accurately)\b", first_sentence, re.I))
            stats[arm] = {
                "queries": len(scores),
                "zero_score_answers": scores.count(0),
                "zero_score_fraction": scores.count(0) / len(scores),
                "zero_score_positive_opening": zero_positive,
                "score_histogram": {
                    "zero": scores.count(0),
                    "above_zero_below_0_75": sum(0 < s < 0.75 for s in scores),
                    "at_least_0_75": sum(s >= 0.75 for s in scores),
                },
                "recall_score_pearson_r": statistics.correlation(recalls, scores),
                "max_generation_ms": max(r["timing"]["generation_ms"] for r in answers[arm].values()),
            }
        prefix = shared_prefix = different = large_difference = 0
        for qid, row in answers["dense_budget"].items():
            ids = row["retrieved_record_ids"]
            prefix += ids == answers["dense20"][qid]["retrieved_record_ids"][:len(ids)]
            common = min(len(ids), len(answers["dense20"][qid]["retrieved_record_ids"]))
            shared_prefix += ids[:common] == answers["dense20"][qid]["retrieved_record_ids"][:common]
            delta = abs(judges["dense_budget"][qid]["normalised_answer_score"]
                        - judges["dense20"][qid]["normalised_answer_score"])
            different += delta > 1e-12
            large_difference += delta >= 0.4
        output["splits"][split] = {
            "per_arm": stats, "budget_document_id_prefix_queries": prefix,
            "same_dense_order_through_shorter_context_queries": shared_prefix,
            "dense_budget_vs_dense20_different_score_queries": different,
            "dense_budget_vs_dense20_score_gap_at_least_0_4": large_difference,
        }
    dest = ROOT / "rag/results/report-evidence/summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["splits"], indent=2))


if __name__ == "__main__":
    main()
