"""Summarise paired pilot outputs; do not turn exploratory scores into proof."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import zipfile

ARMS = ("dense5", "dense20", "rerank5", "dense_budget", "rerank_budget")
ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values):
    return statistics.mean(values)


def interval(values):
    # Query bootstrap; does not estimate variability of generation/judge reruns.
    rng = random.Random(20261112)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return [estimates[125], estimates[4874]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=ROOT / "rag-assets.zip")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--judge-prefix", default="judge_")
    ap.add_argument("--split", choices=("practice", "dev"), default=None,
                    help="Qrel split for recall metrics (default: infer from first answer row)")
    args = ap.parse_args()
    split = args.split
    if split is None:
        for arm in ARMS:
            path = args.run / "answers" / (arm + ".jsonl")
            if path.exists():
                first = rows(path)[0]
                split = first.get("split", "practice")
                break
        split = split or "practice"
    qrels_path = ROOT / "rag/data/rag_corpus" / f"qrels_{split}.tsv"
    if qrels_path.exists():
        raw = qrels_path.read_text(encoding="utf-8-sig")
    else:
        with zipfile.ZipFile(args.assets) as archive:
            raw = archive.read(f"data/rag_corpus/qrels_{split}.tsv").decode("utf-8-sig")
    rels = {}
    for row in csv.DictReader(io.StringIO(raw), delimiter="\t"):
        if float(row.get("score", 1)) > 0:
            rels.setdefault(row["query_id"], set()).add(row["record_id"])
    config = json.loads((args.run / "plan/config.json").read_text(encoding="utf-8"))
    answers, judgements, hashes = {}, {}, {}
    for arm in ARMS:
        path = args.run / "answers" / (arm + ".jsonl")
        data = rows(path)
        answers[arm] = {r["query_id"]: r for r in data}
        if len(data) != len(answers[arm]) or not data:
            raise ValueError("Empty/duplicate answers")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        judge_path = args.run / "answers" / (args.judge_prefix + arm) / "answer_judge_per_query.jsonl"
        if not judge_path.exists():
            judge_path = args.run / (args.judge_prefix + arm) / "answer_judge_per_query.jsonl"
        if judge_path.exists():
            data = rows(judge_path)
            judgements[arm] = {r["query_id"]: r for r in data}
            if len(data) != len(judgements[arm]):
                raise ValueError("Duplicate judgements")
            hashes[str(judge_path)] = hashlib.sha256(judge_path.read_bytes()).hexdigest()
    coverage = set(answers[ARMS[0]])
    if any(set(data) != coverage for data in answers.values()):
        raise ValueError("Arm query coverage differs")
    if not coverage <= set(config["query_ids"]):
        raise ValueError("Unknown query")
    if judgements and (set(judgements) != set(ARMS) or any(set(j) != coverage for j in judgements.values())):
        raise ValueError("Judgement coverage differs")
    per_arm = {}
    for arm, indexed in answers.items():
        data = list(indexed.values())
        observed_usage = [r["usage"] for r in data if r.get("usage") and r["usage"].get("prompt_tokens") is not None]
        per_arm[arm] = {
            "queries": len(data), "mean_local_input_tokens": mean([r["local_input_tokens"] for r in data]),
            "mean_selected_documents": mean([len(r["retrieved_record_ids"]) for r in data]),
            "queries_with_truncated_document": sum(any(d["truncated"] for d in r["selected_documents"]) for r in data),
            "mean_selected_id_qrel_recall": mean([len(set(r["retrieved_record_ids"]) & rels[r["query_id"]]) / len(rels[r["query_id"]]) for r in data]),
            "mean_full_document_qrel_recall": mean([len({d["record_id"] for d in r["selected_documents"] if not d["truncated"]} & rels[r["query_id"]]) / len(rels[r["query_id"]]) for r in data]),
            "mean_reranking_ms": mean([r["timing"]["reranking_ms"] for r in data]),
            "mean_generation_ms": mean([r["timing"]["generation_ms"] for r in data]),
            "median_generation_ms": statistics.median([r["timing"]["generation_ms"] for r in data]),
            "median_attributed_selection_plus_generation_ms": statistics.median([r["timing"]["selection_plus_generation_ms"] for r in data]),
            "usage_coverage": len(observed_usage),
            "server_local_token_mismatches": sum(r["usage"]["prompt_tokens"] != r["local_input_tokens"]
                                                 for r in data if r.get("usage") and r["usage"].get("prompt_tokens") is not None),
            "server_budget_overruns": sum(r["usage"]["prompt_tokens"] > r["input_budget"]
                                          for r in data if r.get("input_budget") is not None and r.get("usage")
                                          and r["usage"].get("prompt_tokens") is not None),
            "max_local_input_tokens": max(r["local_input_tokens"] for r in data),
            "non_stop_completions": sum(r["finish_reason"] != "stop" for r in data),
        }
        if observed_usage:
            per_arm[arm]["mean_server_input_tokens"] = mean([u["prompt_tokens"] for u in observed_usage])
            per_arm[arm]["mean_server_output_tokens"] = mean([u["completion_tokens"] for u in observed_usage])
        if judgements:
            for metric in ("correctness", "grounding", "completeness", "normalised_answer_score"):
                per_arm[arm]["mean_judge_" + metric] = mean([r[metric] for r in judgements[arm].values()])
    paired = {}
    for first, second in (("dense20", "dense5"), ("rerank5", "dense5"), ("rerank_budget", "dense_budget")):
        comparisons = []
        for qid in config["query_ids"]:
            if qid not in coverage:
                continue
            a, b = answers[first][qid], answers[second][qid]
            row = {"query_id": qid, "local_input_token_delta": a["local_input_tokens"] - b["local_input_tokens"],
                   "selection_plus_generation_ms_delta": a["timing"]["selection_plus_generation_ms"] - b["timing"]["selection_plus_generation_ms"]}
            if judgements:
                row["judge_score_delta"] = judgements[first][qid]["normalised_answer_score"] - judgements[second][qid]["normalised_answer_score"]
            comparisons.append(row)
        result = {"query_differences": comparisons}
        if judgements:
            deltas = [r["judge_score_delta"] for r in comparisons]
            result.update(mean_judge_score_delta=mean(deltas), query_bootstrap_95_interval=interval(deltas),
                          wins=sum(v > 0 for v in deltas), ties=sum(v == 0 for v in deltas), losses=sum(v < 0 for v in deltas))
        paired[first + " minus " + second] = result
    output = {"split": split, "queries": len(coverage), "expected_queries": config["queries"],
              "full_sample_completed": len(coverage) == config["queries"],
              "judge_available": bool(judgements), "per_arm": per_arm, "paired": paired,
              "judge_directory_prefix": args.judge_prefix,
              "reranking_cost_provenance": config.get("reused_reranking"),
              "input_files_sha256": hashes,
              "limitations": [(f"Practice split evaluation ({len(coverage)} queries)" if split == "practice" else "Held-out dev validation")
                               + "; single generation and single judge per answer",
                              "Query bootstrap excludes repeated-generation/judge uncertainty",
                              "Timing is single-pass CPU/API observation; no steady-state speedup claim",
                              "With reused scores, selection-plus-generation is an attributed sum; stages may come from different nodes/runs",
                              "Selected-ID qrel recall can count a truncated document whose useful span is absent",
                              "Full-document qrel recall also does not establish evidence sufficiency"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(coverage), "per_arm": per_arm}, indent=2))


if __name__ == "__main__":
    main()
