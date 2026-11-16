"""Offline audit of complete paired answers, judge caches and score summaries.

Uses the course asset ZIP to reconstruct exact gold evidence for cache-key
verification. Does not issue API calls, rerun models, or change any judgement.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.data import load_jsonl
from src.evaluation.answer_judge import Judge, cache_key

run = ROOT / "rag/results/qwen4-answers-3724290"
config = json.loads((run / "config.json").read_text(encoding="utf-8"))
comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
fingerprint = json.loads((run / "answers/generation_config.json").read_text(encoding="utf-8"))
prepared = json.loads((run / "prepared.json").read_text(encoding="utf-8"))
for name, suffix in (("config", ".json"), ("plans", ".jsonl")):
    assert hashlib.sha256((run / (name + suffix)).read_bytes()).hexdigest() == prepared[name + "_sha256"]
with zipfile.ZipFile(ROOT / "rag-assets.zip") as archive:
    corpus_bytes = archive.read("data/rag_corpus/corpus.jsonl")
    assert hashlib.sha256(corpus_bytes).hexdigest() == config["input_sha256"]["corpus"]
    corpus = {r["record_id"]: r["text"] for r in map(json.loads, corpus_bytes.decode().splitlines())}
    qrels = {}
    for r in csv.DictReader(io.StringIO(archive.read("data/rag_corpus/qrels_practice.tsv").decode("utf-8-sig")), delimiter="\t"):
        if float(r["score"]) > 0:
            qrels.setdefault(r["query_id"], {})[r["record_id"]] = float(r["score"])
plans = {(r["query_id"], r["arm"]): r for r in load_jsonl(run / "plans.jsonl")}
retrieval = {r["query_id"]: r for r in load_jsonl(ROOT / "rag/results/qwen4-retrieval-3724174/full/rankings.jsonl")}
ids = config["query_ids"]
arms = ("dense5", "qwen4_matched5")
dimensions = ("correctness", "grounding", "completeness")
judged, answers = {}, {}
audit = {"passed": True, "queries": 350, "all_frozen_plans_and_prompt_hashes_verified": True,
         "all_cache_keys_verified_against_current_answers_and_gold": True,
         "all_scores_matched_to_raw_judge_responses": True, "arms": {}}
for arm in arms:
    source = load_jsonl(run / "answers" / (arm + ".jsonl"))
    answers[arm] = {r["query_id"]: r for r in source}
    data = load_jsonl(run / ("judge_json_" + arm) / "answer_judge_per_query.jsonl")
    judged[arm] = {r["query_id"]: r for r in data}
    cached_rows = load_jsonl(run / ("judge_json_" + arm) / "judge_cache.jsonl")
    cache = {r["query_id"]: r for r in cached_rows}
    assert len(source) == len(data) == len(cached_rows) == len(answers[arm]) == len(judged[arm]) == len(cache) == 350
    assert set(answers[arm]) == set(judged[arm]) == set(cache) == set(ids)
    raw = collections.defaultdict(list)
    for row in load_jsonl(run / ("judge_json_" + arm) / "judge_raw.jsonl"):
        try:
            raw[row["question"], row["candidate_sha256"]].append(Judge.parse(row["raw"]))
        except (ValueError, KeyError, TypeError):
            pass
    for qid in ids:
        a, j, c = answers[arm][qid], judged[arm][qid], cache[qid]
        plan = plans[qid, arm]
        assert a["finish_reason"] == "stop" and a["answer"].strip()
        assert hashlib.sha256(a["prompt"].encode()).hexdigest() == a["prompt_sha256"] == plan["prompt_sha256"]
        assert a["transcript"] == plan["transcript"] == retrieval[qid]["transcript"]
        order = retrieval[qid]["dense" if arm == "dense5" else "qwen_matched"]["record_ids"][:5]
        assert a["retrieved_record_ids"] == order == plan["retrieved_record_ids"]
        gold = [{"record_id": rid, "relevance": score, "text": corpus[rid]}
                for rid, score in sorted(qrels[qid].items(), key=lambda pair: (-pair[1], pair[0]))]
        model_tag = config["judge_model"] + "@" + fingerprint["endpoint"] + "#response_format=json_object"
        assert cache_key(model_tag, a["transcript"], a["answer"], gold, 0) == c["cache_key"]
        assert c["result"] in raw[a["transcript"], hashlib.sha256(a["answer"].encode()).hexdigest()]
        for dim in dimensions:
            assert j[dim] == c["result"][dim] and 0 <= j[dim] <= 4
        assert math.isclose(j["normalised_answer_score"], sum(j[d] for d in dimensions) / 12, abs_tol=1e-12)
    means = {dim: statistics.mean(j[dim] for j in judged[arm].values()) for dim in (*dimensions, "normalised_answer_score")}
    for dim, value in means.items():
        assert math.isclose(value, comparison["arms"][arm]["mean_" + dim], abs_tol=1e-12)
    audit["arms"][arm] = {"means": means,
        "all_zero_answers": sum(j["normalised_answer_score"] == 0 for j in judged[arm].values()),
        "full_score_answers": sum(j["normalised_answer_score"] == 1 for j in judged[arm].values())}
audit["zero_to_positive"] = sum(judged[arms[0]][q]["normalised_answer_score"] == 0 < judged[arms[1]][q]["normalised_answer_score"] for q in ids)
audit["positive_to_zero"] = sum(judged[arms[1]][q]["normalised_answer_score"] == 0 < judged[arms[0]][q]["normalised_answer_score"] for q in ids)
audit["same_ordered_prompt_queries"] = sum(answers[arms[0]][q]["prompt_sha256"] == answers[arms[1]][q]["prompt_sha256"] for q in ids)
audit["retrieval_change_vs_answer_change"] = {}
for qid in ids:
    rd = retrieval[qid]["qwen_matched"]["metrics"]["Recall@5"] - retrieval[qid]["dense"]["metrics"]["Recall@5"]
    ad = judged[arms[1]][qid]["normalised_answer_score"] - judged[arms[0]][qid]["normalised_answer_score"]
    label = lambda d: "improved" if d > 1e-12 else "worse" if d < -1e-12 else "tied"
    group = audit["retrieval_change_vs_answer_change"].setdefault(label(rd), {"improved": 0, "worse": 0, "tied": 0})
    group[label(ad)] += 1
audit["selected_cases"] = {}
for qid in ("q_371c6ca10bf1bf5e", "q_9d78dff597245697", "q_890a6df9ce937b08", "q_9a684db4ff009865", "q_0a12138b81eb7e2a"):
    audit["selected_cases"][qid] = {"question": answers[arms[0]][qid]["transcript"],
        "gold_documents": [{"record_id": rid, "text": corpus[rid]} for rid in qrels[qid]],
        **{arm: {"answer": answers[arm][qid]["answer"], "score": judged[arm][qid]["normalised_answer_score"],
                 "recall_at_5": retrieval[qid]["dense" if arm == "dense5" else "qwen_matched"]["metrics"]["Recall@5"],
                 "judge": judged[arm][qid]["judge_explanations"]} for arm in arms}}
rng = random.Random(20261116)
samples = [rng.choices(ids, k=350) for _ in range(2000)]
for metric in (*dimensions, "normalised_answer_score"):
    deltas = {q: judged[arms[1]][q][metric] - judged[arms[0]][q][metric] for q in ids}
    estimates = sorted(statistics.mean(deltas[q] for q in sample) for sample in samples)
    recorded = comparison["paired_qwen4_minus_dense"][metric]
    assert math.isclose(statistics.mean(deltas.values()), recorded["mean_delta"], abs_tol=1e-12)
    for actual, expected in zip((estimates[50], estimates[1949]), recorded["query_bootstrap_95pct_interval"]):
        assert math.isclose(actual, expected, abs_tol=1e-12)
    assert recorded["improved"] == sum(v > 1e-12 for v in deltas.values())
    assert recorded["worse"] == sum(v < -1e-12 for v in deltas.values())
audit["paired_intervals_and_change_counts_recomputed"] = True
identical = [q for q in ids if answers[arms[0]][q]["answer"] == answers[arms[1]][q]["answer"]]
audit["identical_answer_pairs"] = len(identical)
audit["different_scores_on_identical_answers"] = sum(
    judged[arms[0]][q]["normalised_answer_score"] != judged[arms[1]][q]["normalised_answer_score"] for q in identical)
(run / "answer-evaluation-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({k: v for k, v in audit.items() if k != "selected_cases"}, ensure_ascii=False, indent=2))
