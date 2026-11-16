"""Paired dense5/Qwen-4B answers from frozen rankings; no ASR or reranker inference.

The unchanged scripts/evaluate_answers.py supplies judging. This module freezes
plans, generates/resumes answers, and summarises matched public-judge outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.data import CourseData, load_jsonl
from extensions.evidence_selection import select_context, chat_input_tokens

ARMS = ("dense5", "qwen4_matched5")
MODEL = "utter-project/EuroLLM-22B-Instruct-2512"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def indexed(rows):
    result = {r["query_id"]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError("Duplicate query IDs")
    return result


def build_plan(row, arm, records, count_tokens):
    source = "dense" if arm == "dense5" else "qwen_matched"
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    start = time.perf_counter()
    built = select_context(row["transcript"], row[source]["record_ids"], records,
                           count_tokens, top_k=5)
    context_ms = (time.perf_counter() - start) * 1000
    return {"query_id": row["query_id"], "transcript": row["transcript"],
            "split": "practice", "arm": arm, **built,
            "prompt_sha256": hashlib.sha256(built["prompt"].encode()).hexdigest(),
            "context_build_ms": context_ms,
            "cached_reranking_ms": row[source].get("rerank_ms", 0.0)}


def prepare(args):
    if args.run.exists():
        raise ValueError("Choose a fresh run directory")
    source_config = json.loads((args.rankings.parent / "config.json").read_text())
    source_summary = json.loads((args.rankings.parent / "summary.json").read_text())
    if (source_config["model"]["repo"] != "Qwen/Qwen3-Reranker-4B"
            or not source_summary["completed"] or not source_summary["full_split"]):
        raise ValueError("Require a completed full-practice Qwen-4B run")
    corpus = ROOT / "rag/data/rag_corpus/corpus.jsonl"
    if digest(corpus) != source_config["input_sha256"]["corpus"]:
        raise ValueError("Corpus changed since reranking")
    data = CourseData(corpus.parent)
    rows = load_jsonl(args.rankings)
    if set(indexed(rows)) != {r["query_id"] for r in data.audio_items("practice")}:
        raise ValueError("Require exact practice query coverage")
    from transformers import AutoTokenizer
    from src.generator import SYSTEM_PROMPT
    assets = ROOT / "rag/models/selection"
    manifest = json.loads((assets / "manifest.json").read_text())["tokenizer"]
    if manifest["repo"] != MODEL:
        raise ValueError("Generator tokenizer differs")
    for name, expected in manifest["files_sha256"].items():
        if digest(assets / "tokenizer" / name) != expected:
            raise ValueError("Tokenizer hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True)
    def count_tokens(prompt):
        return chat_input_tokens(tokenizer, SYSTEM_PROMPT, prompt)
    plans = []
    for row in rows:
        if (len(set(row["dense"]["record_ids"])) != 100
                or set(row["dense"]["record_ids"]) != set(row["qwen_matched"]["record_ids"])):
            raise ValueError("Candidate set changed")
        order = list(ARMS)
        random.Random("qwen4-answer-v1:" + row["query_id"]).shuffle(order)
        for arm in order:
            plan = build_plan(row, arm, data.records, count_tokens)
            if plan["local_input_tokens"] > 28672:
                raise ValueError("Input exceeds preflight ceiling; no silent truncation")
            plans.append(plan)
    args.run.mkdir(parents=True)
    paths = [Path(__file__), ROOT / "rag/src/generator.py", ROOT / "rag/extensions/evidence_selection.py",
             ROOT / "rag/scripts/evaluate_answers.py", ROOT / "rag/src/evaluation/answer_judge.py"]
    config = {"schema": "qwen4-paired-answers-v1", "arms": ARMS, "queries": len(rows),
              "query_ids": [r["query_id"] for r in rows], "generator_model": MODEL,
              "generation_temperature": 0.1, "judge_model": MODEL,
              "judge_json_mode": True, "judge_repeats": 1,
              "selection": "Original full text of top 5 records, identical baseline prompt, no context truncation",
              "qwen_ranking_arm": "qwen_matched", "source_reranker": source_config["model"],
              "tokenizer": manifest, "system_prompt": SYSTEM_PROMPT,
              "input_sha256": {"rankings": digest(args.rankings), "corpus": digest(corpus)},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
              "no_asr_or_reranker_inference": True, "qrels_not_loaded_during_selection": True,
              "timing_note": "Fresh generation time measured; rerank time attributed from prior retrieval run; not end-to-end latency"}
    save(args.run / "config.json", config)
    with (args.run / "plans.jsonl").open("x", encoding="utf-8") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, ensure_ascii=False) + "\n")
    save(args.run / "prepared.json", {"config_sha256": digest(args.run / "config.json"),
                                      "plans_sha256": digest(args.run / "plans.jsonl")})
    print("ANSWER_PLANS_READY", len(plans), flush=True)


def check_plan(run):
    marker = json.loads((run / "prepared.json").read_text())
    for name in ("config", "plans"):
        suffix = ".json" if name == "config" else ".jsonl"
        if digest(run / (name + suffix)) != marker[name + "_sha256"]:
            raise ValueError("Frozen plan changed")
    config = json.loads((run / "config.json").read_text())
    for name, expected in config["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Source changed: " + name)
    plans = load_jsonl(run / "plans.jsonl")
    keys = {(p["query_id"], p["arm"]) for p in plans}
    expected = {(q, arm) for q in config["query_ids"] for arm in ARMS}
    if keys != expected or len(plans) != len(expected):
        raise ValueError("Missing/duplicate frozen plans")
    return config, plans


def generate(args):
    config, plans = check_plan(args.run)
    from dotenv import load_dotenv
    load_dotenv(ROOT / "rag/.env")
    from src.generator import ELMGenerator
    generator = ELMGenerator()
    if generator.model != MODEL:
        raise ValueError("Configured generator differs from frozen model")
    fingerprint = {"plans_sha256": digest(args.run / "plans.jsonl"),
                   "config_sha256": digest(args.run / "config.json"),
                   "model": generator.model, "endpoint": str(generator.client.base_url), "temperature": 0.1}
    output = args.run / "answers"
    if output.exists():
        if not args.resume or json.loads((output / "generation_config.json").read_text()) != fingerprint:
            raise ValueError("Resume requires identical frozen generation settings")
    else:
        if args.resume:
            raise ValueError("Cannot resume missing output")
        output.mkdir()
        save(output / "generation_config.json", fingerprint)
    done = {}
    for arm in ARMS:
        path = output / (arm + ".jsonl")
        for row in load_jsonl(path) if path.exists() else []:
            key = (row["query_id"], row["arm"])
            if key in done or row["arm"] != arm:
                raise ValueError("Duplicate/mismatched saved answer")
            done[key] = row
    keys = {(p["query_id"], p["arm"]) for p in plans}
    if not set(done) <= keys:
        raise ValueError("Unexpected saved answer")
    for plan in plans:
        prior = done.get((plan["query_id"], plan["arm"]))
        if prior and (prior["prompt_sha256"] != plan["prompt_sha256"] or
                      prior["transcript"] != plan["transcript"] or prior["finish_reason"] != "stop"):
            raise ValueError("Saved answer requires inspection before resume")
    pending = [q for q in config["query_ids"] if any((q, arm) not in done for arm in ARMS)]
    if args.max_new_queries:
        pending = pending[:args.max_new_queries]
    pending = set(pending)
    for plan in plans:
        key = (plan["query_id"], plan["arm"])
        if key in done or plan["query_id"] not in pending:
            continue
        result = generator.generate(plan["prompt"])
        if not result["answer"]:
            raise ValueError("Empty answer; stop before more API calls")
        row = {**plan, **result, "generation_ms": result["latency_ms"],
               "attributed_selection_plus_generation_ms": result["latency_ms"] +
                   plan["context_build_ms"] + plan["cached_reranking_ms"]}
        with (output / (plan["arm"] + ".jsonl")).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        done[key] = row
        print(f"Generated {len(done)}/{len(plans)} {plan['query_id']} {plan['arm']} "
              f"tokens={plan['local_input_tokens']} finish={result['finish_reason']}", flush=True)
        if result["finish_reason"] != "stop":
            raise ValueError("Non-stop completion saved; inspect before continuing")
    save(output / "generation_summary.json", {"completed": len(done) == len(plans),
         "generated": len(done), "expected": len(plans), "per_arm": dict(Counter(r["arm"] for r in done.values()))})


def smoke(args):
    config, _ = check_plan(args.run)
    qids = config["query_ids"][:2]
    for arm in ARMS:
        answers = indexed(load_jsonl(args.run / "answers" / (arm + ".jsonl")))
        with (args.run / "answers" / ("smoke_" + arm + ".jsonl")).open("w", encoding="utf-8") as stream:
            for qid in qids:
                stream.write(json.dumps(answers[qid], ensure_ascii=False) + "\n")


def summarise(args):
    config, _ = check_plan(args.run)
    result = {"queries": config["queries"], "completed": True, "arms": {}, "paired_qwen4_minus_dense": {},
              "limitations": ["Practice exploratory comparison", "One generation and one judge call per answer",
                              "Same model generates and judges; public qrel evidence is not exhaustive",
                              "Query bootstrap excludes generation/judge rerun uncertainty",
                              "Attributed rerank cost is not measured end-to-end latency"]}
    judged = {}
    for arm in ARMS:
        answers = indexed(load_jsonl(args.run / "answers" / (arm + ".jsonl")))
        judge_dir = args.run / ("judge_json_" + arm)
        judged[arm] = indexed(load_jsonl(judge_dir / "answer_judge_per_query.jsonl"))
        summary = json.loads((judge_dir / "answer_judge_summary.json").read_text())
        if (set(answers) != set(config["query_ids"]) or set(judged[arm]) != set(answers)
                or summary["judge_model"] != MODEL or summary["judge_response_format"] != "json_object"
                or summary["repeats"] != 1 or summary["is_subset"]):
            raise ValueError("Incomplete or incompatible generation/judging outputs")
        values = list(answers.values())
        tokens = [r["usage"]["prompt_tokens"] for r in values if r.get("usage") and r["usage"].get("prompt_tokens") is not None]
        result["arms"][arm] = {k: v for k, v in summary.items() if k.startswith("mean_")}
        result["arms"][arm].update({"mean_local_input_tokens": statistics.mean(r["local_input_tokens"] for r in values),
            "server_usage_coverage": len(tokens), "mean_server_input_tokens": statistics.mean(tokens) if tokens else None,
            "mean_generation_ms": statistics.mean(r["generation_ms"] for r in values),
            "median_generation_ms": statistics.median(r["generation_ms"] for r in values),
            "mean_attributed_selection_plus_generation_ms": statistics.mean(r["attributed_selection_plus_generation_ms"] for r in values)})
    rng = random.Random(20261116)
    indices = [rng.choices(config["query_ids"], k=config["queries"]) for _ in range(2000)]
    for metric in ("correctness", "grounding", "completeness", "normalised_answer_score"):
        deltas = {q: judged[ARMS[1]][q][metric] - judged[ARMS[0]][q][metric] for q in config["query_ids"]}
        estimates = sorted(statistics.mean(deltas[q] for q in sample) for sample in indices)
        result["paired_qwen4_minus_dense"][metric] = {"mean_delta": statistics.mean(deltas.values()),
            "query_bootstrap_95pct_interval": [estimates[50], estimates[1949]],
            "improved": sum(v > 1e-12 for v in deltas.values()),
            "worse": sum(v < -1e-12 for v in deltas.values()), "tied": sum(abs(v) <= 1e-12 for v in deltas.values())}
    save(args.run / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print("QWEN4_ANSWER_EVALUATION_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "generate", "smoke", "summarise"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rankings", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-new-queries", type=int)
    args = ap.parse_args()
    if args.stage == "prepare" and args.rankings is None:
        ap.error("prepare requires --rankings")
    if args.max_new_queries is not None and args.max_new_queries < 1:
        ap.error("max-new-queries must be positive")
    {"prepare": prepare, "generate": generate, "smoke": smoke, "summarise": summarise}[args.stage](args)


if __name__ == "__main__":
    main()
