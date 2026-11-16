"""Prepare frozen evidence plans, then generate paired answers (no ASR).

Preparation does not load qrels. Generate supports a bounded smoke run followed
by strict resume using the same frozen plan and source hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from extensions.evidence_selection import select_context, chat_input_tokens
from src.data import load_jsonl

ARMS = ("dense5", "dense20", "rerank5", "dense_budget", "rerank_budget")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def prepare(args):
    if args.output.exists():
        raise ValueError("Preparation output already exists; choose a new directory")
    sample = load_jsonl(args.sample)
    indexed(sample, "query_id")
    if not sample:
        raise ValueError("Empty sample")
    rankings = indexed(load_jsonl(args.rankings), "query_id")
    records = {row["record_id"]: row["text"] for row in
               load_jsonl(ROOT / "rag/data/rag_corpus/corpus.jsonl")}
    for item in sample:
        rank = rankings[item["query_id"]]
        if item["transcript"] != rank["transcript"]:
            raise ValueError("Frozen transcripts disagree")
        if len(rank["record_ids"]) != 100 or len(set(rank["record_ids"])) != 100:
            raise ValueError("Expected 100 unique frozen candidates")
        if not all(rid in records for rid in rank["record_ids"]):
            raise ValueError("Unknown candidate ID")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from src.generator import SYSTEM_PROMPT
    tokenizer = AutoTokenizer.from_pretrained(str(args.assets / "tokenizer"), local_files_only=True)
    torch.set_num_threads(args.threads)

    def rerank_device():
        if args.rerank_device == "cpu":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise ValueError("--rerank-device cuda requested but CUDA is unavailable")
        return torch.device("cuda")

    def count_tokens(prompt):
        return chat_input_tokens(tokenizer, SYSTEM_PROMPT, prompt)

    if count_tokens("Calibration " * 100) <= count_tokens("Calibration"):
        raise ValueError("Tokenizer calibration failed: longer text must increase count")

    def score(question, ids):
        scores = []
        with torch.inference_mode():
            for start in range(0, len(ids), args.batch_size):
                batch = ids[start:start + args.batch_size]
                inputs = rank_tokenizer([question] * len(batch), [records[rid] for rid in batch],
                                        padding=True, truncation="only_second", max_length=512,
                                        return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                scores.extend(model(**inputs).logits.reshape(-1).tolist())
        if len(scores) != len(ids) or not all(math.isfinite(x) for x in scores):
            raise ValueError("Invalid reranker scores")
        return scores

    asset_manifest = json.loads((args.assets / "manifest.json").read_text(encoding="utf-8"))
    for name, asset in asset_manifest.items():
        for rel, expected in asset["files_sha256"].items():
            if digest(args.assets / name / rel) != expected:
                raise ValueError(f"Asset hash mismatch: {name}/{rel}")
    cached_reranks = None
    cache_provenance = None
    if args.rerank_cache:
        old = json.loads((args.rerank_cache / "config.json").read_text(encoding="utf-8"))
        marker = json.loads((args.rerank_cache / "prepared.json").read_text(encoding="utf-8"))
        for name in ("config", "rerankings"):
            path = args.rerank_cache / (name + (".json" if name == "config" else ".jsonl"))
            if digest(path) != marker[name + "_sha256"]:
                raise ValueError("Rerank cache fingerprint mismatch")
        for p in (args.sample, args.rankings, ROOT / "rag/data/rag_corpus/corpus.jsonl"):
            if old["input_sha256"].get(str(p)) != digest(p):
                raise ValueError("Rerank cache inputs differ")
        if (old["reranker"] != asset_manifest["reranker"] or old["reranker_max_length"] != 512
                or old["reranker_batch_size"] != args.batch_size or old["threads"] != args.threads
                or old.get("reranker_device", "cpu") != args.rerank_device):
            raise ValueError("Rerank cache model/settings differ")
        cached_reranks = indexed(load_jsonl(args.rerank_cache / "rerankings.jsonl"), "query_id")
        if set(cached_reranks) != {r["query_id"] for r in sample}:
            raise ValueError("Rerank cache query coverage differs")
        dtype = old["reranker_dtype"]
        warmup_ms = 0.0
        cache_provenance = {"path": str(args.rerank_cache), "config_sha256": marker["config_sha256"],
                            "rerankings_sha256": marker["rerankings_sha256"],
                            "note": "Reuse only valid reranker scores/times; rebuild all contexts and answers"}
    else:
        device = rerank_device()
        rank_tokenizer = AutoTokenizer.from_pretrained(str(args.assets / "reranker"), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(args.assets / "reranker"), local_files_only=True).to(device).eval()
        dtype = str(next(model.parameters()).dtype)
        # Warmup is explicit and excluded from per-query score time.
        warmup_start = time.perf_counter()
        first = sample[0]
        score(first["transcript"], rankings[first["query_id"]]["record_ids"][:args.batch_size])
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
    args.output.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / "rag/extensions/evidence_selection.py",
                    ROOT / "rag/src/generator.py"]
    config = {"schema": "evidence-selection-pilot-v1", "arms": ARMS,
              "query_ids": [r["query_id"] for r in sample], "queries": len(sample),
              "budget": args.budget, "budget_scope": "full local serialized chat input including system and generation prefix",
              "overflow_policy": "rank-prefix; last document original-character prefix via feasible bisection; stop",
              "tokenizer_note": "Official local tokenizer estimate; ELM server usage is separately recorded",
              "reranker": asset_manifest["reranker"], "tokenizer": asset_manifest["tokenizer"],
              "reranker_max_length": 512, "reranker_truncation": "only_second (document)",
              "reranker_dtype": dtype, "reranker_device": args.rerank_device,
              "reused_reranking": cache_provenance,
              "reranker_batch_size": args.batch_size, "threads": args.threads,
              "warmup_ms": warmup_ms, "model_init_in_timings": False,
              "generation_temperature": 0.1, "request_order_seed": args.seed,
              "expected_generation_model": "utter-project/EuroLLM-22B-Instruct-2512",
              "max_local_input_tokens": args.max_input_tokens,
              "timing_note": "Single-pass rerank/ELM latencies on recorded device; not steady-state cluster throughput",
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "input_sha256": {str(p): digest(p) for p in
                               (args.sample, args.rankings, ROOT / "rag/data/rag_corpus/corpus.jsonl")}}
    save(args.output / "config.json", config)
    with (args.output / "plans.jsonl").open("x", encoding="utf-8") as out, \
         (args.output / "rerankings.jsonl").open("x", encoding="utf-8") as rerank_out:
        for index, item in enumerate(sample, 1):
            qid, question = item["query_id"], item["transcript"]
            ids = rankings[qid]["record_ids"]
            if cached_reranks is not None:
                cached = cached_reranks[qid]
                if len(cached["record_ids"]) != 100 or set(cached["record_ids"]) != set(ids):
                    raise ValueError("Cached candidate IDs differ")
                score_map = dict(zip(cached["record_ids"], cached["scores"], strict=True))
                scores = [score_map[rid] for rid in ids]
                lengths = [cached["original_pair_tokens"][rid] for rid in ids]
                rerank_ms = cached["reranking_ms"]
                if not all(math.isfinite(s) for s in scores):
                    raise ValueError("Invalid cached scores")
            else:
                # Token-length diagnostics are separate from reranker inference timing.
                lengths = [len(rank_tokenizer(question, records[rid], truncation=False)["input_ids"]) for rid in ids]
                start = time.perf_counter()
                scores = score(question, ids)
                rerank_ms = (time.perf_counter() - start) * 1000
            order = sorted(range(len(ids)), key=lambda j: (-scores[j], j))
            reordered = [ids[j] for j in order]
            rerank_out.write(json.dumps({"query_id": qid, "record_ids": reordered,
                                        "scores": [scores[j] for j in order], "reranking_ms": rerank_ms,
                                        "original_pair_tokens": dict(zip(ids, lengths)),
                                        "truncated_pairs": sum(n > 512 for n in lengths)}) + "\n")
            rerank_out.flush()
            arms = list(ARMS)
            random.Random(args.seed + ":" + qid).shuffle(arms)
            for arm in arms:
                start = time.perf_counter()
                built = select_context(question, reordered if arm.startswith("rerank") else ids,
                                       records, count_tokens,
                                       top_k={"dense5": 5, "dense20": 20, "rerank5": 5}.get(arm),
                                       budget=args.budget if arm.endswith("budget") else None)
                selection_ms = (time.perf_counter() - start) * 1000
                if built["local_input_tokens"] > args.max_input_tokens:
                    raise ValueError(f"{qid}/{arm} exceeds preflight input ceiling")
                row = {"query_id": qid, "split": item.get("split", "practice"), "transcript": question, "arm": arm,
                       **built, "reranking_ms": rerank_ms if arm.startswith("rerank") else 0.0,
                       "context_build_ms": selection_ms,
                       "selection_ms": selection_ms + (rerank_ms if arm.startswith("rerank") else 0.0)}
                row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"Prepared [{index}/{len(sample)}] {qid}; rerank={rerank_ms:.1f} ms", flush=True)
    save(args.output / "prepared.json", {"completed": True, "queries": len(sample),
                                          "plans_sha256": digest(args.output / "plans.jsonl"),
                                          "config_sha256": digest(args.output / "config.json"),
                                          "rerankings_sha256": digest(args.output / "rerankings.jsonl")})


def generate(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / "rag/.env")
    from src.generator import ELMGenerator
    prepared = json.loads((args.plan / "prepared.json").read_text(encoding="utf-8"))
    for name in ("plans", "config", "rerankings"):
        suffix = ".json" if name == "config" else ".jsonl"
        if digest(args.plan / (name + suffix)) != prepared[name + "_sha256"]:
            raise ValueError("Frozen preparation file changed")
    config = json.loads((args.plan / "config.json").read_text(encoding="utf-8"))
    for rel, expected in config["source_sha256"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Source changed since plan preparation: " + rel)
    plans = load_jsonl(args.plan / "plans.jsonl")
    keys = [(r["query_id"], r["arm"]) for r in plans]
    if len(keys) != len(set(keys)) or len(keys) != config["queries"] * len(ARMS):
        raise ValueError("Incomplete/duplicate plan")
    generator = ELMGenerator()
    if generator.model != config["expected_generation_model"]:
        raise ValueError("Configured ELM model differs from frozen plan")
    fingerprint = {"plans_sha256": prepared["plans_sha256"], "config_sha256": prepared["config_sha256"],
                   "generation_model": generator.model, "endpoint": str(generator.client.base_url),
                   "temperature": 0.1}
    if args.output.exists():
        if not args.resume:
            raise ValueError("Output exists; use explicit --resume for the same plan")
        if json.loads((args.output / "generation_config.json").read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("Resume configuration differs")
    else:
        if args.resume:
            raise ValueError("Resume output does not exist")
        args.output.mkdir(parents=True)
        save(args.output / "generation_config.json", fingerprint)
    existing = []
    for arm in ARMS:
        path = args.output / (arm + ".jsonl")
        existing.extend(load_jsonl(path) if path.exists() else [])
    done = {(r["query_id"], r["arm"]): r for r in existing}
    if len(done) != len(existing) or not set(done) <= set(keys):
        raise ValueError("Duplicate/unknown saved answers")
    for p in plans:
        prior = done.get((p["query_id"], p["arm"]))
        if prior and (prior["prompt_sha256"] != p["prompt_sha256"] or prior["transcript"] != p["transcript"]):
            raise ValueError("Saved answer does not match frozen prompt")
    pending_ids = [qid for qid in config["query_ids"]
                   if any((qid, arm) not in done for arm in ARMS)]
    if args.max_new_queries:
        pending_ids = pending_ids[:args.max_new_queries]
    for p in plans:
        key = (p["query_id"], p["arm"])
        if p["query_id"] not in pending_ids or key in done:
            continue
        result = generator.generate(p["prompt"])
        if not result["answer"]:
            raise ValueError("Empty answer; stop before spending remaining calls")
        row = {**p, "answer": result["answer"], "usage": result["usage"],
               "response_model": result["response_model"], "finish_reason": result["finish_reason"],
               "timing": {"reranking_ms": p["reranking_ms"], "context_build_ms": p["context_build_ms"],
                          "generation_ms": result["latency_ms"],
                          "selection_plus_generation_ms": p["selection_ms"] + result["latency_ms"]}}
        with (args.output / (p["arm"] + ".jsonl")).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        done[key] = row
        print(f"Generated {len(done)}/{len(plans)} {p['query_id']} {p['arm']} "
              f"local_tokens={p['local_input_tokens']} finish={result['finish_reason']}", flush=True)
        if result["finish_reason"] != "stop":
            raise ValueError("Non-stop completion saved; inspect before resuming")
    summary = {"expected": len(plans), "generated": len(done), "completed": len(done) == len(plans),
               "per_arm": dict(Counter(r["arm"] for r in done.values())),
               "note": "Latency excludes cached ASR/retrieval and model loading; not end-to-end audio latency"}
    save(args.output / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--sample", type=Path, required=True)
    prep.add_argument("--rankings", type=Path, required=True)
    prep.add_argument("--assets", type=Path, default=ROOT / "rag/models/selection")
    prep.add_argument("--rerank-cache", type=Path, default=None)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--budget", type=int, default=4096)
    prep.add_argument("--max-input-tokens", type=int, default=28672)
    prep.add_argument("--batch-size", type=int, default=16)
    prep.add_argument("--threads", type=int, default=2)
    prep.add_argument("--rerank-device", choices=("cpu", "cuda"), default="cpu",
                      help="Device for cross-encoder reranking (use cuda on a GPU node)")
    prep.add_argument("--seed", default="2026-11-12-selection-pilot-v1")
    gen = sub.add_parser("generate")
    gen.add_argument("--plan", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--max-new-queries", type=int, default=None)
    args = ap.parse_args()
    for name in ("budget", "max_input_tokens", "batch_size", "threads", "max_new_queries"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            ap.error(name + " must be positive")
    (prepare if args.stage == "prepare" else generate)(args)


if __name__ == "__main__":
    main()
