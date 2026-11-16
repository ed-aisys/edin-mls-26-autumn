"""Pinned Qwen 0.6B vs frozen dense/MiniLM rankings; no ASR or ELM calls.

Run prepare on a network-enabled login node; run inference only under Slurm.
Primary arm: Qwen on the original-character prefix visible to MiniLM at 512
pair tokens. Secondary arm: original document, capped at 2048 Qwen tokens
including the official template. These are distinct, predeclared comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. Note that the answer '
          'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
KEYS = ("Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10")
ARMS = ("dense", "minilm_512", "qwen_matched", "qwen_native2048")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metrics(ids, relevance):
    return {**{f"Recall@{k}": recall_at_k(ids, relevance, k) for k in (5, 10, 20, 100)},
            "MRR@10": mrr_at_k(ids, relevance, 10),
            "NDCG@10": ndcg_at_k(ids, relevance, 10)}


def ranked(ids, scores):
    if len(ids) != len(scores) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Invalid scores")
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], i))]


def visible_prefix(tokenizer, question, document):
    pair = tokenizer(question, document, truncation="only_second", max_length=512,
                     return_offsets_mapping=True)
    ends = [end for seq, (_, end) in zip(pair.sequence_ids(), pair["offset_mapping"])
            if seq == 1]
    return document[:max(ends, default=0)]


def qwen_input(tokenizer, question, document, max_tokens=2048, forbid_truncation=False):
    # Keep the official single-string formatting; never truncate the question.
    head = (f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: ")
    content = tokenizer.encode(head + document, add_special_tokens=False)
    head_tokens = tokenizer.encode(head, add_special_tokens=False)
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    available = max_tokens - len(prefix) - len(suffix)
    if len(head_tokens) >= available:
        raise ValueError("Question and instruction exceed input limit")
    truncated = len(content) > available
    if truncated and forbid_truncation:
        raise ValueError("Matched document would be truncated again by Qwen")
    ids = prefix + content[:available] + suffix
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}, truncated


def prepare(args):
    from huggingface_hub import snapshot_download
    args.model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=args.model_dir,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
                      max_workers=2)
    files = {str(p.relative_to(args.model_dir)): digest(p)
             for p in sorted(args.model_dir.rglob("*"))
             if p.is_file() and ".cache" not in p.parts and p.name != "experiment_manifest.json"}
    if not any(p.endswith(".safetensors") for p in files):
        raise ValueError("Model weights missing")
    write_json(args.model_dir / "experiment_manifest.json",
               {"repo": MODEL, "revision": REVISION, "files_sha256": files})
    print("QWEN_ASSETS_READY", REVISION, flush=True)


def run(args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    if args.batch_size < 1 or args.bootstrap < 1:
        raise ValueError("Batch size and bootstrap count must be positive")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; submit this run through Slurm")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261116)
    data = CourseData(ROOT / "rag/data/rag_corpus")
    rows = load_jsonl(args.rankings)
    qrels = data.load_qrels("practice")
    expected = {r["query_id"] for r in data.audio_items("practice")}
    seen = [r["query_id"] for r in rows]
    if len(set(seen)) != len(seen) or (set(seen) != expected and not args.allow_subset):
        raise ValueError("Frozen rankings must cover practice exactly (or explicitly allow subset)")
    if not set(seen) <= expected:
        raise ValueError("Unexpected query IDs")
    if args.limit:
        rows = rows[:args.limit]
    for row in rows:
        ids = row["record_ids"]
        if len(ids) != 100 or len(set(ids)) != 100 or not all(i in data.records for i in ids):
            raise ValueError("Expected 100 unique, known candidates")
        if not row["transcript"].strip():
            raise ValueError("Empty frozen transcript")
    manifest = json.loads((args.model_dir / "experiment_manifest.json").read_text())
    if manifest["repo"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("Unexpected Qwen model/revision")
    for name, expected_hash in manifest["files_sha256"].items():
        if digest(args.model_dir / name) != expected_hash:
            raise ValueError("Qwen asset hash mismatch: " + name)
    mini_manifest = json.loads((args.minilm_dir.parent / "manifest.json").read_text())["reranker"]
    for name, expected_hash in mini_manifest["files_sha256"].items():
        if digest(args.minilm_dir / name) != expected_hash:
            raise ValueError("MiniLM asset hash mismatch: " + name)

    args.output.mkdir(parents=True)
    config = {"model": manifest, "minilm": mini_manifest, "split": "practice",
              "queries": len(rows), "full_split": len(rows) == len(expected), "candidate_pool": 100,
              "arms": ARMS, "primary_qwen_arm": "qwen_matched",
              "instruction": INSTRUCTION, "prefix": PREFIX, "suffix": SUFFIX,
              "input_policy": {"minilm_512": "Original document; only_second pair truncation at 512",
                               "qwen_matched": "Original character prefix visible under MiniLM 512-token pair; no second truncation",
                               "qwen_native2048": "Original document; 2048 Qwen tokens including template; keep question"},
              "batch_size": args.batch_size, "python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(), "qwen_dtype": "bfloat16", "minilm_dtype": "float32",
              "scoring": "Qwen last-position yes logit minus no logit; descending, ties use frozen dense order",
              "timing_scope": "Tokenization, transfer, inference, score copy; warmed; excludes model loading and file IO",
              "no_asr_or_elm_calls": True, "no_qrels_in_scoring": True,
              "bootstrap_seed": 20261116, "bootstrap_replicates": args.bootstrap,
              "input_sha256": {"rankings": digest(args.rankings),
                               "corpus": digest(ROOT / "rag/data/rag_corpus/corpus.jsonl"),
                               "qrels": digest(ROOT / "rag/data/rag_corpus/qrels_practice.tsv"),
                               "script": digest(__file__), "metrics": digest(ROOT / "rag/src/metrics.py")}}
    write_json(args.output / "config.json", config)
    mini_tok = AutoTokenizer.from_pretrained(args.minilm_dir, local_files_only=True, use_fast=True)
    mini = AutoModelForSequenceClassification.from_pretrained(
        args.minilm_dir, local_files_only=True, dtype=torch.float32).eval().cuda()
    q_tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, padding_side="left")
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").eval().cuda()
    yes, no = (q_tok.convert_tokens_to_ids(t) for t in ("yes", "no"))
    if q_tok.encode("yes", add_special_tokens=False) != [yes] or q_tok.encode("no", add_special_tokens=False) != [no]:
        raise ValueError("Decision words are not single tokens")

    @torch.inference_mode()
    def score_mini(question, documents):
        scores = []
        for offset in range(0, len(documents), args.batch_size):
            docs = documents[offset:offset + args.batch_size]
            inputs = mini_tok([question] * len(docs), docs, padding=True,
                              truncation="only_second", max_length=512, return_tensors="pt").to("cuda")
            scores.extend(mini(**inputs).logits.flatten().float().cpu().tolist())
        return scores

    @torch.inference_mode()
    def score_qwen(question, documents, matched=False):
        scores, truncated, lengths = [], 0, []
        for offset in range(0, len(documents), args.batch_size):
            encoded = [qwen_input(q_tok, question, d, forbid_truncation=matched)
                       for d in documents[offset:offset + args.batch_size]]
            truncated += sum(t for _, t in encoded)
            lengths.extend(len(v["input_ids"]) for v, _ in encoded)
            inputs = q_tok.pad([v for v, _ in encoded], padding=True, return_tensors="pt").to("cuda")
            logits = qwen(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            scores.extend((logits[:, yes] - logits[:, no]).cpu().tolist())
        return scores, truncated, lengths

    toy_q = "What is the capital of France?"
    toy_docs = ["Paris is the capital of France.", "Bananas are yellow fruit."]
    toy_scores, _, _ = score_qwen(toy_q, toy_docs)
    single_scores = [score_qwen(toy_q, [d])[0][0] for d in toy_docs]
    if toy_scores[0] <= toy_scores[1] or not np.allclose(toy_scores, single_scores, atol=0.25, rtol=0.02):
        raise ValueError("Qwen scoring direction or padding/batching sanity check failed")
    score_mini(toy_q, toy_docs)
    write_json(args.output / "sanity.json", {"toy_batched_logits": toy_scores,
               "toy_single_logits": single_scores, "positive_above_negative": True,
               "limitation": "Implementation sanity only, not domain accuracy validation"})
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    per_query = []
    start_all = time.perf_counter()
    with (args.output / "rankings.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            question, ids = row["transcript"], row["record_ids"]
            docs = [data.record_text(rid) for rid in ids]
            matched_docs = [visible_prefix(mini_tok, question, d) for d in docs]
            item = {"query_id": row["query_id"], "transcript": question,
                    "matched_prefix_shortened_documents": sum(a != b for a, b in zip(docs, matched_docs)),
                    "dense": {"record_ids": ids, "scores": row.get("scores")}}
            # Alternate order to reduce a systematic warm-cache/order advantage.
            methods = ["minilm_512", "qwen_matched", "qwen_native2048"]
            if index % 2 == 0:
                methods.reverse()
            for arm in methods:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if arm == "minilm_512":
                    values, truncations, lengths = score_mini(question, docs), None, None
                else:
                    values, truncations, lengths = score_qwen(
                        question, matched_docs if arm == "qwen_matched" else docs,
                        matched=arm == "qwen_matched")
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                order = ranked(ids, values)
                score_map = dict(zip(ids, values))
                item[arm] = {"record_ids": order, "scores": [score_map[r] for r in order],
                             "rerank_ms": elapsed, "truncated_documents": truncations,
                             "input_tokens": sum(lengths) if lengths else None}
            # Qrels are used only after all model scores/orderings have been fixed.
            relevance = qrels[row["query_id"]]
            for arm in ARMS:
                item[arm]["metrics"] = metrics(item[arm]["record_ids"], relevance)
                if item[arm]["metrics"]["Recall@100"] != item["dense"]["metrics"]["Recall@100"]:
                    raise ValueError("Candidate coverage changed during reranking")
            item["oracle_Recall@5"] = min(5, len(set(ids) & set(relevance))) / len(relevance)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            per_query.append(item)
            if index % 10 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] " + " ".join(
                    f"{arm} R5={np.mean([r[arm]['metrics']['Recall@5'] for r in per_query]):.4f}"
                    for arm in ARMS), flush=True)

    rng = np.random.default_rng(20261116)
    samples = rng.integers(0, len(rows), size=(args.bootstrap, len(rows)))
    summary = {"completed": True, "queries": len(rows), "full_split": config["full_split"],
               "arms": {}, "paired_vs_dense": {}, "paired_qwen_vs_minilm": {},
               "oracle_mean_Recall@5": float(np.mean([r["oracle_Recall@5"] for r in per_query])),
               "peak_allocated_gpu_gib_all_models": torch.cuda.max_memory_allocated() / 2**30,
               "inference_and_output_wall_seconds": time.perf_counter() - start_all,
               "limitations": ["Practice exploratory comparison, not untouched held-out validation",
                               "One timing pass per query; co-resident models; not isolated serving or end-to-end latency",
                               "Qwen-vs-MiniLM changes architecture, training, tokenizer and precision, not parameter count alone",
                               "No answer generation or answer-quality claim"]}
    for arm in ARMS:
        summary["arms"][arm] = {key: float(np.mean([r[arm]["metrics"][key] for r in per_query])) for key in KEYS}
        if arm != "dense":
            times = [r[arm]["rerank_ms"] for r in per_query]
            summary["arms"][arm].update({"mean_rerank_ms": float(np.mean(times)),
                                         "median_rerank_ms": float(np.median(times)),
                                         "p95_rerank_ms": float(np.percentile(times, 95))})
            for base, target in (("dense", "paired_vs_dense"), ("minilm_512", "paired_qwen_vs_minilm")):
                if arm == base:
                    continue
                summary[target][arm] = {}
                for key in KEYS:
                    delta = np.array([r[arm]["metrics"][key] - r[base]["metrics"][key] for r in per_query])
                    interval = np.percentile(delta[samples].mean(axis=1), [2.5, 97.5])
                    summary[target][arm][key] = {"mean_delta": float(delta.mean()),
                        "bootstrap_95pct_interval": interval.tolist(), "improved": int((delta > 1e-12).sum()),
                        "worse": int((delta < -1e-12).sum()), "tied": int((abs(delta) <= 1e-12).sum())}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("QWEN_RETRIEVAL_EVALUATION_COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "run"])
    ap.add_argument("--model-dir", type=Path, default=ROOT / "rag/models/qwen3-reranker-0.6b")
    ap.add_argument("--minilm-dir", type=Path, default=ROOT / "rag/models/selection/reranker")
    ap.add_argument("--rankings", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--allow-subset", action="store_true")
    args = ap.parse_args()
    if args.stage == "run" and (args.rankings is None or args.output is None):
        ap.error("run requires --rankings and --output")
    (prepare if args.stage == "prepare" else run)(args)


if __name__ == "__main__":
    main()
