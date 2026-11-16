import json
import csv
import hashlib
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "rag"))
from src.data import CourseData, load_jsonl
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k

if len(sys.argv) == 1:
    rows = load_jsonl(root / "rag/results/diagnostic-retrieval-3713758/rankings.jsonl")
    # The frozen diagnostic already records every positive qrel's rank, including
    # None for absent candidates, so corpus text is not needed for this ceiling.
    oracle = [min(5, sum(v is not None for v in r["positive_qrel_ranks"].values())) /
              len(r["positive_qrel_ranks"]) for r in rows]
    print(json.dumps({"queries": len(rows), "oracle_mean_Recall@5": sum(oracle) / len(oracle),
                      "dense_mean_Recall@5": sum(r["Recall@5"] for r in rows) / len(rows)}, indent=2))
else:
    summary_path = Path(sys.argv[1])
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    for arm, metrics in result["arms"].items():
        print(arm, json.dumps(metrics))
        if arm != "dense":
            for name in ("Recall@5", "MRR@10", "NDCG@10"):
                print("  vs dense", name, json.dumps(result["paired_vs_dense"][arm][name]))
    if len(sys.argv) > 2 and sys.argv[2] == "--audit":
        run = summary_path.parent
        rows = load_jsonl(run / "rankings.jsonl")
        original_path = root / "rag/results/diagnostic-retrieval-3713758/rankings.jsonl"
        original = {r["query_id"]: r for r in load_jsonl(original_path)}
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == config["input_sha256"]["rankings"]
        assert len(rows) == result["queries"] == 350
        assert {r["query_id"] for r in rows} == set(original)
        qrels = {}
        with (run.parent / "qrels_practice.tsv").open(encoding="utf-8-sig", newline="") as stream:
            for rel in csv.DictReader(stream, delimiter="\t"):
                if float(rel["score"]) > 0:
                    qrels.setdefault(rel["query_id"], {})[rel["record_id"]] = float(rel["score"])
        recomputed = {arm: [] for arm in result["arms"]}
        for row in rows:
            qid = row["query_id"]
            assert row["transcript"] == original[qid]["transcript"]
            assert row["dense"]["record_ids"] == original[qid]["record_ids"]
            rel = qrels[qid]
            for arm in result["arms"]:
                ids = row[arm]["record_ids"]
                assert len(set(ids)) == 100 and set(ids) == set(original[qid]["record_ids"])
                values = {**{f"Recall@{k}": recall_at_k(ids, rel, k) for k in (5, 10, 20, 100)},
                          "MRR@10": mrr_at_k(ids, rel, 10), "NDCG@10": ndcg_at_k(ids, rel, 10)}
                for name, value in values.items():
                    assert math.isclose(value, row[arm]["metrics"][name], abs_tol=1e-12)
                if arm != "dense":
                    score_map = dict(zip(ids, row[arm]["scores"]))
                    expected = sorted(original[qid]["record_ids"], key=lambda rid: (-score_map[rid], original[qid]["record_ids"].index(rid)))
                    assert ids == expected
                recomputed[arm].append(values)
        for arm, values in recomputed.items():
            for name in values[0]:
                assert math.isclose(sum(r[name] for r in values) / len(values), result["arms"][arm][name], abs_tol=1e-12)
        audit = {"passed": True, "queries": len(rows), "unique_candidates_each": 100,
                 "frozen_transcripts_and_dense_rankings_match": True,
                 "all_metrics_recomputed_from_saved_rankings": True,
                 "stable_tie_order_verified": True,
                 "qwen_matched_native_same_top5_order": sum(r["qwen_matched"]["record_ids"][:5] == r["qwen_native2048"]["record_ids"][:5] for r in rows),
                 "matched_shortened_documents": sum(r["matched_prefix_shortened_documents"] for r in rows),
                 "native_truncated_documents": sum(r["qwen_native2048"]["truncated_documents"] for r in rows),
                 "matched_truncated_again_documents": sum(r["qwen_matched"]["truncated_documents"] for r in rows)}
        (run / "independent-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("AUDIT", json.dumps(audit))
