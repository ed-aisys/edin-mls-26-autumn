import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "selection_summary", Path(__file__).resolve().parents[1]
    / "extensions/summarize_selection_experiment.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_split_prefix_medians_and_existing_output(self):
        for split, location in (("practice", "answers"), ("dev", "")):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = root / "run"
                (run / "answers").mkdir(parents=True)
                (run / "plan").mkdir()
                corpus = root / "rag/data/rag_corpus"
                corpus.mkdir(parents=True)
                (corpus / f"qrels_{split}.tsv").write_text(
                    "query_id\trecord_id\tscore\nq1\td1\t1\nq2\td1\t1\n")
                (run / "plan/config.json").write_text(json.dumps(
                    {"queries": 2, "query_ids": ["q1", "q2"]}))
                for arm in summary.ARMS:
                    rows = [{
                        "query_id": qid, "split": split,
                        "local_input_tokens": 10, "retrieved_record_ids": ["d1"],
                        "selected_documents": [{"record_id": "d1", "truncated": False}],
                        "finish_reason": "stop",
                        "timing": {"reranking_ms": 0, "generation_ms": ms,
                                   "selection_plus_generation_ms": ms},
                    } for qid, ms in (("q1", 10), ("q2", 30))]
                    (run / "answers" / f"{arm}.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows))
                    judge = run / location / f"judge_json_{arm}"
                    judge.mkdir()
                    (judge / "answer_judge_per_query.jsonl").write_text(
                        "".join(json.dumps({"query_id": qid, "correctness": 2,
                                            "grounding": 2, "completeness": 2,
                                            "normalised_answer_score": 0.5}) + "\n"
                                for qid in ("q1", "q2")))
                output = run / "comparison.json"
                output.write_text("{}")
                argv = ["summary", "--run", str(run), "--output", str(output),
                        "--judge-prefix", "judge_json_"]
                with patch.object(summary, "ROOT", root), patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary.main()
                result = json.loads(output.read_text())
                self.assertTrue(result["judge_available"])
                self.assertEqual(result["split"], split)
                expected = "Exploratory practice" if split == "practice" else "Held-out dev"
                self.assertTrue(result["limitations"][0].startswith(expected))
                self.assertEqual(result["per_arm"]["dense5"]["median_generation_ms"], 20)
                self.assertEqual(result["paired"]["dense20 minus dense5"]
                                 ["query_bootstrap_95_interval"], [0, 0])


if __name__ == "__main__":
    unittest.main()
