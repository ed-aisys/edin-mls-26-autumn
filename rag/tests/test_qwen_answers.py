"""Check frozen rerank order becomes full-text top-five generator evidence."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.run_qwen_answer_experiment import build_plan, indexed
from src.context_builder import SimpleContextBuilder


class QwenAnswerTests(unittest.TestCase):
    def test_frozen_selection_changes_order_but_preserves_original_text_and_prompt(self):
        records = {str(i): f"Full document {i}: important material at its end." for i in range(7)}
        row = {"query_id": "q", "transcript": "Question?",
               "dense": {"record_ids": list(records)},
               "qwen_matched": {"record_ids": list(reversed(records)), "rerank_ms": 42}}
        class Data:
            def record_text(self, rid):
                return records[rid]
        dense = build_plan(row, "dense5", records, len)
        qwen = build_plan(row, "qwen4_matched5", records, len)
        self.assertEqual(dense["retrieved_record_ids"], ["0", "1", "2", "3", "4"])
        self.assertEqual(qwen["retrieved_record_ids"], ["6", "5", "4", "3", "2"])
        self.assertEqual(dense["prompt"], SimpleContextBuilder().build(
            "Question?", dense["retrieved_record_ids"], Data())["prompt"])
        self.assertIn(records["6"], qwen["prompt"])
        self.assertTrue(all(not d["truncated"] for d in qwen["selected_documents"]))
        self.assertEqual(qwen["prompt_sha256"], hashlib.sha256(qwen["prompt"].encode()).hexdigest())
        self.assertEqual(qwen["cached_reranking_ms"], 42)

    def test_duplicate_answers_fail_instead_of_silently_overwriting(self):
        with self.assertRaises(ValueError):
            indexed([{"query_id": "q"}, {"query_id": "q"}])


if __name__ == "__main__":
    unittest.main()
