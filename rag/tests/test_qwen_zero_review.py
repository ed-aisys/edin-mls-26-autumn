"""Guard against selective coverage and accidental score leakage in diagnosis."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.prepare_qwen_zero_review import ARMS, choose_sample, coverage_group, triage


class ZeroReviewTests(unittest.TestCase):
    def test_labelled_coverage_does_not_assume_a_single_gold_record(self):
        self.assertEqual(coverage_group({"a", "b"}, ["c"]), "none")
        self.assertEqual(coverage_group({"a", "b"}, ["a", "c"]), "partial")
        self.assertEqual(coverage_group({"a", "b"}, ["b", "a"]), "full")
        with self.assertRaises(ValueError):
            coverage_group(set(), ["a"])

    def test_incomplete_pairs_fail_and_zero_questions_are_retained(self):
        selected = ["a", "b", "c", "d", "e"]
        answers = {arm: {"q": {"retrieved_record_ids": selected}} for arm in ARMS}
        zero = {"correctness": 0, "grounding": 0, "completeness": 0, "normalised_answer_score": 0}
        scores = {arm: {"q": zero} for arm in ARMS}
        ranks = {"q": {name: {"record_ids": selected} for name in ("dense", "qwen_matched")}}
        rows = triage(["q"], answers, scores, {"q": {"a", "z"}}, ranks)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_stratum"], "zero_partial")
        self.assertEqual(rows[0]["arms"][ARMS[1]]["retrieved_positive_qrels"], 1)
        scores[ARMS[1]] = {}
        with self.assertRaises(ValueError):
            triage(["q"], answers, scores, {"q": {"a"}}, ranks)

    def test_diagnostic_sample_has_positive_controls_and_is_reproducible(self):
        rows = [{"query_id": f"{group}-{i}", "review_stratum": group}
                for group in ("zero_none", "zero_partial", "zero_full", "positive_control")
                for i in range(10)]
        sample = choose_sample(rows, 2, 42)
        self.assertEqual(sample, choose_sample(list(reversed(rows)), 2, 42))
        self.assertEqual(len(set(sample)), 8)
        self.assertEqual(sum(q.startswith("positive_control") for q in sample), 2)


if __name__ == "__main__":
    unittest.main()
