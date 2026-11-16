import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "model_control", Path(__file__).resolve().parents[1]
    / "extensions/compare_reranker_models.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class ModelControlTests(unittest.TestCase):
    def test_metric_cutoffs_are_distinct(self):
        ranked = [f"d{i}" for i in range(1, 101)]
        values = control.metrics(ranked, {"d6": 1})
        self.assertEqual(values["recall_at_5"], 0)
        self.assertAlmostEqual(values["mrr_at_10"], 1 / 6)
        self.assertGreater(values["ndcg_at_10"], 0)

    def test_replacement_counts_qrel_positive_pairs(self):
        before = ["a", "b", "x", "y", "z"]
        after = ["c", "d", "e", "y", "z"]
        value = control.transition(before, after, {"a", "b", "c"})
        self.assertEqual(value["removed_relevant_pairs"], 2)
        self.assertEqual(value["added_relevant_pairs"], 1)
        self.assertEqual(value["recall_direction"], "worsened")
        reverse = control.transition(after, before, {"a", "b", "c"})
        self.assertEqual(reverse["recall_direction"], "improved")

    def test_equal_recall_can_still_replace_relevant_documents(self):
        value = control.transition(["a"], ["b"], {"a", "b"})
        self.assertEqual(value, {"removed_relevant_pairs": 1,
                                 "added_relevant_pairs": 1,
                                 "recall_direction": "unchanged"})

    def test_bootstrap_is_paired_and_reproducible(self):
        self.assertEqual(control.bootstrap([0, 0, 0]), [0, 0])
        self.assertEqual(control.bootstrap([0.25] * 10), [0.25, 0.25])
        values = [-1, 0, 1, 0.5]
        self.assertEqual(control.bootstrap(values), control.bootstrap(values))


if __name__ == "__main__":
    unittest.main()
