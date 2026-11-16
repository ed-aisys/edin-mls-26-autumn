"""No model downloads: ranking metrics and prompt boundary invariants."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.evaluate_qwen_reranker import metrics, ranked, visible_prefix, qwen_input, PREFIX, SUFFIX


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


class RetrievalTests(unittest.TestCase):
    def test_reordering_preserves_pool_coverage_but_changes_early_recall(self):
        ids = [str(i) for i in range(100)]
        relevant = {"1": 1, "40": 2}
        scores = [0.0] * 100
        scores[40], scores[1] = 9.0, 8.0
        dense, rerank = metrics(ids, relevant), metrics(ranked(ids, scores), relevant)
        self.assertEqual(dense["Recall@5"], 0.5)
        self.assertEqual(rerank["Recall@5"], 1.0)
        self.assertEqual(dense["Recall@100"], rerank["Recall@100"])
        self.assertEqual(rerank["MRR@10"], 1.0)
        self.assertAlmostEqual(rerank["NDCG@10"], 1.0)

    def test_dense_order_breaks_ties_and_nan_is_rejected(self):
        self.assertEqual(ranked(["b", "a", "c"], [1, 1, 2]), ["c", "b", "a"])
        with self.assertRaises(ValueError):
            ranked(["a"], [float("nan")])

    def test_truncation_preserves_question_and_official_suffix(self):
        tok = CharacterTokenizer()
        item, truncated = qwen_input(tok, "KEEP_QUESTION", "d" * 4000, max_tokens=700)
        rendered = "".join(map(chr, item["input_ids"]))
        self.assertTrue(truncated)
        self.assertEqual(len(item["input_ids"]), 700)
        self.assertTrue(rendered.startswith(PREFIX))
        self.assertIn("<Query>: KEEP_QUESTION\n<Document>: ", rendered)
        self.assertTrue(rendered.endswith(SUFFIX))
        with self.assertRaises(ValueError):
            qwen_input(tok, "KEEP_QUESTION", "d" * 4000, max_tokens=700, forbid_truncation=True)

    def test_visible_prefix_uses_document_not_query_offsets(self):
        class Pair(dict):
            def sequence_ids(self):
                return [None, 0, None, 1, 1, None]
        def tokenizer(*args, **kwargs):
            return Pair(offset_mapping=[(0, 0), (0, 200), (0, 0), (0, 3), (4, 7), (0, 0)])
        self.assertEqual(visible_prefix(tokenizer, "question", "one two three"), "one two")


if __name__ == "__main__":
    unittest.main()
