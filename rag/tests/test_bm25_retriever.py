import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.bm25_retriever import BM25Retriever, reciprocal_rank_fusion


class BM25Tests(unittest.TestCase):
    def test_retrieve_orders_by_relevance(self):
        ids = ["a", "b", "c"]
        texts = [
            "tax deduction capital gains broker fee",
            "unrelated fantasy fiction novel",
            "sales tax retail store purchase",
        ]
        bm25 = BM25Retriever(ids, texts)
        out = bm25.retrieve("sales tax retail", k=2)
        self.assertEqual(out["record_ids"][0], "c")

    def test_rrf_merges_lists(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k_out=3)
        self.assertEqual(len(fused), 3)
        self.assertIn("c", fused[:2])


if __name__ == "__main__":
    unittest.main()
