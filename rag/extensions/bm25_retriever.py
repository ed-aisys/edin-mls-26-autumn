"""BM25 over course corpus record texts (lexical baseline for retrieval experiments)."""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25Retriever:
    """Okapi BM25 with an inverted index (no external deps)."""

    def __init__(self, record_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        if len(record_ids) != len(texts):
            raise ValueError("record_ids and texts length mismatch")
        self.record_ids = list(record_ids)
        self.k1 = k1
        self.b = b
        self.N = len(record_ids)
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: dict[str, int] = defaultdict(int)

        for doc_idx, text in enumerate(texts):
            tokens = TOKEN_RE.findall(str(text).lower())
            dl = len(tokens) or 1
            self.doc_lens.append(dl)
            tf: dict[str, int] = defaultdict(int)
            for term in tokens:
                tf[term] += 1
            for term, freq in tf.items():
                self.postings[term].append((doc_idx, freq))
                df[term] += 1

        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1.0
        self.idf = {
            term: math.log((self.N - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def retrieve(self, query: str, k: int = 100) -> dict:
        if k < 1:
            raise ValueError("k must be positive")
        k = min(k, self.N)
        start = time.perf_counter()
        qterms = TOKEN_RE.findall(str(query).lower())
        if not qterms:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {"record_ids": [], "scores": [], "latency_ms": elapsed_ms}

        scores: dict[int, float] = defaultdict(float)
        for term in set(qterms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc_idx, freq in posting:
                dl = self.doc_lens[doc_idx]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * freq * (self.k1 + 1) / denom

        if not scores:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {"record_ids": [], "scores": [], "latency_ms": elapsed_ms}

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "record_ids": [self.record_ids[i] for i, _ in ranked],
            "scores": [float(s) for _, s in ranked],
            "latency_ms": elapsed_ms,
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k_out: int = 100,
    c: float = 60.0,
) -> list[str]:
    """RRF merge of ranked record_id lists (Cormack et al., c default 60)."""
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, rid in enumerate(lst, start=1):
            scores[rid] += 1.0 / (c + rank)
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [rid for rid, _ in ordered[:k_out]]
