# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""BM25 scorer for keyword search over a fixed corpus."""

import math
import re


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, treating underscores as word separators."""
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


class BM25Scorer:
    """BM25 ranking over a fixed list of documents.

    Pre-computes IDF and average document length at construction time
    so that repeated queries are fast.

    Args:
        docs: List of raw text documents to index.
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._tokenized: list[list[str]] = [tokenize(doc) for doc in docs]
        n = len(self._tokenized)
        self._avgdl = sum(len(d) for d in self._tokenized) / n if n else 1.0

        # Document frequency per term
        df: dict[str, int] = {}
        for doc in self._tokenized:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # IDF with Robertson-Sparck Jones smoothing
        self._idf: dict[str, float] = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """Return a BM25 score for each document in the corpus."""
        terms = tokenize(query)
        result = []
        for doc_tokens in self._tokenized:
            dl = len(doc_tokens)
            score = 0.0
            for term in terms:
                idf = self._idf.get(term, 0.0)
                if idf == 0.0:
                    continue
                tf = doc_tokens.count(term)
                score += (
                    idf
                    * (tf * (self.k1 + 1))
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                )
            result.append(score)
        return result

    def top_k(self, query: str, k: int = 10) -> list[int]:
        """Return indices of the top-k scoring documents (descending order)."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:k]]
