"""Independently labeled passage retrieval with deterministic tie-breaking."""

import math
import re
from collections import Counter

import numpy as np


def scores(ranked, labels, k):
    if k < 1:
        raise ValueError("k must be positive")
    relevant = {key: value for key, value in (labels or {}).items() if value > 0}
    if not relevant:
        return dict.fromkeys(("precision", "recall", "hit", "rr", "ndcg"))
    seen = set()
    gains = []
    for key in ranked[:k]:
        gains.append(0 if key in seen else relevant.get(key, 0))
        seen.add(key)
    matched = sum(x > 0 for x in gains)

    def dcg(values):
        return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(values))

    return {
        "precision": matched / k,
        "recall": matched / len(relevant),
        "hit": int(matched > 0),
        "rr": next((1 / (i + 1) for i, v in enumerate(gains) if v), 0),
        "ndcg": dcg(gains) / dcg(sorted(relevant.values(), reverse=True)[:k]),
    }


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, documents, method, embedding=None, reranker=None):
        self.keys = sorted(documents)
        self.texts = [documents[k]["text"] for k in self.keys]
        self.terms = [Counter(tokens(t)) for t in self.texts]
        self.method = method
        self.lengths = [sum(t.values()) for t in self.terms]
        self.avg = np.mean(self.lengths)
        self.df = Counter(word for t in self.terms for word in t)
        if method != "bm25":
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(2)
            self.encoder = SentenceTransformer(
                embedding, local_files_only=True, device="cpu"
            )
            self.matrix = self.encoder.encode(
                self.texts, normalize_embeddings=True, show_progress_bar=False
            )
        if method == "rerank":
            from sentence_transformers import CrossEncoder

            self.cross = CrossEncoder(reranker, local_files_only=True, device="cpu")

    def search(self, query, k):
        n = len(self.keys)
        bm = np.zeros(n)
        for word in tokens(query):
            idf = math.log(1 + (n - self.df[word] + 0.5) / (self.df[word] + 0.5))
            for i, t in enumerate(self.terms):
                f = t[word]
                bm[i] += (
                    idf
                    * f
                    * 2.2
                    / (f + 1.2 * (0.25 + 0.75 * self.lengths[i] / self.avg))
                )

        def order(values):
            return np.argsort(-values, kind="stable")

        values = bm
        if self.method != "bm25":
            vector = self.encoder.encode(
                query, normalize_embeddings=True, show_progress_bar=False
            )
            semantic = self.matrix @ vector
            if self.method == "semantic":
                values = semantic
            else:
                values = np.zeros(n)
                for ranks in (order(bm), order(semantic)):
                    for i, index in enumerate(ranks):
                        values[index] += 1 / (61 + i)
        ranked = order(values)
        if self.method == "rerank":
            ids = ranked[: min(n, max(k, 5))]
            cross = self.cross.predict(
                [(query, self.texts[i]) for i in ids], show_progress_bar=False
            )
            ranked = ids[order(cross)]
            values = dict(zip(ids, cross))
        return [
            {"chunk_id": self.keys[i], "rank": rank, "score": float(values[i])}
            for rank, i in enumerate(ranked[:k], 1)
        ]
