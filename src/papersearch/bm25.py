import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its not of on or "
    "that the their these this to was were which with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(texts)

        postings: dict[str, tuple[list[int], list[int]]] = {}
        doc_len = np.zeros(self.n_docs, dtype=np.float64)
        for doc, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len[doc] = len(tokens)
            for term, tf in Counter(tokens).items():
                docs, tfs = postings.setdefault(term, ([], []))
                docs.append(doc)
                tfs.append(tf)

        self.postings = {
            term: (np.array(docs, dtype=np.int64), np.array(tfs, dtype=np.float64))
            for term, (docs, tfs) in postings.items()
        }
        avg_len = doc_len.mean() if self.n_docs else 1.0
        self.len_norm = k1 * (1.0 - b + b * doc_len / avg_len)

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term in sorted(set(tokenize(query))):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            out[docs] += self.idf(term) * tf * (self.k1 + 1.0) / (tf + self.len_norm[docs])
        return out

    def search(self, query: str, k: int):
        s = self.scores(query)
        k = min(k, self.n_docs)
        order = np.lexsort((np.arange(self.n_docs), -s))[:k]
        return order, s[order]
