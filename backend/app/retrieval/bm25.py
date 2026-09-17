"""BM25 lexical index over the passage corpus."""
import pickle
from pathlib import Path
from typing import List, Tuple
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25LexicalIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._ids: List[int] = []
        self._texts: List[str] = []

    def build(self, ids: List[int], texts: List[str]) -> None:
        self._ids = ids
        self._texts = texts
        self._bm25 = BM25Okapi([tokenize(t) for t in texts], k1=self.k1, b=self.b)

    def query(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        assert self._bm25 is not None, "BM25 index not built — run scripts/build_index.py"
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in ranked]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"ids": self._ids, "texts": self._texts,
                         "k1": self.k1, "b": self.b}, f)

    @classmethod
    def load(cls, path: str) -> "BM25LexicalIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.build(data["ids"], data["texts"])
        return idx
