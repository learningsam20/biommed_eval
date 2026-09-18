"""Dense retriever: sentence-transformers encode + vector-store search."""
from typing import Dict, List, Tuple


class DenseRetriever:
    """Encode queries/passages with sentence-transformers; search the vector store."""

    def __init__(self, model_name: str, vector_store, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.store = vector_store
        self.batch_size = batch_size
        self._id_to_text: Dict[str, str] = {}

    def encode(self, texts: List[str]) -> List[List[float]]:
        """L2-normalized embeddings; cosine search then equals inner product."""
        return self.model.encode(texts, batch_size=self.batch_size,
                                 show_progress_bar=False, normalize_embeddings=True).tolist()

    def index_corpus(self, ids: List[int], texts: List[str]) -> None:
        """Embed the corpus in batches and upsert into Pinecone or Qdrant."""
        self.store.create_index_if_not_exists()
        for s in range(0, len(ids), self.batch_size):
            chunk_ids = [str(i) for i in ids[s:s + self.batch_size]]
            chunk_texts = texts[s:s + self.batch_size]
            vecs = self.encode(chunk_texts)
            metas = [{"passage_id": pid, "text": t[:2000]} for pid, t in zip(chunk_ids, chunk_texts)]
            self.store.upsert(chunk_ids, vecs, metas)
            for pid, t in zip(chunk_ids, chunk_texts):
                self._id_to_text[pid] = t

    def query(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Single-query dense search: ``(passage_id, cosine_score)``."""
        return self.query_many([query], top_k)[0]

    def query_many(self, queries: List[str], top_k: int = 50) -> List[List[Tuple[int, float]]]:
        """Batch-encode all queries in one forward pass, then search each vector."""
        if not queries:
            return []
        vecs = self.encode(queries)
        return [[(int(pid), score) for pid, score in self.store.search(qv, top_k)]
                for qv in vecs]
