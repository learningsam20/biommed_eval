"""Vector-store abstraction: Pinecone (primary, serverless) + Qdrant (fallback)."""
from typing import List, Protocol, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential


class VectorStore(Protocol):
    def create_index_if_not_exists(self) -> None: ...
    def upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[dict]) -> None: ...
    def search(self, vector: List[float], top_k: int) -> List[Tuple[str, float]]: ...
    def fetch_texts(self, ids: List[str]) -> dict: ...
    def health_check(self) -> bool: ...


class PineconeVectorStore:
    """Serverless Pinecone store with auto-create + dim validation."""

    def __init__(self, api_key: str, index_name: str, cloud: str = "aws",
                 region: str = "us-east-1", metric: str = "cosine",
                 dimension: int = 768, namespace: str = "", batch_size: int = 100):
        from pinecone import Pinecone, ServerlessSpec
        self._ServerlessSpec = ServerlessSpec
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.metric = metric
        self.dimension = dimension
        self.namespace = namespace or ""
        self.batch_size = batch_size
        self._index = None

    def create_index_if_not_exists(self) -> None:
        """Create the serverless index, or fail if an existing one has the wrong dim."""
        existing = [i.name for i in self.client.list_indexes()]
        if self.index_name not in existing:
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=self._ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        else:
            desc = self.client.describe_index(self.index_name)
            if desc.dimension != self.dimension:
                raise ValueError(
                    f"Pinecone index '{self.index_name}' dim={desc.dimension} "
                    f"!= EMBEDDING_DIM={self.dimension}. Re-create index or fix .env."
                )
        self._index = self.client.Index(self.index_name)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=16))
    def upsert(self, ids, vectors, metadatas) -> None:
        if self._index is None:
            self.create_index_if_not_exists()
        for s in range(0, len(ids), self.batch_size):
            batch = [
                {"id": str(i), "values": v, "metadata": m}
                for i, v, m in zip(ids[s:s + self.batch_size],
                                   vectors[s:s + self.batch_size],
                                   metadatas[s:s + self.batch_size])
            ]
            self._index.upsert(vectors=batch, namespace=self.namespace or None)

    def search(self, vector, top_k) -> List[Tuple[str, float]]:
        """Nearest neighbors: ``(vector_id, cosine_score)``."""
        if self._index is None:
            self.create_index_if_not_exists()
        res = self._index.query(vector=vector, top_k=top_k,
                                include_metadata=True, namespace=self.namespace or None)
        return [(m.id, float(m.score)) for m in res.matches]

    def fetch_texts(self, ids: List[str]) -> dict:
        """Map passage ID → stored text snippet for the evidence panel."""
        if self._index is None:
            self.create_index_if_not_exists()
        try:
            res = self._index.fetch(ids=[str(i) for i in ids],
                                    namespace=self.namespace or None)
            return {v.id: (v.metadata or {}).get("text", "") for v in res.vectors.values()}
        except Exception:
            return {}

    def health_check(self) -> bool:
        try:
            self.client.list_indexes()
            return True
        except Exception:
            return False


class QdrantVectorStore:
    """Local fallback store."""

    def __init__(self, url: str, collection: str, dimension: int = 768, api_key=None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._Distance = Distance
        self._VectorParams = VectorParams
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection = collection
        self.dimension = dimension

    def create_index_if_not_exists(self) -> None:
        """Create a cosine collection if it is missing."""
        from qdrant_client.models import Distance, VectorParams
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, metadatas) -> None:
        from qdrant_client.models import PointStruct
        self.create_index_if_not_exists()
        points = [PointStruct(id=abs(hash(i)) % (2**63), vector=v, payload={"pid": i, **m})
                  for i, v, m in zip(ids, vectors, metadatas)]
        for s in range(0, len(points), 100):
            self.client.upsert(collection_name=self.collection, points=points[s:s + 100])

    def search(self, vector, top_k):
        hits = self.client.search(collection_name=self.collection,
                                  query_vector=vector, limit=top_k)
        return [(str(h.payload.get("pid", h.id)), float(h.score)) for h in hits]

    def fetch_texts(self, ids: List[str]) -> dict:
        """Scroll by payload ``pid`` and return id → text."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=[FieldCondition(key="pid",
                                      match=MatchAny(any=[str(i) for i in ids]))]),
                limit=len(ids), with_vectors=False)
            return {str(r.payload.get("pid")): r.payload.get("text", "") for r in records}
        except Exception:
            return {}

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


def get_vector_store(settings) -> VectorStore:
    """Pinecone or Qdrant from ``VECTOR_DB``. Same interface for index, search, fetch."""
    if settings.VECTOR_DB.lower() == "pinecone":
        if not settings.PINECONE_API_KEY:
            raise RuntimeError("VECTOR_DB=pinecone but PINECONE_API_KEY is empty in .env")
        return PineconeVectorStore(
            api_key=settings.PINECONE_API_KEY, index_name=settings.PINECONE_INDEX_NAME,
            cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION,
            metric=settings.PINECONE_METRIC, dimension=settings.EMBEDDING_DIM,
            namespace=settings.PINECONE_NAMESPACE, batch_size=settings.PINECONE_BATCH_SIZE,
        )
    return QdrantVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION,
                             dimension=settings.EMBEDDING_DIM, api_key=settings.QDRANT_API_KEY)
