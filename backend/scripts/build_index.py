"""Build BM25 pickle + dense vector index. Usage: python -m scripts.build_index [--limit N]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_corpus(dataset_id: str, limit: int = 0):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, "text-corpus")["passages"]
    ids, texts = [], []
    for row in ds:
        ids.append(int(row["id"]))
        texts.append(row["passage"])
        if limit and len(ids) >= limit:
            break
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-dense", action="store_true", help="build BM25 only")
    args = ap.parse_args()

    from app.config import get_settings
    from app.retrieval.bm25 import BM25LexicalIndex
    from app.retrieval.dense import DenseRetriever
    from app.models.vector_store import get_vector_store
    s = get_settings()
    print(f"loading corpus {s.HF_DATASET_ID} ...")
    ids, texts = load_corpus(s.HF_DATASET_ID, args.limit)
    print(f"passages: {len(ids)}")

    from app.config import resolve_path
    bm25 = BM25LexicalIndex(k1=s.BM25_K1, b=s.BM25_B)
    bm25.build(ids, texts)
    bm25.save(str(resolve_path(s.BM25_INDEX_PATH)))
    print(f"BM25 saved -> {resolve_path(s.BM25_INDEX_PATH)}")

    from app.config import resolve_path as _rp
    _rp(s.DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(_rp(s.DATA_DIR) / "passages.jsonl", "w") as f:
        for pid, tx in zip(ids, texts):
            f.write(json.dumps({"id": pid, "passage": tx}) + "\n")
    print("cached passages.jsonl")

    if args.skip_dense:
        print("skipping dense index (--skip-dense)")
        return
    store = get_vector_store(s)
    store.create_index_if_not_exists()
    print(f"vector index ready ({s.VECTOR_DB}/{getattr(s, 'PINECONE_INDEX_NAME', getattr(s, 'QDRANT_COLLECTION', ''))})")
    retriever = DenseRetriever(s.EMBEDDING_MODEL, store, s.EMBEDDING_BATCH_SIZE)
    retriever.index_corpus(ids, texts)
    print("dense upsert complete")

    from app.config import resolve_path as _rp
    _rp(s.DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(_rp(s.DATA_DIR) / "passages.jsonl", "w") as f:
        for pid, t in zip(ids, texts):
            f.write(json.dumps({"id": pid, "passage": t}) + "\n")
    print("cached passages.jsonl")


if __name__ == "__main__":
    main()
