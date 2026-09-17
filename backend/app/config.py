"""Central settings — .env is the single source of truth."""
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")  # export HF_TOKEN etc. for huggingface_hub/datasets
from typing import List, Optional
from urllib.parse import urlparse, urlunparse
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LLM_PROVIDER: str = "openrouter"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    LLM_TEMPERATURE_EXPANSION: float = 0.3
    LLM_TEMPERATURE_ANSWER: float = 0.1

    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = 64

    VECTOR_DB: str = "pinecone"

    PINECONE_API_KEY: Optional[str] = None
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "bioasq-passages"
    PINECONE_METRIC: str = "cosine"
    PINECONE_NAMESPACE: str = "bioasq"
    PINECONE_BATCH_SIZE: int = 100

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "bioasq_passages"

    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    LEXICAL_TOP_K: int = 50
    DENSE_TOP_K: int = 50
    HYBRID_TOP_K: int = 20
    FUSION_METHOD: str = "weighted"
    RRF_K: int = 60
    WEIGHT_BM25: float = 0.5
    WEIGHT_DENSE: float = 0.5

    QUERY_EXPANSION_ENABLED: bool = True
    EXPANSION_COUNT: int = 3

    EVAL_QUERY_SET: str = "data/eval_queries_100.json"
    EVAL_SEED: int = 42
    EVAL_OUTPUT_DIR: str = "results"
    JUDGE_MODEL: str = "same_as_llm"
    JUDGE_TEMPERATURE: float = 0.0

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 5268
    FRONTEND_URL: str = "http://localhost:5269"
    LOG_LEVEL: str = "INFO"

    HF_TOKEN: Optional[str] = None
    HF_DATASET_ID: str = "rag-datasets/rag-mini-bioasq"
    DATA_DIR: str = "data"

    class Config:
        env_file = str(REPO_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def cors_origins(frontend_url: str) -> List[str]:
    """Allow both localhost and 127.0.0.1 — browsers treat them as distinct origins."""
    seen: List[str] = []
    for raw in frontend_url.split(","):
        url = raw.strip().rstrip("/")
        if not url:
            continue
        for candidate in (url, _swap_loopback(url)):
            if candidate and candidate not in seen:
                seen.append(candidate)
    return seen


def _swap_loopback(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host == "localhost":
        alt = "127.0.0.1"
    elif host == "127.0.0.1":
        alt = "localhost"
    else:
        return ""
    netloc = f"{alt}:{parsed.port}" if parsed.port else alt
    if parsed.username:
        user = parsed.username
        if parsed.password:
            user = f"{user}:{parsed.password}"
        netloc = f"{user}@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc)).rstrip("/")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
