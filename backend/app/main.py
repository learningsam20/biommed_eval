"""FastAPI entry point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import cors_origins, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    logger.info("starting biomed-hybrid-search provider=%s vectordb=%s embed=%s",
                settings.LLM_PROVIDER, settings.VECTOR_DB, settings.EMBEDDING_MODEL)
    try:
        from app.models.vector_store import get_vector_store
        from app.generation.llm import get_llm_client
        from app.api.search import _state
        _state["llm"] = get_llm_client(settings)
        _state["store"] = get_vector_store(settings)
        # BM25 + dense retrievers are loaded lazily if index files exist;
        # scripts/build_index.py creates them. Missing files -> search returns [].
        from app.config import resolve_path
        from app.retrieval.bm25 import BM25LexicalIndex
        if resolve_path(settings.BM25_INDEX_PATH).exists():
            _state["bm25"] = BM25LexicalIndex.load(str(resolve_path(settings.BM25_INDEX_PATH)))
            logger.info("bm25 loaded")
        try:
            from app.retrieval.dense import DenseRetriever
            _state["dense"] = DenseRetriever(settings.EMBEDDING_MODEL,
                                             _state["store"], settings.EMBEDDING_BATCH_SIZE)
            logger.info("dense retriever ready")
        except Exception as e:
            logger.warning("dense unavailable: %s", e)
        logger.info("lifespan ready")
    except Exception as e:
        logger.warning("lifespan partial init: %s", e)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="BioMed Hybrid Search API", version="1.0.0", lifespan=lifespan)
    origins = cors_origins(settings.FRONTEND_URL)
    logger.info("cors allow_origins=%s", origins)
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    from app.api import health, search
    from app.config import get_settings as gs

    @app.get("/api/config")
    async def public_config(s=__import__("fastapi").Depends(gs)):
        return {"llm_provider": s.LLM_PROVIDER, "embedding_model": s.EMBEDDING_MODEL,
                "vector_db": s.VECTOR_DB, "fusion_method": s.FUSION_METHOD,
                "lexical_top_k": s.LEXICAL_TOP_K, "dense_top_k": s.DENSE_TOP_K,
                "hybrid_top_k": s.HYBRID_TOP_K}

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(search.router, prefix="/api", tags=["search"])
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.BACKEND_HOST, port=s.BACKEND_PORT, reload=True)
