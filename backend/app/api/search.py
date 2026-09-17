"""Search / expand / answer endpoints."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends
from app.config import Settings, get_settings
from app.models.schemas import (
    AnswerRequest, AnswerResponse, EvidenceItem, ExpandRequest, ExpandResponse,
    GenerateRequest, GenerateResponse, SearchRequest, SearchResponse,
)

router = APIRouter()

# In-memory singletons wired at startup (lazy for testability)
_state: dict = {}


def _pubmed_url(pid: int) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"


def _retrieve(req: SearchRequest, settings: Settings) -> Tuple[List[Tuple[int, float]], List[str], str]:
    """Shared retrieval path used by /search and /answer. Returns (ranked, expansions, signal)."""
    from app.retrieval.hybrid import dedupe_by_best_score, reciprocal_rank_fusion, weighted_fusion

    bm25 = _state.get("bm25")
    dense = _state.get("dense")
    llm = _state.get("llm")

    expansions = [req.query]
    if req.use_expansion and settings.QUERY_EXPANSION_ENABLED and llm is not None:
        from app.retrieval.expansion import expand_query
        expansions = expand_query(req.query, llm, settings.EXPANSION_COUNT,
                                  settings.LLM_TEMPERATURE_EXPANSION)

    queries = expansions if req.use_expansion else [req.query]
    need_lex = req.mode in ("lexical", "hybrid") and bm25 is not None
    need_dense = req.mode in ("dense", "hybrid") and dense is not None

    def run_lex():
        out: List[Tuple[int, float]] = []
        if not need_lex:
            return out
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(queries)))) as pool:
            for hits in pool.map(lambda q: bm25.query(q, settings.LEXICAL_TOP_K), queries):
                out.extend(hits)
        return dedupe_by_best_score(out)

    def run_dense():
        if not need_dense:
            return []
        lists = dense.query_many(queries, settings.DENSE_TOP_K)
        merged: List[Tuple[int, float]] = []
        for hits in lists:
            merged.extend(hits)
        return dedupe_by_best_score(merged)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_lex = pool.submit(run_lex)
        fut_dense = pool.submit(run_dense)
        lex_all = fut_lex.result()
        dense_all = fut_dense.result()

    if req.mode == "lexical":
        ranked = sorted(lex_all, key=lambda x: x[1], reverse=True)[:req.top_k]
        signal = "bm25"
    elif req.mode == "dense":
        ranked = sorted(dense_all, key=lambda x: x[1], reverse=True)[:req.top_k]
        signal = "dense"
    else:
        if req.fusion == "weighted":
            ranked = weighted_fusion(lex_all, dense_all, settings.WEIGHT_BM25,
                                     settings.WEIGHT_DENSE, req.top_k)
        else:
            ranked = reciprocal_rank_fusion([lex_all, dense_all], settings.RRF_K, req.top_k)
        signal = f"hybrid-{req.fusion}"
    return ranked, expansions, signal


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, settings: Settings = Depends(get_settings)):
    t0 = time.perf_counter()
    ranked, expansions, signal = _retrieve(req, settings)
    store = _state.get("store")
    id_to_text: dict = store.fetch_texts([str(pid) for pid, _ in ranked]) if store else {}
    results = [EvidenceItem(passage_id=pid, text=str(id_to_text.get(str(pid), ""))[:2000],
                            score=float(s), signal=signal, source_url=_pubmed_url(pid))
               for pid, s in ranked]
    logger.info(
        "search recorded query=%r mode=%s fusion=%s expansion=%s n=%d ids_scores=%s",
        req.query, req.mode, req.fusion, req.use_expansion, len(ranked),
        [{"passage_id": pid, "score": float(s)} for pid, s in ranked],
    )
    return SearchResponse(original_query=req.query, expansions=expansions[1:],
                          results=results,
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          config={"mode": req.mode, "fusion": req.fusion, "top_k": req.top_k,
                                  "use_expansion": req.use_expansion})


@router.post("/expand", response_model=ExpandResponse)
async def expand(req: ExpandRequest, settings: Settings = Depends(get_settings)):
    llm = _state.get("llm")
    if llm is None:
        return ExpandResponse(original_query=req.query, expansions=[])
    from app.retrieval.expansion import expand_query
    ex = expand_query(req.query, llm, req.count, settings.LLM_TEMPERATURE_EXPANSION)
    return ExpandResponse(original_query=req.query, expansions=ex[1:])


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, settings: Settings = Depends(get_settings)):
    """Answer from already-retrieved evidence (avoids re-running search)."""
    t0 = time.perf_counter()
    llm = _state.get("llm")
    evidence = [(e.passage_id, e.text) for e in req.evidence]
    if llm is None or not evidence:
        return GenerateResponse(answer="INSUFFICIENT_EVIDENCE", citations=[],
                                insufficient_evidence=True,
                                latency_ms=(time.perf_counter() - t0) * 1000)
    from app.generation.answer import generate_grounded_answer
    text, cites, insuf = generate_grounded_answer(req.query, evidence, llm,
                                                  settings.LLM_TEMPERATURE_ANSWER)
    return GenerateResponse(answer=text, citations=cites, insufficient_evidence=insuf,
                            latency_ms=(time.perf_counter() - t0) * 1000)


@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest, settings: Settings = Depends(get_settings)):
    t0 = time.perf_counter()
    sr = await search(SearchRequest(query=req.query, mode=req.mode, fusion=req.fusion,
                                    top_k=req.top_k, use_expansion=req.use_expansion), settings)
    llm = _state.get("llm")
    evidence = [(r.passage_id, r.text) for r in sr.results]
    if llm is None or not evidence:
        return AnswerResponse(answer="INSUFFICIENT_EVIDENCE", citations=[],
                              insufficient_evidence=True, evidence=sr.results,
                              expansions=sr.expansions,
                              latency_ms=(time.perf_counter() - t0) * 1000)
    from app.generation.answer import generate_grounded_answer
    text, cites, insuf = generate_grounded_answer(req.query, evidence, llm,
                                                  settings.LLM_TEMPERATURE_ANSWER)
    return AnswerResponse(answer=text, citations=cites, insufficient_evidence=insuf,
                          evidence=sr.results, expansions=sr.expansions,
                          latency_ms=(time.perf_counter() - t0) * 1000)
