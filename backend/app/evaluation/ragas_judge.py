"""RAGAS 0.4 metrics via llm_factory + Ollama/OpenRouter OpenAI-compatible APIs."""
import asyncio
import math
from typing import Any, Dict, List

_LOOP: asyncio.AbstractEventLoop | None = None
_LLM = None
_LLM_KEY = None


def _loop() -> asyncio.AbstractEventLoop:
    """Reuse one loop so httpx is not closed after asyncio.run() tears it down."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_LOOP)
    return _LOOP


def _finite(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) or math.isinf(v) else v


def _score_value(result: Any) -> float:
    if result is None:
        return 0.0
    if hasattr(result, "value"):
        return _finite(result.value)
    return _finite(result)


def _ragas_llm(temperature: float = 0.0):
    """Cached InstructorLLM / llm_factory client. Reused so httpx keeps one event loop."""
    global _LLM, _LLM_KEY
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.llms.base import InstructorLLM, InstructorModelArgs

    from app.config import get_settings
    from app.evaluation.judge import JUDGE_MAX_TOKENS

    s = get_settings()
    key = (s.LLM_PROVIDER, s.OLLAMA_MODEL, s.OPENROUTER_MODEL, temperature)
    if _LLM is not None and _LLM_KEY == key:
        return _LLM
    if s.LLM_PROVIDER.lower() == "ollama":
        # Ollama OpenAI-compat + instructor JSON mode; JSON is more reliable on
        # small local models than MD_JSON (which can echo the schema).
        client = instructor.from_openai(
            AsyncOpenAI(
                api_key="ollama",
                base_url=s.OLLAMA_BASE_URL.rstrip("/") + "/v1",
                timeout=20.0,
            ),
            mode=instructor.Mode.JSON,
        )
        _LLM = InstructorLLM(
            client=client,
            model=s.OLLAMA_MODEL,
            provider="openai",
            model_args=InstructorModelArgs(
                temperature=max(temperature, 0.0),
                max_tokens=JUDGE_MAX_TOKENS,
                top_p=0.9,
            ),
            extra_body={"think": False, "options": {"think": False}},
        )
        _LLM_KEY = key
        return _LLM
    client = AsyncOpenAI(
        api_key=s.OPENROUTER_API_KEY or "missing",
        base_url=s.OPENROUTER_BASE_URL.rstrip("/"),
    )
    _LLM = llm_factory(
        s.OPENROUTER_MODEL,
        client=client,
        temperature=max(temperature, 0.0),
        max_tokens=JUDGE_MAX_TOKENS,
    )
    _LLM_KEY = key
    return _LLM


async def _ascore(metric, **kwargs) -> float:
    result = await metric.ascore(**kwargs)
    return _score_value(result)


async def _score_named(name: str, metric, **kwargs) -> float:
    try:
        return await _ascore(metric, **kwargs)
    except Exception as e:
        print(f"  ragas {name} error: {type(e).__name__}: {e}")
        return 0.0


def score_ragas_one(question: str, reference: str, answer: str,
                    contexts: List[str], llm_client, temperature: float = 0.0) -> Dict[str, float]:
    """RAGAS triad mapped to our keys: correctness / groundedness / context_relevance.

    FactualCorrectness uses ``mode=precision`` (answer claims supported by gold).
    Faithfulness and ContextRelevance use retrieved passages, not the gold IDs.
    """
    from ragas.metrics.collections import ContextRelevance, FactualCorrectness, Faithfulness

    del llm_client  # RAGAS talks to the provider directly
    llm = _ragas_llm(temperature=temperature)
    ctx = contexts or [""]
    response = answer or ""
    ref = reference or ""

    async def _run():
        corr = await _score_named(
            "factual_correctness",
            # Precision: fraction of answer claims supported by gold — matches
            # custom/DeepEval "correctness". Default F1 is 0 when a short RAG
            # answer misses extra gold claims.
            FactualCorrectness(llm=llm, mode="precision"),
            response=response,
            reference=ref,
        )
        faith = await _score_named(
            "faithfulness",
            Faithfulness(llm=llm),
            user_input=question,
            response=response,
            retrieved_contexts=ctx,
        )
        crel = await _score_named(
            "context_relevance",
            ContextRelevance(llm=llm),
            user_input=question,
            retrieved_contexts=ctx,
        )
        return corr, faith, crel

    corr, faith, crel = _loop().run_until_complete(_run())
    return {
        "correctness": _finite(corr),
        "groundedness": _finite(faith),
        "context_relevance": _finite(crel),
    }
