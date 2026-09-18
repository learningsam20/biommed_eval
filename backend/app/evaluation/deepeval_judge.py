"""DeepEval RAG triad via OllamaModel (or OpenRouter-compatible wrapper)."""
import json
from typing import Dict, List

from deepeval.models import DeepEvalBaseLLM


class AppDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, client, temperature: float = 0.0, name: str = "app-llm"):
        self._client = client
        self._temperature = temperature
        super().__init__(name)

    def load_model(self, *args, **kwargs):
        return self._client

    def get_model_name(self, *args, **kwargs) -> str:
        return getattr(self, "name", None) or "app-llm"

    def generate(self, prompt: str, *args, **kwargs):
        schema = kwargs.get("schema")
        extra = ""
        if schema is not None:
            try:
                extra = "\nReturn ONLY JSON matching this schema:\n" + json.dumps(
                    schema.model_json_schema()
                )
            except Exception:
                extra = "\nReturn ONLY valid JSON."
        raw = self._client.chat(
            "You are a strict evaluator. Follow the instructions exactly.",
            prompt + extra,
            self._temperature,
        ) or ""
        if schema is None:
            return raw
        blob = raw[raw.find("{"): raw.rfind("}") + 1]
        return schema.model_validate(json.loads(blob))

    async def a_generate(self, prompt: str, *args, **kwargs):
        return self.generate(prompt, *args, **kwargs)


def _safe_score(metric) -> float:
    val = getattr(metric, "score", None)
    try:
        v = float(val if val is not None else 0.0)
        if v != v:  # NaN
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _model(llm_client, temperature: float):
    """OllamaModel with think off, or a thin wrapper around the app OpenRouter client."""
    import os

    from app.config import get_settings
    from app.evaluation.judge import JUDGE_MAX_TOKENS
    os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "1"
    s = get_settings()
    if s.LLM_PROVIDER.lower() == "ollama":
        from deepeval.models.llms.ollama_model import OllamaModel, retry_ollama
        from pydantic import BaseModel
        from typing import Optional

        class JudgeOllamaModel(OllamaModel):
            @retry_ollama
            def generate(self, prompt: str, schema: Optional[BaseModel] = None):
                chat_model = self.load_model()
                kwargs = {
                    "model": self.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": schema.model_json_schema() if schema else None,
                    "options": {
                        "temperature": self.temperature,
                        **(self.generation_kwargs or {}),
                    },
                }
                try:
                    response = chat_model.chat(think=False, **kwargs)
                except TypeError:
                    response = chat_model.chat(**kwargs)
                content = response.message.content
                return (
                    schema.model_validate_json(content) if schema else content,
                    0,
                )

            @retry_ollama
            async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
                return self.generate(prompt, schema)

        return JudgeOllamaModel(
            model=s.OLLAMA_MODEL,
            base_url=s.OLLAMA_BASE_URL,
            temperature=max(temperature, 0.0),
            timeout=20.0,
            generation_kwargs={
                "num_predict": JUDGE_MAX_TOKENS,
                "num_ctx": getattr(s, "OLLAMA_NUM_CTX", 16384),
            },
        )
    return AppDeepEvalLLM(llm_client, temperature=temperature, name=s.OPENROUTER_MODEL)


def _measure(metric, case) -> None:
    try:
        metric.measure(case, _show_indicator=False)
    except TypeError:
        metric.measure(case)


def score_deepeval_one(question: str, reference: str, answer: str,
                       contexts: List[str], llm_client, temperature: float = 0.0) -> Dict[str, float]:
    """DeepEval RAG triad: AnswerRelevancy, Faithfulness, ContextualRelevancy → our keys."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    model = _model(llm_client, temperature)
    case = LLMTestCase(
        input=question,
        actual_output=answer or "",
        expected_output=reference or "",
        retrieval_context=contexts or [""],
    )
    relevancy = AnswerRelevancyMetric(model=model, include_reason=False, async_mode=False)
    faith = FaithfulnessMetric(
        model=model,
        include_reason=False,
        async_mode=False,
        truths_extraction_limit=8,
    )
    crel = ContextualRelevancyMetric(model=model, include_reason=False, async_mode=False)
    errors = []
    for metric in (relevancy, faith, crel):
        try:
            _measure(metric, case)
        except Exception as e:
            errors.append(f"{type(metric).__name__}: {type(e).__name__}: {e}")
    if errors:
        print("  deepeval metric errors: " + " | ".join(errors[:3]))
    return {
        "correctness": _safe_score(relevancy),
        "groundedness": _safe_score(faith),
        "context_relevance": _safe_score(crel),
    }
