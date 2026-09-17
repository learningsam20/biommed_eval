"""LLM client factory: OpenRouter (OpenAI-compatible) + Ollama, with usage/cost tracking."""
from dataclasses import dataclass, field
from typing import Optional, Protocol


# Ballpark USD per 1M tokens when provider does not return a billable amount.
# Ollama is local → $0. OpenRouter free-tier / unknown models use a conservative mid-tier rate.
_DEFAULT_PROMPT_PER_1M = 0.50
_DEFAULT_COMPLETION_PER_1M = 1.50


@dataclass
class UsageAccumulator:
    """Process-wide LLM usage counter (shared by the client instance)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float = 0.0) -> None:
        self.prompt_tokens += max(0, prompt)
        self.completion_tokens += max(0, completion)
        self.calls += 1
        self.cost_usd += max(0.0, cost)

    def snapshot(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "llm_calls": self.calls,
            "cost_usd": round(self.cost_usd, 6),
        }

    def reset(self) -> None:
        self.prompt_tokens = self.completion_tokens = self.calls = 0
        self.cost_usd = 0.0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when the provider omits usage (~4 chars/token)."""
    return max(1, len(text) // 4)


class LLMClient(Protocol):
    usage: UsageAccumulator

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str: ...


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str,
                 usage: Optional[UsageAccumulator] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             max_retries=5)  # free-tier 429 backoff
        self.model = model
        self.usage = usage or UsageAccumulator()

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        if u is not None:
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
        else:
            pt = _estimate_tokens(system + user)
            ct = _estimate_tokens(text)
        cost = (pt * _DEFAULT_PROMPT_PER_1M + ct * _DEFAULT_COMPLETION_PER_1M) / 1_000_000
        self.usage.add(pt, ct, cost)
        return text


class OllamaClient:
    def __init__(self, base_url: str, model: str,
                 usage: Optional[UsageAccumulator] = None):
        import ollama
        self.client = ollama.Client(host=base_url)
        self.model = model
        self.usage = usage or UsageAccumulator()

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        resp = self.client.chat(model=self.model,
                                messages=[{"role": "system", "content": system},
                                          {"role": "user", "content": user}],
                                options={"temperature": temperature})
        text = resp["message"]["content"]
        # Local inference: track tokens for accounting, cost stays $0
        pt = int(resp.get("prompt_eval_count") or _estimate_tokens(system + user))
        ct = int(resp.get("eval_count") or _estimate_tokens(text))
        self.usage.add(pt, ct, 0.0)
        return text


def get_llm_client(settings) -> LLMClient:
    usage = UsageAccumulator()
    if settings.LLM_PROVIDER.lower() == "ollama":
        return OllamaClient(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL, usage)
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty in .env")
    return OpenRouterClient(settings.OPENROUTER_API_KEY,
                            settings.OPENROUTER_MODEL, settings.OPENROUTER_BASE_URL, usage)
