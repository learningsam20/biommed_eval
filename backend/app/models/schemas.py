"""Pydantic request/response schemas."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    passage_id: int
    text: str
    score: float
    signal: str = "hybrid"
    source_url: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "weighted"
    top_k: int = 20
    use_expansion: bool = False  # off by default — expansion adds ~1 LLM round-trip


class SearchResponse(BaseModel):
    original_query: str
    expansions: List[str] = Field(default_factory=list)
    results: List[EvidenceItem]
    latency_ms: float
    config: dict


class ExpandRequest(BaseModel):
    query: str
    count: int = 3


class ExpandResponse(BaseModel):
    original_query: str
    expansions: List[str]


class AnswerRequest(BaseModel):
    query: str
    mode: Literal["lexical", "dense", "hybrid"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "weighted"
    top_k: int = 20
    use_expansion: bool = False


class AnswerResponse(BaseModel):
    answer: str
    citations: List[int]
    insufficient_evidence: bool
    evidence: List[EvidenceItem]
    expansions: List[str] = Field(default_factory=list)
    latency_ms: float


class GenerateRequest(BaseModel):
    query: str
    evidence: List[EvidenceItem]


class GenerateResponse(BaseModel):
    answer: str
    citations: List[int]
    insufficient_evidence: bool
    latency_ms: float


class EvalItem(BaseModel):
    query_id: int
    question: str
    answer: str
    relevant_passage_ids: List[int]
    qtype: str = "unknown"
