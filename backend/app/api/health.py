# Health Check Endpoint
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Liveness probe for the API (does not check vector DB or LLM)."""
    return {"status": "healthy", "service": "biomedical-hybrid-search"}
