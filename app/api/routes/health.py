"""Health check route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple liveness endpoint."""
    return {"status": "ok"}
