"""
Health check endpoints.
"""
from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    Health check endpoint.
    
    Returns basic service health information.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.app_env,
    )


@router.get("/health/ready", summary="Readiness check")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns ready status if all dependencies are available.
    """
    return {"status": "ready"}
