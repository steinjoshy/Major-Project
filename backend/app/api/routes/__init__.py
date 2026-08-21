"""
API router initialization.
"""
from fastapi import APIRouter

from backend.app.api.routes import analytics, data, forecast, health, inventory, models

api_router = APIRouter()

# Include all route modules
api_router.include_router(health.router)
api_router.include_router(data.router, prefix="/data")
api_router.include_router(forecast.router, prefix="/forecast")
api_router.include_router(inventory.router, prefix="/inventory")
api_router.include_router(models.router, prefix="/models")
api_router.include_router(analytics.router, prefix="/analytics")
