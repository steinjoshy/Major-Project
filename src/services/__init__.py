"""
Service layer for AI Demand Forecasting and Inventory Optimization.

This package provides pure-Python domain services extracted from the Streamlit monolith.
Services are designed to be used independently of any web framework.
"""

from src.services.ingestion_service import IngestionService, IngestionError
from src.services.forecasting_service import (
    ForecastingService,
    ModelConfig,
    TrainingResult,
    ForecastResult,
)
from src.services.inventory_service import (
    InventoryService,
    InventoryParams,
    InventoryRecommendations,
    InventoryProjection,
)
from src.services.model_comparison_service import (
    ModelComparisonService,
    ModelMetrics,
    ComparisonResult,
)
from src.services.model_registry import ModelRegistry, ModelMetadata, create_registry

__all__ = [
    # Ingestion
    'IngestionService',
    'IngestionError',
    # Forecasting
    'ForecastingService',
    'ModelConfig',
    'TrainingResult',
    'ForecastResult',
    # Inventory
    'InventoryService',
    'InventoryParams',
    'InventoryRecommendations',
    'InventoryProjection',
    # Model Comparison
    'ModelComparisonService',
    'ModelMetrics',
    'ComparisonResult',
    # Model Registry
    'ModelRegistry',
    'ModelMetadata',
    'create_registry',
]