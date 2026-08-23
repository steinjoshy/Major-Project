"""
Model schemas for comparison and registry.
"""
from typing import Any

from pydantic import BaseModel, Field


class MetricType(str):
    """Metric type enumeration."""
    RMSE = "RMSE"
    MAE = "MAE"
    MAPE = "MAPE"


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    n_samples: int


class ModelComparisonRequest(BaseModel):
    """Request for model comparison."""
    metric: str = Field(default="RMSE", description="Metric to sort by (RMSE, MAE, MAPE)")


class ModelComparisonResponse(BaseModel):
    """Model comparison response."""
    models: list[ModelMetrics]
    best_model: str
    best_metrics: dict[str, float]
    improvement_pct: float
    metric_used: str


class BestModelResponse(BaseModel):
    """Best model response."""
    best_model: str
    metrics: dict[str, float]
    improvement_pct: float


class ModelRegistryEntry(BaseModel):
    """Model registry entry."""
    name: str
    model_type: str
    version: str
    created_at: str
    training_config: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    feature_config: dict[str, Any] = {}
    file_path: str | None = None
    description: str = ""
    tags: list[str] = []


class ModelRegistryRequest(BaseModel):
    """Request to register a model."""
    name: str
    model_type: str
    version: str = "1.0.0"
    training_config: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    feature_config: dict[str, Any] | None = None
    file_path: str | None = None
    description: str = ""
    tags: list[str] = []
    overwrite: bool = False


class ModelRegistryResponse(BaseModel):
    """Model registry response."""
    models: list[dict[str, Any]]
    total: int


class ModelRegistryDetailResponse(BaseModel):
    """Single model registry detail."""
    name: str
    model_type: str
    version: str
    created_at: str
    training_config: dict[str, Any]
    metrics: dict[str, float]
    feature_config: dict[str, Any]
    file_path: str | None
    description: str
    tags: list[str]
