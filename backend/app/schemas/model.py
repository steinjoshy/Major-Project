"""
Model schemas for comparison and registry.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


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
    models: List[ModelMetrics]
    best_model: str
    best_metrics: Dict[str, float]
    improvement_pct: float
    metric_used: str


class BestModelResponse(BaseModel):
    """Best model response."""
    best_model: str
    metrics: Dict[str, float]
    improvement_pct: float


class ModelRegistryEntry(BaseModel):
    """Model registry entry."""
    name: str
    model_type: str
    version: str
    created_at: str
    training_config: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}
    feature_config: Dict[str, Any] = {}
    file_path: Optional[str] = None
    description: str = ""
    tags: List[str] = []


class ModelRegistryRequest(BaseModel):
    """Request to register a model."""
    name: str
    model_type: str
    version: str = "1.0.0"
    training_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    feature_config: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    description: str = ""
    tags: List[str] = []
    overwrite: bool = False


class ModelRegistryResponse(BaseModel):
    """Model registry response."""
    models: List[Dict[str, Any]]
    total: int


class ModelRegistryDetailResponse(BaseModel):
    """Single model registry detail."""
    name: str
    model_type: str
    version: str
    created_at: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    feature_config: Dict[str, Any]
    file_path: Optional[str]
    description: str
    tags: List[str]