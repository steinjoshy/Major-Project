"""
Forecasting schemas.
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ModelType(str, Enum):
    """Model type enumeration."""
    LSTM = "lstm"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class TrainingConfig(BaseModel):
    """Training configuration."""
    seq_length: int = Field(default=30, ge=10, le=100)
    lstm_epochs: int = Field(default=50, ge=1, le=500)
    lstm_batch_size: int = Field(default=32, ge=8, le=256)
    arima_order: List[int] = Field(default=[1, 1, 1], min_length=3, max_length=3)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ModelTrainingRequest(BaseModel):
    """Request to train a model."""
    model_type: ModelType = Field(..., description="Model type to train")
    config: Optional[TrainingConfig] = None
    sales_column: str = "Sales"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ModelTrainingResponse(BaseModel):
    """Response after model training."""
    model_name: str
    model_type: str
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    training_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = {}


class TrainingResultResponse(BaseModel):
    """Training result with metrics."""
    model_name: str
    model_type: str
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    history: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    """Request for predictions on test data."""
    model_type: ModelType = Field(..., description="Model to use for prediction")


class PredictionResponse(BaseModel):
    """Prediction response."""
    model_name: str
    predictions: List[float]
    n_samples: int


class FutureForecastRequest(BaseModel):
    """Request for future forecasting."""
    forecast_steps: int = Field(default=30, ge=1, le=365, description="Number of future steps")
    include_ensemble: bool = True


class ForecastPoint(BaseModel):
    """Single forecast point."""
    date: str
    value: float
    model: str


class FutureForecastResponse(BaseModel):
    """Future forecast response."""
    forecasts: Dict[str, List[float]]
    dates: List[str]
    horizon: int
    last_training_date: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    is_trained: bool
    training_config: Optional[Dict] = None
    metrics: Optional[Dict[str, float]] = None
    last_trained: Optional[datetime] = None
    file_path: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models."""
    models: List[ModelInfo]
    total: int


class ModelTrainingJobRequest(BaseModel):
    """Request to start a training job."""
    model_type: ModelType
    config: Optional[TrainingConfig] = None
    sales_column: str = "Sales"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ModelTrainingJobResponse(BaseModel):
    """Response after starting a training job."""
    job_id: str
    status: str = "queued"
    message: str = "Training job queued successfully"