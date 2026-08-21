"""
Forecasting schemas.
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    arima_order: list[int] = Field(default=[1, 1, 1], min_length=3, max_length=3)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ModelTrainingRequest(BaseModel):
    """Request to train a model."""
    model_type: ModelType = Field(..., description="Model type to train")
    config: TrainingConfig | None = None
    sales_column: str = "Sales"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ModelTrainingResponse(BaseModel):
    """Response after model training."""
    model_name: str
    model_type: str
    train_loss: float | None = None
    val_loss: float | None = None
    epochs_trained: int | None = None
    training_time_seconds: float | None = None
    metadata: dict[str, Any] = {}


class TrainingResultResponse(BaseModel):
    """Training result with metrics."""
    model_name: str
    model_type: str
    train_loss: float | None = None
    val_loss: float | None = None
    epochs_trained: int | None = None
    history: dict[str, list[float]] | None = None
    metadata: dict[str, Any] = {}


class PredictionRequest(BaseModel):
    """Request for predictions on test data."""
    model_type: ModelType = Field(..., description="Model to use for prediction")


class PredictionResponse(BaseModel):
    """Prediction response."""
    model_name: str
    predictions: list[float]
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
    forecasts: dict[str, list[float]]
    dates: list[str]
    horizon: int
    last_training_date: str | None = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    is_trained: bool
    training_config: dict | None = None
    metrics: dict[str, float] | None = None
    last_trained: datetime | None = None
    file_path: str | None = None


class ModelListResponse(BaseModel):
    """List of available models."""
    models: list[ModelInfo]
    total: int


class ModelTrainingJobRequest(BaseModel):
    """Request to start a training job."""
    model_type: ModelType
    config: TrainingConfig | None = None
    sales_column: str = "Sales"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class ModelTrainingJobResponse(BaseModel):
    """Response after starting a training job."""
    job_id: str
    status: str = "queued"
    message: str = "Training job queued successfully"
