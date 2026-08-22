"""
Database models for the demand forecasting system.
"""
import enum
from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
    CheckConstraint,
    JSON,
    func,
    text,
    TypeDecorator,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON

from backend.app.db.base import Base


class JSONVariant(TypeDecorator):
    """A type that uses JSONB for PostgreSQL and JSON for SQLite."""
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class JobStatus(str, enum.Enum):
    """Status of a forecasting job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, enum.Enum):
    """Type of forecasting model."""
    LSTM = "lstm"
    HYBRID = "hybrid"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"


class User(Base):
    """User account."""
    __tablename__ = "users"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    datasets: Mapped[List["Dataset"]] = relationship(back_populates="owner", lazy="selectin")
    jobs: Mapped[List["ForecastingJob"]] = relationship(back_populates="owner", lazy="selectin")

    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_created_at", "created_at"),
    )


class Dataset(Base):
    """Uploaded dataset with metadata."""
    __tablename__ = "datasets"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    date_column: Mapped[str] = mapped_column(String(100), nullable=False)
    sales_column: Mapped[str] = mapped_column(String(100), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    date_range_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    date_range_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    stats: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    owner_id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    owner: Mapped["User"] = relationship(back_populates="datasets", lazy="selectin")
    jobs: Mapped[List["ForecastingJob"]] = relationship(back_populates="dataset", lazy="selectin")

    __table_args__ = (
        Index("ix_datasets_owner_id", "owner_id"),
        Index("ix_datasets_created_at", "created_at"),
        Index("ix_datasets_name", "name"),
    )


class ForecastingJob(Base):
    """A forecasting training/forecasting job."""
    __tablename__ = "forecasting_jobs"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    job_type: Mapped[str] = mapped_column(
        SQLEnum("train", "forecast", "train_all", name="job_type_enum"),
        nullable=False,
        default="train_all",
    )
    status: Mapped[JobStatus] = mapped_column(
        SQLEnum(JobStatus),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Configuration
    config: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    seq_length: Mapped[int] = mapped_column(Integer, default=30)
    lstm_epochs: Mapped[int] = mapped_column(Integer, default=50)
    lstm_batch_size: Mapped[int] = mapped_column(Integer, default=32)
    arima_order: Mapped[list] = mapped_column(JSONVariant(), default=lambda: [1, 1, 1])
    forecast_steps: Mapped[int] = mapped_column(Integer, default=30)
    test_size: Mapped[float] = mapped_column(Float, default=0.2)
    
    # Model references
    lstm_model_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    hybrid_model_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Results
    results: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    lstm_metrics: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    hybrid_metrics: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    best_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Ownership
    owner_id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    dataset_id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    owner: Mapped["User"] = relationship(back_populates="jobs", lazy="selectin")
    dataset: Mapped["Dataset"] = relationship(back_populates="jobs", lazy="selectin")
    forecasts: Mapped[List["Forecast"]] = relationship(back_populates="job", lazy="selectin")

    __table_args__ = (
        Index("ix_forecasting_jobs_owner_id", "owner_id"),
        Index("ix_forecasting_jobs_dataset_id", "dataset_id"),
        Index("ix_forecasting_jobs_status", "status"),
        Index("ix_forecasting_jobs_created_at", "created_at"),
        Index("ix_forecasting_jobs_status_created", "status", "created_at"),
    )


class Forecast(Base):
    """Forecast results."""
    __tablename__ = "forecasts"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), ForeignKey("forecasting_jobs.id"), nullable=False)
    model_type: Mapped[ModelType] = mapped_column(
        SQLEnum(ModelType),
        nullable=False,
    )
    forecast_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_data: Mapped[list] = mapped_column(JSONVariant(), nullable=False)
    forecast_dates: Mapped[list] = mapped_column(JSONVariant(), nullable=False)
    forecast_metadata: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    job: Mapped["ForecastingJob"] = relationship(back_populates="forecasts", lazy="selectin")

    __table_args__ = (
        Index("ix_forecasts_job_id", "job_id"),
        Index("ix_forecasts_model_type", "model_type"),
        Index("ix_forecasts_created_at", "created_at"),
    )


class InventoryAnalysis(Base):
    """Inventory analysis results."""
    __tablename__ = "inventory_analysis"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), ForeignKey("forecasting_jobs.id"), nullable=False)
    forecast_id: Mapped[Optional[uuid4]] = mapped_column(UUID(as_uuid=True), ForeignKey("forecasts.id"), nullable=True)
    
    # Input parameters
    service_level: Mapped[float] = mapped_column(Float, nullable=False)
    lead_time: Mapped[int] = mapped_column(Integer, nullable=False)
    current_stock: Mapped[float] = mapped_column(Float, nullable=False)
    annual_demand: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    holding_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ordering_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    forecast_data: Mapped[list] = mapped_column(JSONVariant(), nullable=False)
    
    # Results
    safety_stock: Mapped[float] = mapped_column(Float, nullable=False)
    reorder_point: Mapped[float] = mapped_column(Float, nullable=False)
    average_daily_demand: Mapped[float] = mapped_column(Float, nullable=False)
    demand_std_dev: Mapped[float] = mapped_column(Float, nullable=False)
    lead_time_demand: Mapped[float] = mapped_column(Float, nullable=False)
    safety_stock_basis: Mapped[str] = mapped_column(String(50), nullable=False)
    economic_order_quantity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stockout_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overstock_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    projection: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    job: Mapped["ForecastingJob"] = relationship(lazy="selectin")
    forecast: Mapped[Optional["Forecast"]] = relationship(lazy="selectin")

    __table_args__ = (
        Index("ix_inventory_analysis_job_id", "job_id"),
        Index("ix_inventory_analysis_created_at", "created_at"),
    )


class ModelRegistry(Base):
    """Model registry for trained models."""
    __tablename__ = "model_registry"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    model_type: Mapped[ModelType] = mapped_column(SQLEnum(ModelType), nullable=False)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Configuration
    training_config: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    feature_config: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Tags
    tags: Mapped[list] = mapped_column(JSONVariant(), default=list)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    __table_args__ = (
        Index("ix_model_registry_model_type", "model_type"),
        Index("ix_model_registry_created_at", "created_at"),
        Index("ix_model_registry_tags", "tags", postgresql_using="gin"),
    )


class ModelMetrics(Base):
    """Detailed metrics for trained models."""
    __tablename__ = "model_metrics"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), ForeignKey("model_registry.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Metrics
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Test details
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    test_period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    forecast_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Additional metrics
    additional_metrics: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    model: Mapped["ModelRegistry"] = relationship(lazy="selectin")

    __table_args__ = (
        Index("ix_model_metrics_model_id", "model_id"),
        Index("ix_model_metrics_model_version", "model_id", "model_version"),
        Index("ix_model_metrics_evaluated_at", "evaluated_at"),
        UniqueConstraint("model_id", "model_version", name="uq_model_metrics_version"),
    )


class AuditLog(Base):
    """Audit log for important operations."""
    __tablename__ = "audit_logs"

    id: Mapped[uuid4] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[uuid4]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_id: Mapped[Optional[uuid4]] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    old_values: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    new_values: Mapped[Optional[dict]] = mapped_column(JSONVariant(), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship(lazy="selectin")

    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )