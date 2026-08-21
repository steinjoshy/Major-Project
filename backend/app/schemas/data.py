"""
Data ingestion schemas.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class DataSourceType(str):
    """Data source type enumeration."""
    CSV = "csv"
    EXCEL = "excel"
    FOLDER = "folder"


class ColumnInfo(BaseModel):
    """Column information."""
    name: str
    dtype: str
    sample_values: List[Any] = []
    null_count: int = 0
    unique_count: Optional[int] = None


class DataSummary(BaseModel):
    """Data summary after ingestion."""
    rows: int
    columns: List[str]
    column_info: List[ColumnInfo]
    date_range: Optional[Dict[str, str]] = None
    date_column: Optional[str] = None
    sales_column: Optional[str] = None
    warnings: List[str] = []


class ValidationResult(BaseModel):
    """Data validation result."""
    valid: bool
    rows: int
    columns: List[str]
    errors: List[str] = []
    warnings: List[str] = []


class ColumnDetectionRequest(BaseModel):
    """Request for column auto-detection."""
    columns: List[str] = Field(..., description="List of column names")


class ColumnDetectionResponse(BaseModel):
    """Response with detected columns."""
    date_column: Optional[str] = None
    sales_column: Optional[str] = None
    detected_columns: Dict[str, str] = Field(default_factory=dict)


class DataCleanRequest(BaseModel):
    """Request for data cleaning."""
    date_column: str = Field(..., description="Date column name")
    sales_column: str = Field(..., description="Sales/demand column name")


class DataCleanResponse(BaseModel):
    """Response after data cleaning."""
    success: bool
    rows_before: int
    rows_after: int
    operations: List[str] = []
    warnings: List[str] = []


class FeatureEngineeringRequest(BaseModel):
    """Request for feature engineering."""
    date_column: str = "Date"
    sales_column: str = "Sales"
    lag_features: List[int] = Field(default=[1, 7, 14, 30])
    rolling_windows: List[int] = Field(default=[7, 14, 30])


class LSTMDataPrepRequest(BaseModel):
    """Request for LSTM data preparation."""
    sales_column: str = "Sales"
    seq_length: int = Field(default=30, ge=10, le=100)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class LSTMDataPrepResponse(BaseModel):
    """Response with LSTM data preparation results."""
    X_train_shape: List[int]
    X_test_shape: List[int]
    y_train_shape: List[int]
    y_test_shape: List[int]
    train_size: int
    seq_length: int
    test_size: float


class HybridDataPrepRequest(BaseModel):
    """Request for Hybrid model data preparation."""
    sales_column: str = "Sales"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class HybridDataPrepResponse(BaseModel):
    """Response with Hybrid data preparation results."""
    train_size: int
    test_size: int
    split_idx: int


class DataUploadResponse(BaseModel):
    """Response after data upload."""
    success: bool
    upload_id: str
    summary: DataSummary
    warnings: List[str] = []


class DataPreviewResponse(BaseModel):
    """Data preview response."""
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int