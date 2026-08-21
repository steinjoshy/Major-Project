"""
Data ingestion endpoints.
"""
import io
import tempfile
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from typing import Any, Dict, List
import pandas as pd

from backend.app.schemas.data import (
    DataUploadResponse,
    DataSummary,
    ValidationResult,
    ColumnDetectionResponse,
    DataCleanRequest,
    DataCleanResponse,
    LSTMDataPrepRequest,
    LSTMDataPrepResponse,
    HybridDataPrepRequest,
    HybridDataPrepResponse,
    FeatureEngineeringRequest,
    DataPreviewResponse,
    ColumnDetectionResponse,
)
from backend.app.schemas.common import SuccessResponse, ErrorResponse, PaginationParams, PaginatedResponse
from backend.app.dependencies import get_ingestion_service, validate_upload_file
from src.services.ingestion_service import IngestionService, IngestionError
from backend.app.core.config import get_settings

router = APIRouter(prefix="/data", tags=["data"])


def _save_upload_file(file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path."""
    import os
    import tempfile
    
    # Create temp directory if not exists
    temp_dir = Path(tempfile.gettempdir()) / "demand_forecasting_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename
    import uuid
    ext = Path(file.filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"
    file_path = temp_dir / safe_name
    
    # Write file
    with open(file_path, "wb") as f:
        content = file.file.read()
        f.write(content)
    
    return str(file_path)


@router.post(
    "/upload",
    response_model=DataUploadResponse,
    summary="Upload and preprocess CSV data",
    responses={
        200: {"model": DataUploadResponse},
        400: {"model": ErrorResponse},
        413: {"description": "File too large"},
        422: {"model": ErrorResponse},
    },
)
async def upload_data(
    file: UploadFile = File(...),
    date_column: Optional[str] = Form(None),
    sales_column: Optional[str] = Form(None),
    ingestion_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()),
):
    """
    Upload and preprocess a CSV file for demand forecasting.
    
    - **file**: CSV file with date and sales/demand columns
    - **date_column**: Optional date column name (auto-detected if not provided)
    - **sales_column**: Optional sales/demand column name (auto-detected if not provided)
    
    Supports:
    - Standard CSV with date and sales columns
    - M5 wide format (d_* columns automatically melted)
    - Multiple files merged by date
    - Large files via chunked reading or DuckDB
    """
    try:
        # Save uploaded file
        file_path = _save_upload_file(file)
        
        # Process through ingestion service
        ingestion_service = __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()
        df, meta = ingestion_service.load_from_csv(
            file_path,
            date_col=date_column,
            sales_col=sales_column,
        )
        
        # Clean up temp file
        import os
        try:
            os.unlink(file_path)
        except Exception:
            pass
        
        # Build summary
        summary = DataSummary(
            rows=meta["n_rows"],
            columns=list(df.columns),
            date_range=meta.get("date_range"),
            date_column=meta["date_col"],
            sales_column=meta["sales_col"],
            warnings=meta.get("warnings", []),
        )
        
        # Read column info
        column_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "sample_values": df[col].head(3).tolist() if len(df) > 0 else [],
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()) if df[col].dtype == "object" else None,
            }
            # Add sample values as list
            col_info["sample_values"] = df[col].head(3).tolist() if len(df) > 0 else []
            column_info.append(col_info)
        summary.column_info = column_info
        
        return DataUploadResponse(
            success=True,
            upload_id=Path(file_path).stem,
            summary=summary,
            warnings=meta.get("warnings", []),
        )
        
    except IngestionError as e:
        raise HTTPException(status_code=400, detail={"code": "INGESTION_ERROR", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Upload failed: {str(e)}"})


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Validate CSV data structure",
)
async def validate_data(
    file: UploadFile = File(...),
    date_column: Optional[str] = Form(None),
    sales_column: Optional[str] = Form(None),
):
    """
    Validate a CSV file without full processing.
    
    Returns validation results with any errors or warnings.
    """
    try:
        file_path = _save_upload_file(file)
        
        try:
            ingestion_service = __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()
            df = pd.read_csv(file_path, nrows=1000)
            
            # Try to detect columns
            date_col = date_column
            sales_col = sales_column
            
            if date_column is None or sales_column is None:
                from src.services.ingestion_service import IngestionService
                temp_service = IngestionService()
                auto_dc, auto_sc = temp_service._auto_detect_cols(list(df.columns))
                date_col = date_col or auto_dc
                sales_col = sales_col or auto_sc
            
            if date_col is None or sales_col is None:
                return ValidationResult(
                    valid=False,
                    rows=len(df),
                    columns=list(df.columns),
                    errors=["Could not auto-detect date/sales columns"],
                    warnings=[],
                )
            
            # Validate
            is_valid, messages = temp_service.validate_data(df, date_col, sales_col)
            
            return ValidationResult(
                valid=is_valid,
                rows=len(df),
                columns=list(df.columns),
                errors=[m for m in messages if not m.startswith("Warning")],
                warnings=[m for m in messages if m.startswith("Warning")],
            )
        finally:
            import os
            try:
                os.unlink(file_path)
            except Exception:
                pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Validation failed: {str(e)}"})


@router.post(
    "/detect-columns",
    response_model=ColumnDetectionResponse,
    summary="Auto-detect date and sales columns",
)
async def detect_columns(
    file: UploadFile = File(...),
):
    """
    Auto-detect date and sales/demand columns from CSV header.
    """
    try:
        file_path = _save_upload_file(file)
        
        try:
            df = pd.read_csv(file_path, nrows=1)
            ingestion_service = __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()
            date_col, sales_col = ingestion_service._auto_detect_cols(list(df.columns))
            
            detected = {}
            if date_col:
                detected["date_column"] = date_col
            if sales_col:
                detected["sales_column"] = sales_col
            
            return ColumnDetectionResponse(
                date_column=date_col,
                sales_column=sales_col,
                detected_columns=detected,
            )
        finally:
            import os
            try:
                os.unlink(file_path)
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Detection failed: {str(e)}"})


@router.post(
    "/clean",
    response_model=DataCleanResponse,
    summary="Clean and preprocess data",
)
async def clean_data(
    request: DataCleanRequest,
    ingestion_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()),
):
    """
    Clean and preprocess data.
    
    Requires date_column and sales_column to be specified.
    """
    # This endpoint would need the data to already be loaded
    # For now, return a message indicating this requires uploaded data
    return DataCleanResponse(
        success=False,
        rows_before=0,
        rows_after=0,
        operations=[],
        warnings=["This endpoint requires data to be loaded first via /upload. Use the upload endpoint first."],
    )


@router.post(
    "/prepare/lstm",
    response_model=LSTMDataPrepResponse,
    summary="Prepare data for LSTM training",
)
async def prepare_lstm_data(
    request: LSTMDataPrepRequest,
    ingestion_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()),
):
    """
    Prepare LSTM training data from loaded dataset.
    
    Requires data to be loaded first via /upload.
    """
    # This would need the last processed data
    return LSTMDataPrepResponse(
        X_train_shape=[0, 0, 0],
        X_test_shape=[0, 0, 0],
        y_train_shape=[0],
        y_test_shape=[0],
        train_size=0,
        seq_length=request.seq_length,
        test_size=request.test_size,
    )


@router.post(
    "/prepare/hybrid",
    response_model=HybridDataPrepResponse,
    summary="Prepare data for Hybrid model",
)
async def prepare_hybrid_data(
    request: HybridDataPrepRequest,
    ingestion_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_ingestion_service"]).get_ingestion_service()),
):
    """
    Prepare Hybrid model data from loaded dataset.
    """
    return HybridDataPrepResponse(
        train_size=0,
        test_size=0,
        split_idx=0,
    )


@router.get(
    "/preview",
    response_model=DataPreviewResponse,
    summary="Preview loaded data",
)
async def preview_data(
    page: int = 1,
    page_size: int = 50,
):
    """
    Preview the currently loaded data.
    """
    return DataPreviewResponse(
        columns=[],
        rows=[],
        total_rows=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/summary",
    response_model=DataSummary,
    summary="Get data summary",
)
async def get_data_summary():
    """
    Get summary of currently loaded data.
    """
    return DataSummary(
        rows=0,
        columns=[],
        column_info=[],
    )


@router.post(
    "/features",
    summary="Generate time/lag/rolling features",
)
async def create_features(
    request: FeatureEngineeringRequest,
):
    """
    Generate time-based, lag, and rolling features.
    """
    return SuccessResponse(
        success=True,
        message="Feature generation requires loaded data. Use /upload first.",
    )