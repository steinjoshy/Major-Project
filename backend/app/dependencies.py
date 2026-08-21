"""
API dependencies for dependency injection.
"""
from fastapi import File, HTTPException, UploadFile

from backend.app.core.config import get_settings
from src.services.forecasting_service import ForecastingService
from src.services.ingestion_service import IngestionService
from src.services.inventory_service import InventoryService
from src.services.model_comparison_service import ModelComparisonService
from src.services.model_registry import ModelRegistry

# Service instances (singleton pattern for development)
_ingestion_service: IngestionService | None = None
_forecasting_service: ForecastingService | None = None
_inventory_service: InventoryService | None = None
_comparison_service: ModelComparisonService | None = None
_model_registry: ModelRegistry | None = None


def get_ingestion_service() -> IngestionService:
    """Get or create ingestion service."""
    global _ingestion_service
    if _ingestion_service is None:
        settings = get_settings()
        _ingestion_service = IngestionService(
            use_duckdb=True,
            min_rows_lstm=100
        )
    return _ingestion_service


def get_forecasting_service() -> ForecastingService:
    """Get or create forecasting service."""
    global _forecasting_service
    if _forecasting_service is None:
        _forecasting_service = ForecastingService()
    return _forecasting_service


def get_inventory_service() -> InventoryService:
    """Get or create inventory service."""
    global _inventory_service
    if _inventory_service is None:
        _inventory_service = InventoryService()
    return _inventory_service


def get_comparison_service() -> ModelComparisonService:
    """Get or create model comparison service."""
    global _comparison_service
    if _comparison_service is None:
        _comparison_service = ModelComparisonService()
    return _comparison_service


def get_model_registry() -> ModelRegistry:
    """Get or create model registry."""
    global _model_registry
    if _model_registry is None:
        settings = get_settings()
        storage_dir = settings.model_registry_dir
        _model_registry = ModelRegistry(storage_dir=storage_dir) if storage_dir else ModelRegistry()
    return _model_registry


# File upload validation
async def validate_upload_file(
    file: UploadFile = File(...),
    max_size: int = 100 * 1024 * 1024,  # 100MB
) -> UploadFile:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join({'.csv', '.xlsx', '.xls'})}"
        )

    return file


# Common parameters
class CommonQueryParams:
    """Common query parameters."""
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
    ):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), 100)
