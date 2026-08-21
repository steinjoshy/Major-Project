"""
Common schemas shared across the API.
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: "ErrorDetail" = Field(..., description="Error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "INVALID_DATA",
                    "message": "Required column 'date' was not found."
                }
            }
        }


class ErrorDetail(BaseModel):
    """Error detail."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context"
    )


class SuccessResponse(BaseModel):
    """Standard success response wrapper."""
    success: bool = True
    message: str | None = None
    data: Any | None = None


class PaginationParams(BaseModel):
    """Pagination query parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    environment: str = "development"


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobBase(BaseModel):
    """Base job model."""
    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict | None = None


class JobCreateResponse(BaseModel):
    """Response for job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = JobStatus.QUEUED
    message: str = "Job queued successfully"


class JobStatusResponse(BaseModel):
    """Job status response."""
    id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict | None = None


class FileUploadResponse(BaseModel):
    """File upload response."""
    filename: str
    size: int
    content_type: str
    upload_id: str
