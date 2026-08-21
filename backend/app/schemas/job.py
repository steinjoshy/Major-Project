"""
Job queue schemas.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str):
    """Job type enumeration."""
    TRAIN_LSTM = "train_lstm"
    TRAIN_HYBRID = "train_hybrid"
    TRAIN_ALL = "train_all"
    FORECAST = "forecast"
    INVENTORY = "inventory"


class JobCreate(BaseModel):
    """Job creation request."""
    job_type: JobType = Field(..., description="Type of job to execute")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")


class JobCreateResponse(BaseModel):
    """Job creation response."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = "queued"
    message: str = "Job queued successfully"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobStatusResponse(BaseModel):
    """Job status response."""
    id: str
    job_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    payload: Dict[str, Any] = {}


class JobListResponse(BaseModel):
    """Job list response."""
    jobs: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int