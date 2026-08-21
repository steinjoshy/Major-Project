"""
Forecasting endpoints.
"""
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.app.schemas.common import (
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    SuccessResponse,
)
from backend.app.schemas.forecast import (
    FutureForecastRequest,
    FutureForecastResponse,
    ModelInfo,
    ModelListResponse,
    ModelTrainingJobRequest,
    ModelTrainingRequest,
    ModelTrainingResponse,
)

router = APIRouter(prefix="/forecast", tags=["forecast"])

# In-memory job storage (replace with Redis in production)
_job_store: dict[str, dict[str, Any]] = {}
_job_counter = 0


def _generate_job_id() -> str:
    global _job_counter
    _job_counter += 1
    return f"job_{_job_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
)
async def list_models():
    """
    List all available forecasting models and their status.
    """
    settings = __import__("backend.app.core.config", fromlist=["get_settings"]).get_settings()

    # For now, return static info about available models
    models = [
        ModelInfo(
            name="LSTM",
            type="lstm",
            is_trained=False,
            training_config={
                "seq_length": 30,
                "epochs": 50,
                "batch_size": 32,
            },
            metrics=None,
            last_trained=None,
            file_path=None,
        ),
        ModelInfo(
            name="Hybrid ARIMA+XGBoost",
            type="hybrid",
            is_trained=False,
            training_config={
                "arima_order": [1, 1, 1],
                "residual_lags": 10,
            },
            metrics=None,
            last_trained=None,
            file_path=None,
        ),
    ]

    return ModelListResponse(models=models, total=len(models))


@router.post(
    "/train",
    response_model=ModelTrainingResponse,
    summary="Train a forecasting model",
)
async def train_model(
    request: ModelTrainingRequest,
):
    """
    Train a forecasting model (LSTM or Hybrid ARIMA+XGBoost).
    
    This is a synchronous training endpoint. For long-running training,
    use the /jobs/train endpoint instead.
    """

    # This is a simplified version - in reality, we'd need the data to be loaded
    # For now, return a placeholder response
    return ModelTrainingResponse(
        model_name=request.model_type.value,
        model_type=request.model_type.value,
        train_loss=None,
        val_loss=None,
        epochs_trained=0,
        training_time_seconds=None,
        metadata={
            "message": "Training requires data to be loaded first. Use the data upload endpoints first.",
            "config": request.config.dict() if request.config else {},
        },
    )


@router.post(
    "/future",
    response_model=FutureForecastResponse,
    summary="Generate future forecasts",
)
async def generate_future_forecast(
    request: FutureForecastRequest,
):
    """
    Generate future forecasts from trained models.
    
    Requires models to be trained first via /train endpoint.
    """
    return FutureForecastResponse(
        forecasts={},
        dates=[],
        horizon=request.forecast_steps,
        last_training_date=None,
    )


@router.post(
    "/jobs/train",
    response_model=JobCreateResponse,
    summary="Queue a model training job",
)
async def queue_training_job(
    request: ModelTrainingJobRequest,
    background_tasks: BackgroundTasks,
):
    """
    Queue a model training job for background execution.
    
    Returns a job ID that can be polled for status.
    """
    job_id = _generate_job_id()

    job_data = {
        "id": job_id,
        "job_type": "train",
        "status": JobStatus.QUEUED.value,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "error": None,
        "result": None,
        "payload": request.dict(),
    }

    _job_store[job_id] = job_data

    # Add background task
    background_tasks.add_task(_run_training_job, job_id, request)

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Training job queued successfully",
    )


async def _run_training_job(job_id: str, request: ModelTrainingJobRequest):
    """Background task to run model training."""
    job = _job_store.get(job_id)
    if not job:
        return

    job["status"] = JobStatus.RUNNING.value
    job["started_at"] = datetime.utcnow()

    try:
        # TODO: Implement actual training logic here
        # This would use ForecastingService.train_all_models()
        # For now, simulate completion
        import asyncio
        await asyncio.sleep(2)  # Simulate work

        job["status"] = JobStatus.COMPLETED.value
        job["completed_at"] = datetime.utcnow()
        job["progress"] = 1.0
        job["result"] = {
            "message": "Training completed (simulated)",
            "models": ["LSTM", "Hybrid ARIMA+XGBoost"],
        }
    except Exception as e:
        job["status"] = JobStatus.FAILED.value
        job["completed_at"] = datetime.utcnow()
        job["error"] = str(e)


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get training job status",
)
async def get_job_status(
    job_id: str,
):
    """
    Get status of a training job.
    """
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found"})

    return JobStatusResponse(**job)


@router.get(
    "/jobs",
    summary="List training jobs",
)
async def list_jobs(
    page: int = 1,
    page_size: int = 20,
):
    """
    List all training jobs with pagination.
    """
    jobs = list(_job_store.values())
    total = len(jobs)

    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "jobs": jobs[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post(
    "/predict",
    summary="Generate predictions on test data",
)
async def predict(
    model_type: str = "lstm",
):
    """
    Generate predictions on test data using trained model.
    """
    return SuccessResponse(
        success=False,
        message="This endpoint requires trained models. Use /train first.",
    )


@router.get(
    "/models/{model_name}",
    summary="Get model details",
)
async def get_model_info(
    model_name: str,
):
    """
    Get detailed information about a specific model.
    """
    return SuccessResponse(
        success=False,
        message="Model info requires trained models. Use /models to list available models.",
    )


@router.delete(
    "/models/{model_name}",
    summary="Delete a model",
)
async def delete_model(
    model_name: str,
):
    """
    Delete a trained model from memory.
    """
    return SuccessResponse(
        success=False,
        message="Model deletion not yet implemented.",
    )
