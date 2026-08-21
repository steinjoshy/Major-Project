"""
Analytics endpoints for model insights.
"""

from fastapi import APIRouter, Depends

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get(
    "/summary",
    summary="Get analytics summary",
)
async def get_analytics_summary(
    comparison_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()),
    forecasting_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_forecasting_service"]).get_forecasting_service()),
):
    """
    Get a summary of all analytics data.
    """
    comparison_service = __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()
    forecasting_service = __import__("backend.app.dependencies", fromlist=["get_forecasting_service"]).get_forecasting_service()

    results = comparison_service.get_results()
    training_results = forecasting_service.get_training_results()

    return {
        "models_evaluated": len(results),
        "models_trained": len(training_results),
        "best_model": None,  # Would come from comparison
        "total_training_time": sum(r.metadata.get("training_time", 0) for r in training_results.values()),
        "forecasting_ready": forecasting_service.is_trained(),
    }


@router.get(
    "/model-performance",
    summary="Get model performance over time",
)
async def get_model_performance():
    """
    Get model performance trends (placeholder for future implementation).
    """
    return {
        "message": "Model performance tracking not yet implemented. Will be available when model persistence is added.",
        "planned_features": [
            "Performance drift detection",
            "Retraining triggers",
            "A/B testing results",
        ],
    }


@router.get(
    "/data-quality",
    summary="Get data quality metrics",
)
async def get_data_quality():
    """
    Get data quality metrics for the loaded dataset.
    """
    return {
        "message": "Data quality metrics will be available after data upload via /api/data/upload",
        "planned_metrics": [
            "Completeness",
            "Consistency",
            "Timeliness",
            "Accuracy",
            "Outlier detection",
        ],
    }


@router.get(
    "/forecast-accuracy",
    summary="Get forecast accuracy tracking",
)
async def get_forecast_accuracy():
    """
    Get forecast accuracy metrics over time (placeholder).
    """
    return {
        "message": "Forecast accuracy tracking requires model persistence and actuals collection.",
        "planned_features": [
            "Rolling MAPE/RMSE",
            "Bias tracking",
            "Confidence interval calibration",
        ],
    }
