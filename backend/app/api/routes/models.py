"""
Model comparison and registry endpoints.
"""

from fastapi import APIRouter, Depends, Form, HTTPException

from backend.app.schemas.model import (
    BestModelResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ModelMetrics,
    ModelRegistryDetailResponse,
    ModelRegistryRequest,
    ModelRegistryResponse,
)

router = APIRouter(prefix="/models", tags=["models"])


# Model Comparison Endpoints

@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    summary="Compare multiple models",
)
async def compare_models(
    request: ModelComparisonRequest,
    y_true: list[float] = Form(...),
    comparison_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()),
):
    """
    Compare multiple models and return sorted results.
    
    - **y_true**: Actual test values
    - **predictions**: Dictionary of model_name -> predictions array
    - **metric**: Metric to sort by (RMSE, MAE, MAPE)
    """
    try:
        predictions_str = Form(...)
        # In practice, you'd pass predictions as JSON
        # For now, we'll expect a JSON string
        pass

        # This endpoint needs y_true and predictions_dict
        # For a proper implementation, we'd accept multipart with JSON
        raise HTTPException(
            status_code=501,
            detail={"code": "NOT_IMPLEMENTED", "message": "This endpoint requires multipart with JSON. Use /compare/json endpoint."}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Comparison failed: {str(e)}"})


@router.post(
    "/compare/json",
    response_model=ModelComparisonResponse,
    summary="Compare models with JSON payload",
)
async def compare_models_json(
    y_true: list[float],
    predictions: dict[str, list[float]],
    metric: str = "RMSE",
    comparison_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()),
):
    """
    Compare multiple models using JSON payload.
    
    - **y_true**: Actual test values
    - **predictions**: Dictionary mapping model_name -> predictions array
    - **metric**: Metric to sort by (RMSE, MAE, MAPE)
    """
    try:
        comparison_service = __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()

        result = comparison_service.compare_models(y_true, predictions, metric)

        return ModelComparisonResponse(
            models=[m.to_dict() for m in result.metrics_table.to_dict('records')],
            best_model=result.best_model,
            best_metrics=result.best_metrics.to_dict(),
            improvement_pct=result.improvement_pct,
            metric_used=result.metric_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Comparison failed: {str(e)}"})


@router.post(
    "/evaluate",
    response_model=ModelMetrics,
    summary="Evaluate a single model",
)
async def evaluate_model(
    y_true: list[float],
    y_pred: list[float],
    model_name: str = "Model",
    comparison_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()),
):
    """
    Evaluate a single model and return metrics.
    """
    try:
        comparison_service = __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()

        metrics = comparison_service.evaluate_model(y_true, y_pred, model_name)

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Evaluation failed: {str(e)}"})


@router.get(
    "/best",
    response_model=BestModelResponse,
    summary="Get best model from last comparison",
)
async def get_best_model(
    metric: str = "RMSE",
    comparison_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()),
):
    """
    Get the best model from the last comparison.
    """
    try:
        comparison_service = __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()
        best = comparison_service.get_best_model(metric)
        return BestModelResponse(**best)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Failed to get best model: {str(e)}"})


@router.get(
    "/metrics",
    summary="Get all evaluated model metrics",
)
async def get_all_metrics(
    comparison_service = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()),
):
    """
    Get all evaluated model metrics from the last comparison.
    """
    comparison_service = __import__("backend.app.dependencies", fromlist=["get_comparison_service"]).get_comparison_service()
    results = comparison_service.get_results()
    return {name: m.to_dict() for name, m in results.items()}


# Model Registry Endpoints

@router.post(
    "/registry",
    response_model=ModelRegistryDetailResponse,
    summary="Register a model",
)
async def register_model(
    request: ModelRegistryRequest,
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    Register a model in the registry.
    """
    try:
        model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()

        metadata = model_registry.register(
            name=request.name,
            model_type=request.model_type,
            model_object=None,  # Object would be passed separately in practice
            version=request.version,
            training_config=request.training_config,
            metrics=request.metrics,
            feature_config=request.feature_config,
            file_path=request.file_path,
            description=request.description,
            tags=request.tags,
            overwrite=request.overwrite,
        )

        return ModelRegistryDetailResponse(**metadata.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "MODEL_EXISTS", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Registration failed: {str(e)}"})


@router.get(
    "/registry",
    response_model=ModelRegistryResponse,
    summary="List all registered models",
)
async def list_models(
    model_type: str | None = None,
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    List all registered models, optionally filtered by type.
    """
    model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()
    models = model_registry.list(model_type=model_type)

    return ModelRegistryResponse(
        models=[m.to_dict() for m in models],
        total=len(models),
    )


@router.get(
    "/registry/{model_name}",
    response_model=ModelRegistryDetailResponse,
    summary="Get model details",
)
async def get_model(
    model_name: str,
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    Get detailed information about a registered model.
    """
    model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()
    metadata = model_registry.get(model_name)

    if metadata is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": f"Model '{model_name}' not found"})

    return ModelRegistryDetailResponse(**metadata.to_dict())


@router.delete(
    "/registry/{model_name}",
    summary="Remove a model from registry",
)
async def remove_model(
    model_name: str,
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    Remove a model from the registry.
    """
    model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()

    success = model_registry.remove(model_name)

    if not success:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": f"Model '{model_name}' not found"})

    return {"success": True, "message": f"Model '{model_name}' removed"}


@router.get(
    "/registry/latest",
    summary="Get latest registered model",
)
async def get_latest_model(
    model_type: str | None = None,
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    Get the most recently registered model.
    """
    model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()
    latest = model_registry.get_latest(model_type)

    if latest is None:
        raise HTTPException(status_code=404, detail={"code": "NO_MODELS", "message": "No models registered"})

    return ModelRegistryDetailResponse(**latest.to_dict())


@router.post(
    "/registry/{model_name}/metrics",
    summary="Update model metrics",
)
async def update_model_metrics(
    model_name: str,
    metrics: dict[str, float],
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    Update metrics for a registered model.
    """
    model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()

    success = model_registry.update_metrics(model_name, metrics)

    if not success:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": f"Model '{model_name}' not found"})

    return {"success": True, "message": f"Metrics updated for '{model_name}'"}


@router.post(
    "/registry/{model_name}/tags",
    summary="Add tags to a model",
)
async def add_model_tags(
    model_name: str,
    tags: list[str],
    model_registry = Depends(lambda: __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()),
):
    """
    Add tags to a registered model.
    """
    model_registry = __import__("backend.app.dependencies", fromlist=["get_model_registry"]).get_model_registry()

    success = model_registry.add_tags(model_name, tags)

    if not success:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": f"Model '{model_name}' not found"})

    return {"success": True, "message": f"Tags added to '{model_name}'"}
