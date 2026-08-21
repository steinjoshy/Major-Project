"""
Inventory optimization endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Form
from typing import Any, Dict, List, Optional
from datetime import datetime

from backend.app.schemas.inventory import (
    InventoryParams,
    InventoryRecommendationsResponse,
    InventoryProjectionRequest,
    InventoryProjectionResponse,
    RiskAnalysisRequest,
    RiskAnalysisResponse,
    InventoryReportResponse,
)
from backend.app.schemas.common import SuccessResponse, ErrorResponse
from backend.app.dependencies import get_inventory_service
from src.services.inventory_service import InventoryService, InventoryParams as ServiceInventoryParams
from backend.app.core.config import get_settings

router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.post(
    "/analyze",
    response_model=InventoryRecommendationsResponse,
    summary="Generate inventory recommendations",
)
async def analyze_inventory(
    demand_data: List[float] = Form(...),
    lead_time: int = Form(default=7, ge=1, le=365),
    service_level: float = Form(default=0.95, gt=0.0, lt=1.0),
    forecast_data: Optional[List[float]] = Form(None),
    annual_demand: Optional[float] = Form(None, gt=0),
    holding_cost: Optional[float] = Form(None, gt=0),
    ordering_cost: Optional[float] = Form(None, gt=0),
):
    """
    Generate comprehensive inventory recommendations.
    
    Calculates:
    - Safety Stock (historical or forecast-based)
    - Reorder Point (ROP)
    - Lead-time demand
    - Economic Order Quantity (EOQ) if cost parameters provided
    - Risk indicators
    """
    try:
        inventory_service = __import__("backend.app.dependencies", fromlist=["get_inventory_service"]).get_inventory_service()
        
        params = InventoryParams(
            service_level=service_level,
            lead_time=lead_time,
            annual_demand=annual_demand,
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
        )
        
        recommendations = inventory_service.generate_recommendations(
            demand_data=demand_data,
            params=params,
            forecast_data=forecast_data,
        )
        
        return InventoryRecommendationsResponse(**recommendations.to_dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Analysis failed: {str(e)}"})


@router.post(
    "/project",
    response_model=InventoryProjectionResponse,
    summary="Project inventory levels",
)
async def project_inventory(
    request: InventoryProjectionRequest,
):
    """
    Project inventory levels over the forecast horizon.
    
    Simulates day-by-day inventory levels with order triggers.
    """
    try:
        inventory_service = __import__("backend.app.dependencies", fromlist=["get_inventory_service"]).get_inventory_service()
        
        projection = inventory_service.project_inventory(
            forecast_demand=request.forecast_demand,
            current_stock=request.current_stock,
            reorder_point=request.reorder_point,
            lead_time=request.lead_time,
            safety_stock=request.safety_stock,
        )
        
        return InventoryProjectionResponse(
            periods=projection.periods.tolist(),
            forecast_demand=projection.forecast_demand.tolist(),
            inventory_levels=projection.inventory_levels.tolist(),
            orders_placed=projection.orders_placed.tolist(),
            reorder_point=projection.reorder_point,
            safety_stock=projection.safety_stock,
            lead_time=projection.lead_time,
            stockout_days=projection.get_stockout_days(),
            below_safety_stock_days=projection.get_below_safety_stock_days(),
            total_orders_placed=projection.get_total_orders_placed(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Projection failed: {str(e)}"})


@router.post(
    "/risk",
    response_model=RiskAnalysisResponse,
    summary="Calculate risk metrics",
)
async def analyze_risk(
    request: RiskAnalysisRequest,
):
    """
    Calculate stockout and overstock risk metrics.
    """
    try:
        inventory_service = __import__("backend.app.dependencies", fromlist=["get_inventory_service"]).get_inventory_service()
        
        stockout_risk = inventory_service.calculate_stockout_risk(
            demand_data=request.demand_data,
            current_stock=request.current_stock,
            lead_time=request.lead_time,
            service_level=request.service_level,
        )
        
        overstock_risk = None
        # Overstock risk requires holding cost
        # Could add holding_cost to request if needed
        
        return RiskAnalysisResponse(
            stockout_risk=stockout_risk,
            overstock_risk=overstock_risk,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Risk analysis failed: {str(e)}"})


@router.post(
    "/report",
    response_model=InventoryReportResponse,
    summary="Generate inventory report",
)
async def generate_report(
    demand_data: List[float] = Form(...),
    lead_time: int = Form(default=7, ge=1, le=365),
    service_level: float = Form(default=0.95, gt=0.0, lt=1.0),
    forecast_data: Optional[List[float]] = Form(None),
    annual_demand: Optional[float] = Form(None, gt=0),
    holding_cost: Optional[float] = Form(None, gt=0),
    ordering_cost: Optional[float] = Form(None, gt=0),
):
    """
    Generate a human-readable inventory optimization report.
    """
    try:
        inventory_service = __import__("backend.app.dependencies", fromlist=["get_inventory_service"]).get_inventory_service()
        
        params = InventoryParams(
            service_level=service_level,
            lead_time=lead_time,
            annual_demand=annual_demand,
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
        )
        
        recommendations = inventory_service.generate_recommendations(
            demand_data=demand_data,
            params=params,
            forecast_data=forecast_data,
        )
        
        report = inventory_service.generate_report(recommendations)
        
        return InventoryReportResponse(
            report=report,
            recommendations=recommendations.to_dict(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Report generation failed: {str(e)}"})


@router.post(
    "/safety-stock",
    summary="Calculate safety stock",
)
async def calculate_safety_stock(
    demand_std: float = Form(..., gt=0),
    lead_time: int = Form(default=7, ge=1),
    service_level: float = Form(default=0.95, gt=0.0, lt=1.0),
):
    """
    Calculate safety stock from demand standard deviation.
    """
    from scipy import stats as scipy_stats
    from src.inventory.optimization import InventoryOptimization
    
    try:
        optimizer = InventoryOptimization(service_level=service_level)
        ss = optimizer.calculate_safety_stock(demand_std, lead_time)
        
        return {
            "safety_stock": ss,
            "z_score": optimizer.z_score,
            "lead_time": lead_time,
            "demand_std": demand_std,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Calculation failed: {str(e)}"})


@router.post(
    "/reorder-point",
    summary="Calculate reorder point",
)
async def calculate_reorder_point(
    avg_demand: float = Form(..., ge=0),
    lead_time: int = Form(default=7, ge=1),
    safety_stock: float = Form(..., ge=0),
):
    """
    Calculate reorder point from average demand, lead time, and safety stock.
    """
    from src.inventory.optimization import InventoryOptimization
    
    try:
        optimizer = InventoryOptimization(service_level=0.95)
        rop = optimizer.calculate_reorder_point(avg_demand, lead_time, safety_stock)
        
        return {
            "reorder_point": rop,
            "avg_demand": avg_demand,
            "lead_time": lead_time,
            "safety_stock": safety_stock,
            "demand_during_lead_time": avg_demand * lead_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Calculation failed: {str(e)}"})


@router.post(
    "/eoq",
    summary="Calculate Economic Order Quantity",
)
async def calculate_eoq(
    annual_demand: float = Form(..., gt=0),
    holding_cost: float = Form(..., gt=0),
    ordering_cost: float = Form(..., gt=0),
):
    """
    Calculate Economic Order Quantity (EOQ).
    """
    from src.inventory.optimization import InventoryOptimization
    
    try:
        optimizer = InventoryOptimization(service_level=0.95)
        eoq = optimizer.calculate_economic_order_quantity(annual_demand, holding_cost, ordering_cost)
        
        if eoq is None:
            raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": "Invalid input parameters for EOQ"})
        
        return {
            "economic_order_quantity": eoq,
            "annual_demand": annual_demand,
            "holding_cost": holding_cost,
            "ordering_cost": ordering_cost,
            "formula": "EOQ = sqrt(2 * D * S / H)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": f"Calculation failed: {str(e)}"})


@router.get(
    "/params",
    summary="Get default inventory parameters",
)
async def get_default_params():
    """
    Get default inventory calculation parameters.
    """
    return {
        "service_level": 0.95,
        "lead_time": 7,
        "current_stock": 0.0,
        "annual_demand": None,
        "holding_cost": None,
        "ordering_cost": None,
    }