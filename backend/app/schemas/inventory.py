"""
Inventory optimization schemas.
"""
from pydantic import BaseModel, Field


class InventoryParams(BaseModel):
    """Inventory calculation parameters."""
    service_level: float = Field(default=0.95, gt=0.0, lt=1.0, description="Service level (0-1)")
    lead_time: int = Field(default=7, ge=1, le=365, description="Lead time in days")
    current_stock: float = Field(default=0.0, ge=0.0, description="Current stock level")
    annual_demand: float | None = Field(default=None, gt=0, description="Annual demand for EOQ")
    holding_cost: float | None = Field(default=None, gt=0, description="Holding cost per unit per year")
    ordering_cost: float | None = Field(default=None, gt=0, description="Ordering cost per order")


class InventoryRecommendationsResponse(BaseModel):
    """Inventory recommendations response."""
    safety_stock: float = Field(..., description="Safety stock quantity")
    reorder_point: float = Field(..., description="Reorder point")
    average_daily_demand: float = Field(..., description="Average daily demand")
    demand_std_dev: float = Field(..., description="Demand standard deviation")
    lead_time_days: int = Field(..., description="Lead time in days")
    service_level_pct: float = Field(..., description="Service level percentage")
    z_score: float = Field(..., description="Z-score for service level")
    safety_stock_basis: str = Field(..., description="Basis for safety stock calculation")
    demand_during_lead_time: float = Field(..., description="Expected demand during lead time")
    economic_order_quantity: float | None = Field(None, description="EOQ if parameters provided")
    stockout_risk: float | None = Field(None, description="Stockout probability (0-1)")
    overstock_risk: float | None = Field(None, description="Overstock cost risk")


class InventoryProjectionRequest(BaseModel):
    """Request for inventory projection."""
    forecast_demand: list[float] = Field(..., description="Forecasted daily demand")
    current_stock: float = Field(..., ge=0, description="Current stock level")
    lead_time: int = Field(default=7, ge=1, le=365, description="Lead time in days")
    reorder_point: float | None = Field(None, ge=0, description="Reorder point (optional)")
    safety_stock: float | None = Field(None, ge=0, description="Safety stock (optional)")


class InventoryProjectionPoint(BaseModel):
    """Single projection point."""
    period: int
    forecast_demand: float
    inventory_level: float
    order_placed: bool


class InventoryProjectionResponse(BaseModel):
    """Inventory projection response."""
    periods: list[int]
    forecast_demand: list[float]
    inventory_levels: list[float]
    orders_placed: list[bool]
    reorder_point: float
    safety_stock: float
    lead_time: int
    stockout_days: int
    below_safety_stock_days: int
    total_orders_placed: int


class RiskAnalysisRequest(BaseModel):
    """Request for risk analysis."""
    demand_data: list[float] = Field(..., description="Historical demand data")
    current_stock: float = Field(..., ge=0, description="Current stock level")
    lead_time: int = Field(default=7, ge=1, description="Lead time in days")
    service_level: float = Field(default=0.95, gt=0, lt=1, description="Service level")


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response."""
    stockout_risk: float = Field(..., description="Stockout probability (0-1)")
    overstock_risk: float | None = Field(None, description="Overstock cost risk")


class InventoryReportResponse(BaseModel):
    """Inventory report response."""
    report: str = Field(..., description="Formatted text report")
    recommendations: dict | None = None
