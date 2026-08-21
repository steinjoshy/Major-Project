"""
Inventory Service for AI Demand Forecasting.

Provides inventory optimization calculations: Safety Stock, ROP, EOQ, projections.
Pure Python service with no Streamlit dependencies.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats as scipy_stats

from src.inventory.optimization import InventoryOptimization


@dataclass
class InventoryParams:
    """Container for inventory calculation parameters."""
    service_level: float = 0.95
    lead_time: int = 7
    current_stock: float = 0.0
    annual_demand: Optional[float] = None
    holding_cost: Optional[float] = None
    ordering_cost: Optional[float] = None


@dataclass
class InventoryRecommendations:
    """Container for inventory optimization results."""
    safety_stock: float
    reorder_point: float
    average_daily_demand: float
    demand_std_dev: float
    lead_time_days: int
    service_level_pct: float
    z_score: float
    safety_stock_basis: str
    demand_during_lead_time: float
    economic_order_quantity: Optional[float] = None
    stockout_risk: Optional[float] = None
    overstock_risk: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'safety_stock': self.safety_stock,
            'reorder_point': self.reorder_point,
            'average_daily_demand': self.average_daily_demand,
            'demand_std_dev': self.demand_std_dev,
            'lead_time_days': self.lead_time_days,
            'service_level_pct': self.service_level_pct,
            'z_score': self.z_score,
            'safety_stock_basis': self.safety_stock_basis,
            'demand_during_lead_time': self.demand_during_lead_time,
            'economic_order_quantity': self.economic_order_quantity,
            'stockout_risk': self.stockout_risk,
            'overstock_risk': self.overstock_risk,
        }


@dataclass
class InventoryProjection:
    """Container for inventory projection results."""
    periods: np.ndarray
    forecast_demand: np.ndarray
    inventory_levels: np.ndarray
    orders_placed: np.ndarray
    reorder_point: float
    safety_stock: float
    lead_time: int
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'Period': self.periods,
            'Forecast_Demand': np.round(self.forecast_demand, 1),
            'Inventory_Level': np.round(self.inventory_levels, 1),
            'Order_Placed': self.orders_placed,
        })
    
    def get_stockout_days(self) -> int:
        """Count days with zero inventory."""
        return int(np.sum(self.inventory_levels <= 0))
    
    def get_below_safety_stock_days(self) -> int:
        """Count days below safety stock."""
        return int(np.sum(self.inventory_levels <= self.safety_stock))
    
    def get_total_orders_placed(self) -> int:
        """Count total orders placed."""
        return int(np.sum(self.orders_placed))


class InventoryService:
    """
    Service for inventory optimization calculations.
    
    Supports Safety Stock, Reorder Point, EOQ, inventory projections,
    and risk analysis. No Streamlit dependencies.
    """
    
    def __init__(self, service_level: float = 0.95):
        """
        Initialize the inventory service.
        
        Args:
            service_level: Target service level (0-1)
        """
        if not 0 < service_level < 1:
            raise ValueError("service_level must be between 0 and 1")
        self.optimizer = InventoryOptimization(service_level=service_level)
        self._last_recommendations: Optional[InventoryRecommendations] = None
        self._last_projection: Optional[InventoryProjection] = None
    
    # ------------------------------------------------------------------
    # Core Calculations
    # ------------------------------------------------------------------
    
    def calculate_safety_stock(
        self,
        demand_std: float,
        lead_time: int = 1,
    ) -> float:
        """
        Calculate safety stock using historical demand std.
        
        Args:
            demand_std: Standard deviation of daily demand
            lead_time: Lead time in days
            
        Returns:
            Safety stock quantity
        """
        return self.optimizer.calculate_safety_stock(demand_std, lead_time)
    
    def calculate_forecast_based_safety_stock(
        self,
        forecast_array: Union[np.ndarray, List[float]],
        lead_time: int = 1,
    ) -> float:
        """
        Calculate safety stock using forecast variability.
        
        Args:
            forecast_array: Array of forecasted demand values
            lead_time: Lead time in days
            
        Returns:
            Safety stock quantity
        """
        return self.optimizer.calculate_forecast_based_safety_stock(
            np.asarray(forecast_array), lead_time
        )
    
    def calculate_reorder_point(
        self,
        avg_demand: float,
        lead_time: int,
        safety_stock: float,
    ) -> float:
        """
        Calculate reorder point.
        
        Args:
            avg_demand: Average daily demand
            lead_time: Lead time in days
            safety_stock: Safety stock quantity
            
        Returns:
            Reorder point
        """
        return self.optimizer.calculate_reorder_point(avg_demand, lead_time, safety_stock)
    
    def calculate_economic_order_quantity(
        self,
        annual_demand: float,
        holding_cost: float,
        ordering_cost: float,
    ) -> Optional[float]:
        """
        Calculate Economic Order Quantity.
        
        Args:
            annual_demand: Annual demand in units
            holding_cost: Cost to hold 1 unit for 1 year
            ordering_cost: Fixed cost per order
            
        Returns:
            EOQ or None if inputs invalid
        """
        return self.optimizer.calculate_economic_order_quantity(
            annual_demand, holding_cost, ordering_cost
        )
    
    def calculate_demand_statistics(
        self,
        demand_data: Union[np.ndarray, List[float]],
    ) -> Dict[str, float]:
        """
        Calculate demand descriptive statistics.
        
        Args:
            demand_data: Historical demand array
            
        Returns:
            Dictionary with mean, std, min, max, CV, n_periods
        """
        return self.optimizer.calculate_demand_statistics(demand_data)
    
    def calculate_lead_time_demand(
        self,
        avg_demand: float,
        lead_time: int,
    ) -> float:
        """
        Calculate demand during lead time.
        
        Args:
            avg_demand: Average daily demand
            lead_time: Lead time in days
            
        Returns:
            Lead time demand
        """
        return avg_demand * lead_time
    
    # ------------------------------------------------------------------
    # High-Level Recommendations
    # ------------------------------------------------------------------
    
    def generate_recommendations(
        self,
        demand_data: Union[np.ndarray, List[float]],
        params: Optional[InventoryParams] = None,
        forecast_data: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> InventoryRecommendations:
        """
        Generate comprehensive inventory recommendations.
        
        Args:
            demand_data: Historical demand array
            params: Inventory parameters (service_level, lead_time, etc.)
            forecast_data: Optional forecast array for forecast-based SS
            
        Returns:
            InventoryRecommendations object
        """
        if params is None:
            params = InventoryParams()
        
        self.optimizer.service_level = params.service_level
        self.optimizer.z_score = float(scipy_stats.norm.ppf(params.service_level))
        
        # Generate recommendations using the optimizer
        recs = self.optimizer.generate_inventory_recommendations(
            demand_data=np.asarray(demand_data, dtype=float),
            lead_time=params.lead_time,
            forecast_data=np.asarray(forecast_data, dtype=float) if forecast_data is not None else None,
            annual_demand=params.annual_demand,
            holding_cost=params.holding_cost,
            ordering_cost=params.ordering_cost,
        )
        
        self._last_recommendations = InventoryRecommendations(
            safety_stock=recs['safety_stock'],
            reorder_point=recs['reorder_point'],
            average_daily_demand=recs['average_daily_demand'],
            demand_std_dev=recs['demand_std_dev'],
            lead_time_days=recs['lead_time_days'],
            service_level_pct=recs['service_level_pct'],
            z_score=recs['z_score'],
            safety_stock_basis=recs['safety_stock_basis'],
            demand_during_lead_time=recs['demand_during_lead_time'],
            economic_order_quantity=recs.get('economic_order_quantity'),
        )
        
        return self._last_recommendations
    
    def generate_recommendations_simple(
        self,
        demand_data: Union[np.ndarray, List[float]],
        lead_time: int = 7,
        service_level: float = 0.95,
        forecast_data: Optional[Union[np.ndarray, List[float]]] = None,
        annual_demand: Optional[float] = None,
        holding_cost: Optional[float] = None,
        ordering_cost: Optional[float] = None,
    ) -> InventoryRecommendations:
        """
        Simplified interface for generating recommendations.
        
        Args:
            demand_data: Historical demand array
            lead_time: Lead time in days
            service_level: Service level (0-1)
            forecast_data: Optional forecast for forecast-based SS
            annual_demand: For EOQ
            holding_cost: For EOQ
            ordering_cost: For EOQ
            
        Returns:
            InventoryRecommendations
        """
        params = InventoryParams(
            service_level=service_level,
            lead_time=lead_time,
            annual_demand=annual_demand,
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
        )
        return self.generate_recommendations(demand_data, params, forecast_data)
    
    # ------------------------------------------------------------------
    # Inventory Projection
    # ------------------------------------------------------------------
    
    def project_inventory(
        self,
        forecast_demand: Union[np.ndarray, List[float]],
        current_stock: float,
        reorder_point: float,
        lead_time: int,
        safety_stock: Optional[float] = None,
    ) -> InventoryProjection:
        """
        Project inventory levels over forecast horizon.
        
        Args:
            forecast_demand: Array of forecasted daily demand
            current_stock: Starting inventory level
            reorder_point: Reorder point trigger
            lead_time: Lead time in days
            safety_stock: Safety stock (used for order sizing)
            
        Returns:
            InventoryProjection object
        """
        forecast_demand = np.asarray(forecast_demand, dtype=float)
        
        if safety_stock is None:
            safety_stock = self.calculate_forecast_based_safety_stock(
                forecast_demand, lead_time
            )
        
        proj_df = self.optimizer.forecast_inventory_levels(
            current_stock=current_stock,
            forecast_demand=forecast_demand,
            reorder_point=reorder_point,
            lead_time=lead_time,
            safety_stock=safety_stock,
        )
        
        self._last_projection = InventoryProjection(
            periods=proj_df['Period'].values,
            forecast_demand=proj_df['Forecast_Demand'].values,
            inventory_levels=proj_df['Inventory_Level'].values,
            orders_placed=proj_df['Order_Placed'].values,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            lead_time=lead_time,
        )
        
        return self._last_projection
    
    def project_inventory_simple(
        self,
        forecast_demand: Union[np.ndarray, List[float]],
        current_stock: float,
        lead_time: int = 7,
        reorder_point: Optional[float] = None,
        safety_stock: Optional[float] = None,
    ) -> InventoryProjection:
        """
        Simplified inventory projection.
        
        Args:
            forecast_demand: Forecasted daily demand
            current_stock: Starting inventory
            lead_time: Lead time
            reorder_point: If None, calculated from forecast
            safety_stock: If None, calculated from forecast
            
        Returns:
            InventoryProjection
        """
        forecast_demand = np.asarray(forecast_demand, dtype=float)
        
        if safety_stock is None:
            safety_stock = self.calculate_forecast_based_safety_stock(
                forecast_demand, lead_time
            )
        
        if reorder_point is None:
            avg_demand = float(np.mean(forecast_demand))
            reorder_point = self.calculate_reorder_point(avg_demand, lead_time, safety_stock)
        
        return self.project_inventory(
            forecast_demand, current_stock, reorder_point, lead_time, safety_stock
        )
    
    # ------------------------------------------------------------------
    # Risk Analysis
    # ------------------------------------------------------------------
    
    def calculate_stockout_risk(
        self,
        demand_data: Union[np.ndarray, List[float]],
        current_stock: float,
        lead_time: int,
        service_level: float = 0.95,
    ) -> float:
        """
        Calculate probability of stockout during lead time.
        
        Args:
            demand_data: Historical demand
            current_stock: Current inventory
            lead_time: Lead time
            service_level: Target service level
            
        Returns:
            Stockout probability (0-1)
        """
        optimizer = InventoryOptimization(service_level=service_level)
        stats = optimizer.calculate_demand_statistics(demand_data)
        
        # Demand during lead time ~ Normal(mean * LT, std * sqrt(LT))
        lt_mean = stats['mean_demand'] * lead_time
        lt_std = stats['std_demand'] * np.sqrt(lead_time)
        
        if lt_std == 0:
            return 0.0 if current_stock >= lt_mean else 1.0
        
        # P(Demand during LT > current_stock)
        z = (current_stock - lt_mean) / lt_std
        stockout_prob = 1 - scipy_stats.norm.cdf(z)
        
        return max(0.0, min(1.0, float(stockout_prob)))
    
    def calculate_overstock_risk(
        self,
        demand_data: Union[np.ndarray, List[float]],
        current_stock: float,
        holding_cost: float,
        time_horizon: int = 30,
    ) -> float:
        """
        Estimate overstock cost risk.
        
        Args:
            demand_data: Historical demand
            current_stock: Current inventory
            holding_cost: Cost per unit per day
            time_horizon: Days to evaluate
            
        Returns:
            Expected overstock cost
        """
        stats = self.calculate_demand_statistics(demand_data)
        expected_demand = stats['mean_demand'] * time_horizon
        
        if current_stock <= expected_demand:
            return 0.0
        
        excess = current_stock - expected_demand
        return excess * holding_cost * time_horizon
    
    def add_risk_metrics(self, recommendations: InventoryRecommendations) -> InventoryRecommendations:
        """Add stockout and overstock risk to recommendations."""
        # This would require demand data which isn't in the recommendations object
        # Kept for interface completeness
        return recommendations
    
    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    
    def generate_report(
        self,
        recommendations: Optional[InventoryRecommendations] = None,
    ) -> str:
        """
        Generate human-readable inventory report.
        
        Args:
            recommendations: Recommendations object (uses last if None)
            
        Returns:
            Formatted report string
        """
        recs = recommendations or self._last_recommendations
        if recs is None:
            return "No recommendations available."
        
        return self.optimizer.generate_optimization_report(recs.to_dict())
    
    def get_last_recommendations(self) -> Optional[InventoryRecommendations]:
        """Get last generated recommendations."""
        return self._last_recommendations
    
    def get_last_projection(self) -> Optional[InventoryProjection]:
        """Get last generated projection."""
        return self._last_projection