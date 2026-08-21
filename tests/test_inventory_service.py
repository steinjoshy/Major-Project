"""Tests for InventoryService."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.services.inventory_service import (
    InventoryService,
    InventoryParams,
    InventoryRecommendations,
    InventoryProjection,
)


class TestInventoryService:
    """Test InventoryService functionality."""
    
    @pytest.fixture
    def service(self):
        """Create InventoryService instance."""
        return InventoryService(service_level=0.95)
    
    @pytest.fixture
    def demand_data(self):
        """Sample demand data."""
        np.random.seed(42)
        return np.random.poisson(50, 200)
    
    @pytest.fixture
    def forecast_data(self):
        """Sample forecast data."""
        np.random.seed(42)
        return np.random.poisson(50, 30).astype(float)
    
    # ------------------------------------------------------------------
    # Core Calculations
    # ------------------------------------------------------------------
    
    def test_calculate_safety_stock(self, service):
        """Test safety stock calculation."""
        ss = service.calculate_safety_stock(demand_std=10.0, lead_time=7)
        
        expected = 1.645 * 10.0 * np.sqrt(7)
        assert ss == pytest.approx(expected, rel=0.001)
        assert ss > 0
    
    def test_calculate_safety_stock_zero_std(self, service):
        """Test safety stock with zero std."""
        ss = service.calculate_safety_stock(demand_std=0.0, lead_time=7)
        assert ss == 0.0
    
    def test_calculate_forecast_based_safety_stock(self, service, forecast_data):
        """Test forecast-based safety stock."""
        ss = service.calculate_forecast_based_safety_stock(forecast_data, lead_time=7)
        
        expected = 1.645 * np.std(forecast_data, ddof=1) * np.sqrt(7)
        assert ss == pytest.approx(expected, rel=0.001)
    
    def test_calculate_reorder_point(self, service):
        """Test reorder point calculation."""
        rop = service.calculate_reorder_point(avg_demand=50.0, lead_time=7, safety_stock=20.0)
        
        assert rop == 50.0 * 7 + 20.0  # 370.0
    
    def test_calculate_economic_order_quantity(self, service):
        """Test EOQ calculation."""
        eoq = service.calculate_economic_order_quantity(
            annual_demand=10000, holding_cost=2.0, ordering_cost=50.0
        )
        
        expected = np.sqrt(2 * 10000 * 50 / 2)
        assert eoq == pytest.approx(expected, rel=0.001)
    
    def test_eoq_invalid_inputs(self, service):
        """Test EOQ with invalid inputs returns None."""
        assert service.calculate_economic_order_quantity(0, 2, 50) is None
        assert service.calculate_economic_order_quantity(10000, 0, 50) is None
        assert service.calculate_economic_order_quantity(10000, 2, 0) is None
        assert service.calculate_economic_order_quantity(10000, -1, 50) is None
    
    def test_calculate_demand_statistics(self, service, demand_data):
        """Test demand statistics calculation."""
        stats = service.calculate_demand_statistics(demand_data)
        
        assert 'mean_demand' in stats
        assert 'std_demand' in stats
        assert 'min_demand' in stats
        assert 'max_demand' in stats
        assert 'cv' in stats
        assert 'n_periods' in stats
        
        assert stats['n_periods'] == 200
        assert stats['mean_demand'] > 0
        assert stats['std_demand'] >= 0
        assert stats['cv'] >= 0
    
    def test_calculate_lead_time_demand(self, service):
        """Test lead time demand calculation."""
        ltd = service.calculate_lead_time_demand(avg_demand=50.0, lead_time=7)
        assert ltd == 350.0
    
    # ------------------------------------------------------------------
    # High-Level Recommendations
    # ------------------------------------------------------------------
    
    def test_generate_recommendations_historical(self, service, demand_data):
        """Test recommendations with historical data only."""
        recs = service.generate_recommendations_simple(
            demand_data=demand_data,
            lead_time=7,
            service_level=0.95,
        )
        
        assert isinstance(recs, InventoryRecommendations)
        assert recs.safety_stock > 0
        assert recs.reorder_point > recs.safety_stock
        assert recs.safety_stock_basis == 'historical demand variability'
        assert recs.average_daily_demand > 0
    
    def test_generate_recommendations_forecast_based(self, service, demand_data, forecast_data):
        """Test recommendations with forecast data."""
        recs = service.generate_recommendations_simple(
            demand_data=demand_data,
            lead_time=7,
            service_level=0.95,
            forecast_data=forecast_data,
        )
        
        assert recs.safety_stock_basis == 'forecast variability'
        # Forecast-based SS should use forecast std
        forecast_ss = 1.645 * np.std(forecast_data, ddof=1) * np.sqrt(7)
        assert recs.safety_stock == pytest.approx(forecast_ss, rel=0.001)
    
    def test_generate_recommendations_with_eoq(self, service, demand_data):
        """Test recommendations with EOQ parameters."""
        recs = service.generate_recommendations_simple(
            demand_data=demand_data,
            lead_time=7,
            service_level=0.95,
            annual_demand=18250,
            holding_cost=2.0,
            ordering_cost=50.0,
        )
        
        assert recs.economic_order_quantity is not None
        expected = np.sqrt(2 * 18250 * 50 / 2)
        assert recs.economic_order_quantity == pytest.approx(expected, rel=0.001)
    
    def test_generate_recommendations_with_params_object(self, service, demand_data):
        """Test recommendations using InventoryParams object."""
        params = InventoryParams(
            service_level=0.95,
            lead_time=7,
            annual_demand=18250,
            holding_cost=2.0,
            ordering_cost=50.0,
        )
        
        recs = service.generate_recommendations(demand_data, params)
        
        assert isinstance(recs, InventoryRecommendations)
        assert recs.economic_order_quantity is not None
    
    def test_generate_report(self, service, demand_data):
        """Test report generation."""
        recs = service.generate_recommendations_simple(demand_data, lead_time=7)
        report = service.generate_report(recs)
        
        assert 'INVENTORY OPTIMIZATION REPORT' in report
        assert 'Safety Stock' in report
        assert 'Reorder Point' in report
        assert str(recs.safety_stock) in report
        assert str(recs.reorder_point) in report
    
    # ------------------------------------------------------------------
    # Inventory Projection
    # ------------------------------------------------------------------
    
    def test_project_inventory(self, service, forecast_data):
        """Test inventory projection."""
        projection = service.project_inventory_simple(
            forecast_demand=forecast_data,
            current_stock=1000,
            lead_time=7,
        )
        
        assert isinstance(projection, InventoryProjection)
        assert len(projection.periods) == len(forecast_data)
        assert len(projection.inventory_levels) == len(forecast_data)
        assert (projection.inventory_levels >= 0).all()
        assert projection.reorder_point > 0
    
    def test_project_inventory_with_params(self, service, forecast_data):
        """Test inventory projection with explicit parameters."""
        recs = service.generate_recommendations_simple(
            demand_data=np.array([50]*100),
            lead_time=7,
            service_level=0.95,
        )
        
        projection = service.project_inventory(
            forecast_demand=forecast_data,
            current_stock=1000,
            reorder_point=recs.reorder_point,
            lead_time=7,
            safety_stock=recs.safety_stock,
        )
        
        assert len(projection.periods) == len(forecast_data)
        assert (projection.inventory_levels >= 0).all()
    
    def test_projection_to_dataframe(self, service, forecast_data):
        """Test projection DataFrame conversion."""
        projection = service.project_inventory_simple(forecast_data, 1000, 7)
        df = projection.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Period', 'Forecast_Demand', 'Inventory_Level', 'Order_Placed']
    
    def test_projection_stockout_days(self, service):
        """Test stockout day counting."""
        # Low stock scenario
        forecast = np.array([100, 100, 100, 100])
        projection = service.project_inventory_simple(forecast, current_stock=50, lead_time=1)
        
        assert projection.get_stockout_days() > 0
    
    def test_projection_below_safety_stock_days(self, service, forecast_data):
        """Test days below safety stock counting."""
        projection = service.project_inventory_simple(forecast_data, current_stock=1000, lead_time=7)
        
        below_ss = projection.get_below_safety_stock_days()
        assert below_ss >= 0
        assert below_ss <= len(forecast_data)
    
    def test_projection_total_orders(self, service, forecast_data):
        """Test total orders counting."""
        projection = service.project_inventory_simple(forecast_data, current_stock=100, lead_time=7)
        
        total_orders = projection.get_total_orders_placed()
        assert total_orders >= 0
        assert total_orders == int(np.sum(projection.orders_placed))
    
    # ------------------------------------------------------------------
    # Risk Analysis
    # ------------------------------------------------------------------
    
    def test_calculate_stockout_risk(self, service, demand_data):
        """Test stockout risk calculation."""
        risk = service.calculate_stockout_risk(
            demand_data=demand_data,
            current_stock=100,
            lead_time=7,
            service_level=0.95,
        )
        
        assert 0.0 <= risk <= 1.0
    
    def test_stockout_risk_zero_std(self, service):
        """Test stockout risk with zero demand variance."""
        demand = np.array([50] * 100)
        risk = service.calculate_stockout_risk(demand, current_stock=400, lead_time=7)
        assert risk == 0.0  # Stock covers exact demand
        
        risk = service.calculate_stockout_risk(demand, current_stock=300, lead_time=7)
        assert risk == 1.0  # Stock below demand
    
    def test_calculate_overstock_risk(self, service, demand_data):
        """Test overstock risk calculation."""
        risk = service.calculate_overstock_risk(
            demand_data=demand_data,
            current_stock=5000,
            holding_cost=0.1,
            time_horizon=30,
        )
        
        assert risk >= 0.0
    
    def test_overstock_risk_no_excess(self, service, demand_data):
        """Test overstock risk when stock <= expected demand."""
        stats = service.calculate_demand_statistics(demand_data)
        expected = stats['mean_demand'] * 30
        
        risk = service.calculate_overstock_risk(
            demand_data=demand_data,
            current_stock=expected,
            holding_cost=0.1,
            time_horizon=30,
        )
        
        assert risk == 0.0
    
    # ------------------------------------------------------------------
    # Recommendations Serialization
    # ------------------------------------------------------------------
    
    def test_recommendations_to_dict(self, service, demand_data):
        """Test recommendations serialization."""
        recs = service.generate_recommendations_simple(demand_data, lead_time=7)
        recs_dict = recs.to_dict()
        
        assert isinstance(recs_dict, dict)
        assert 'safety_stock' in recs_dict
        assert 'reorder_point' in recs_dict
        assert recs_dict['safety_stock'] == recs.safety_stock
    
    def test_projection_to_dict(self, service, forecast_data):
        """Test projection DataFrame conversion."""
        projection = service.project_inventory_simple(forecast_data, 1000, 7)
        df = projection.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(forecast_data)
    
    # ------------------------------------------------------------------
    # Edge Cases
    # ------------------------------------------------------------------
    
    def test_invalid_service_level(self):
        """Test invalid service level raises error."""
        with pytest.raises(ValueError):
            InventoryService(service_level=0.0)
        with pytest.raises(ValueError):
            InventoryService(service_level=1.0)
        with pytest.raises(ValueError):
            InventoryService(service_level=1.5)
    
    def test_demand_statistics_filters_negative(self, service):
        """Test negative demand values are filtered."""
        demand = np.array([10, -5, 20, np.nan, 30, -1, 40])
        stats = service.calculate_demand_statistics(demand)
        
        # Only [10, 20, 30, 40] used
        assert stats['n_periods'] == 4
        assert stats['mean_demand'] == 25.0
    
    def test_constant_demand(self, service):
        """Test with constant demand."""
        demand = np.array([50] * 100)
        stats = service.calculate_demand_statistics(demand)
        
        assert stats['std_demand'] == 0.0
        assert stats['cv'] == 0.0
    
    def test_last_recommendations_storage(self, service, demand_data):
        """Test last recommendations are stored."""
        assert service.get_last_recommendations() is None
        
        service.generate_recommendations_simple(demand_data)
        
        last = service.get_last_recommendations()
        assert last is not None
        assert isinstance(last, InventoryRecommendations)
    
    def test_last_projection_storage(self, service, forecast_data):
        """Test last projection is stored."""
        assert service.get_last_projection() is None
        
        service.project_inventory_simple(forecast_data, 1000, 7)
        
        last = service.get_last_projection()
        assert last is not None
        assert isinstance(last, InventoryProjection)
    
    def test_inventory_params_dataclass(self):
        """Test InventoryParams dataclass."""
        params = InventoryParams(
            service_level=0.95,
            lead_time=7,
            current_stock=1000,
            annual_demand=18250,
            holding_cost=2.0,
            ordering_cost=50.0,
        )
        
        assert params.service_level == 0.95
        assert params.lead_time == 7
        assert params.current_stock == 1000
    
    def test_inventory_recommendations_dataclass(self, service, demand_data):
        """Test InventoryRecommendations dataclass."""
        recs = service.generate_recommendations_simple(demand_data)
        
        assert isinstance(recs, InventoryRecommendations)
        assert hasattr(recs, 'safety_stock')
        assert hasattr(recs, 'reorder_point')
        assert hasattr(recs, 'safety_stock_basis')
    
    def test_inventory_projection_dataclass(self, service, forecast_data):
        """Test InventoryProjection dataclass."""
        projection = service.project_inventory_simple(forecast_data, 1000, 7)
        
        assert isinstance(projection, InventoryProjection)
        assert hasattr(projection, 'periods')
        assert hasattr(projection, 'forecast_demand')
        assert hasattr(projection, 'inventory_levels')
        assert hasattr(projection, 'orders_placed')