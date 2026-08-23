"""Tests for model comparison metrics and inventory optimization."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.inventory.optimization import InventoryOptimization
from src.models.model_comparison import ModelComparison


class TestModelComparisonMetrics:
    """Test RMSE, MAE, MAPE calculations."""

    def test_rmse_basic(self):
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 33, 37, 52])

        rmse = ModelComparison.rmse(y_true, y_pred)

        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(rmse - expected) < 1e-10

    def test_rmse_perfect(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])

        assert ModelComparison.rmse(y_true, y_pred) == 0.0

    def test_mae_basic(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])

        mae = ModelComparison.mae(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        assert abs(mae - expected) < 1e-10

    def test_mape_basic(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 330])

        mape = ModelComparison.mape(y_true, y_pred)
        expected = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        assert abs(mape - expected) < 1e-6

    def test_mape_zero_handling(self):
        """MAPE should handle zeros in y_true with epsilon."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 190])

        mape = ModelComparison.mape(y_true, y_pred)
        # Should not raise, epsilon prevents div/0
        assert np.isfinite(mape)

    def test_mape_negative_values(self):
        """MAPE uses absolute y_true in denominator."""
        y_true = np.array([-100, -200, -300])
        y_pred = np.array([-110, -190, -330])

        mape = ModelComparison.mape(y_true, y_pred)
        expected = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
        assert abs(mape - expected) < 1e-6


class TestModelComparisonEvaluation:
    """Test model evaluation and comparison."""

    def test_evaluate_model(self):
        comparator = ModelComparison()
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 33, 37, 52])

        metrics = comparator.evaluate_model(y_true, y_pred, "TestModel")

        assert metrics['Model'] == 'TestModel'
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'MAPE (%)' in metrics
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))

    def test_compare_models_sorts_by_rmse(self):
        comparator = ModelComparison()
        y_true = np.array([10, 20, 30, 40, 50])

        preds = {
            'ModelA': np.array([12, 18, 33, 37, 52]),  # RMSE ≈ 2.45
            'ModelB': np.array([10, 20, 30, 40, 50]),  # Perfect, RMSE = 0
            'ModelC': np.array([15, 25, 25, 45, 55]),  # RMSE = 5.0
        }

        df = comparator.compare_models(y_true, preds)

        assert len(df) == 3
        # Sorted by RMSE ascending: ModelB (0), ModelA (~2.45), ModelC (5.0)
        assert list(df['Model']) == ['ModelB', 'ModelA', 'ModelC']
        assert df.iloc[0]['RMSE'] == 0.0

    def test_get_best_model(self):
        comparator = ModelComparison()
        y_true = np.array([10, 20, 30])
        preds = {
            'A': np.array([10, 20, 30]),
            'B': np.array([15, 25, 35]),
        }
        df = comparator.compare_models(y_true, preds)
        best = comparator.get_best_model(df, 'RMSE')

        assert best['best_model'] == 'A'
        assert best['metrics']['RMSE'] == 0.0
        assert best['improvement_pct'] == 100.0

    def test_get_best_model_different_metric(self):
        comparator = ModelComparison()
        y_true = np.array([10, 20, 30])
        preds = {
            'A': np.array([10, 20, 30]),
            'B': np.array([15, 25, 35]),
        }
        df = comparator.compare_models(y_true, preds)
        best = comparator.get_best_model(df, 'MAE')

        assert best['best_model'] == 'A'

    def test_generate_comparison_summary(self):
        comparator = ModelComparison()
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}

        result = comparator.generate_comparison_summary(y_true, preds)

        assert 'comparison_table' in result
        assert 'best_model_info' in result
        assert result['best_model_info']['best_model'] == 'A'

    def test_export_results(self, tmp_path):
        comparator = ModelComparison()
        df = pd.DataFrame({
            'Model': ['A', 'B'],
            'MAE': [1.0, 2.0],
            'RMSE': [1.5, 2.5],
            'MAPE (%)': [5.0, 10.0]
        })

        filepath = tmp_path / "comparison.csv"
        comparator.export_results(str(filepath), df)

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 2


class TestInventoryOptimization:
    """Test Safety Stock, ROP, EOQ, and projection calculations."""

    def test_init_service_level(self):
        inv = InventoryOptimization(service_level=0.95)
        assert inv.service_level == 0.95
        assert inv.z_score == pytest.approx(1.645, rel=0.01)

    def test_init_invalid_service_level(self):
        with pytest.raises(ValueError):
            InventoryOptimization(service_level=0.0)
        with pytest.raises(ValueError):
            InventoryOptimization(service_level=1.0)
        with pytest.raises(ValueError):
            InventoryOptimization(service_level=1.5)

    def test_calculate_demand_statistics(self):
        inv = InventoryOptimization(service_level=0.95)
        demand = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        stats = inv.calculate_demand_statistics(demand)

        assert stats['mean_demand'] == 55.0
        assert stats['std_demand'] == pytest.approx(30.2765, rel=0.01)  # ddof=1
        assert stats['min_demand'] == 10.0
        assert stats['max_demand'] == 100.0
        assert stats['cv'] == pytest.approx(30.2765/55, rel=0.01)
        assert stats['n_periods'] == 10

    def test_calculate_demand_statistics_filters_negative(self):
        inv = InventoryOptimization(service_level=0.95)
        demand = np.array([10, -5, 20, np.nan, 30, -1, 40])

        stats = inv.calculate_demand_statistics(demand)

        # Only positive, finite values used: [10, 20, 30, 40]
        assert stats['n_periods'] == 4
        assert stats['mean_demand'] == 25.0

    def test_calculate_safety_stock_historical(self):
        inv = InventoryOptimization(service_level=0.95)
        ss = inv.calculate_safety_stock(demand_std=10.0, lead_time=7)

        expected = 1.645 * 10.0 * np.sqrt(7)
        assert ss == pytest.approx(expected, rel=0.001)

    def test_calculate_safety_stock_zero_std(self):
        inv = InventoryOptimization(service_level=0.95)
        ss = inv.calculate_safety_stock(demand_std=0.0, lead_time=7)
        assert ss == 0.0

    def test_calculate_reorder_point(self):
        inv = InventoryOptimization(service_level=0.95)
        rop = inv.calculate_reorder_point(avg_demand=50.0, lead_time=7, safety_stock=20.0)

        assert rop == 50.0 * 7 + 20.0  # 370.0

    def test_calculate_economic_order_quantity(self):
        inv = InventoryOptimization(service_level=0.95)
        eoq = inv.calculate_economic_order_quantity(
            annual_demand=10000, holding_cost=2.0, ordering_cost=50.0
        )

        expected = np.sqrt(2 * 10000 * 50 / 2)
        assert eoq == pytest.approx(expected, rel=0.001)

    def test_eoq_invalid_inputs(self):
        inv = InventoryOptimization(service_level=0.95)
        assert inv.calculate_economic_order_quantity(0, 2, 50) is None
        assert inv.calculate_economic_order_quantity(10000, 0, 50) is None
        assert inv.calculate_economic_order_quantity(10000, 2, 0) is None
        assert inv.calculate_economic_order_quantity(10000, -1, 50) is None

    def test_calculate_forecast_based_safety_stock(self):
        inv = InventoryOptimization(service_level=0.95)
        forecast = np.array([45, 55, 50, 60, 40, 55, 50])  # std ≈ 7

        ss = inv.calculate_forecast_based_safety_stock(forecast, lead_time=7)

        expected = 1.645 * np.std(forecast, ddof=1) * np.sqrt(7)
        assert ss == pytest.approx(expected, rel=0.001)

    def test_generate_inventory_recommendations_historical(self):
        inv = InventoryOptimization(service_level=0.95)
        demand = np.array([50] * 100 + [60] * 100)  # mean=55, std≈5

        recs = inv.generate_inventory_recommendations(
            demand_data=demand, lead_time=7
        )

        assert 'safety_stock' in recs
        assert 'reorder_point' in recs
        assert 'average_daily_demand' in recs
        assert 'demand_std_dev' in recs
        assert 'lead_time_days' in recs
        assert 'service_level_pct' in recs
        assert 'z_score' in recs
        assert recs['safety_stock_basis'] == 'historical demand variability'

    def test_generate_inventory_recommendations_forecast_based(self):
        inv = InventoryOptimization(service_level=0.95)
        demand = np.array([50] * 100)
        forecast = np.array([40, 60, 55, 45, 65, 50, 55])  # Higher variability

        recs = inv.generate_inventory_recommendations(
            demand_data=demand, lead_time=7, forecast_data=forecast
        )

        assert recs['safety_stock_basis'] == 'forecast variability'
        # Forecast-based SS should use forecast std
        forecast_ss = inv.calculate_forecast_based_safety_stock(forecast, 7)
        assert recs['safety_stock'] == pytest.approx(forecast_ss, rel=0.001)

    def test_generate_inventory_recommendations_with_eoq(self):
        inv = InventoryOptimization(service_level=0.95)
        demand = np.array([50] * 365)  # Annual = 18250

        recs = inv.generate_inventory_recommendations(
            demand_data=demand,
            lead_time=7,
            annual_demand=18250,
            holding_cost=2.0,
            ordering_cost=50.0
        )

        assert 'economic_order_quantity' in recs
        expected_eoq = np.sqrt(2 * 18250 * 50 / 2)
        assert recs['economic_order_quantity'] == pytest.approx(expected_eoq, rel=0.001)

    def test_forecast_inventory_levels(self):
        inv = InventoryOptimization(service_level=0.95)
        forecast = np.array([50, 55, 60, 55, 50, 45, 40])
        current_stock = 200
        lead_time = 3

        # Calculate SS and ROP first
        demand_hist = np.array([50] * 100)
        recs = inv.generate_inventory_recommendations(demand_hist, lead_time=lead_time)
        ss = recs['safety_stock']
        rop = recs['reorder_point']

        proj_df = inv.forecast_inventory_levels(
            current_stock=current_stock,
            forecast_demand=forecast,
            reorder_point=rop,
            lead_time=lead_time,
            safety_stock=ss
        )

        assert isinstance(proj_df, pd.DataFrame)
        assert len(proj_df) == len(forecast)
        assert list(proj_df.columns) == ['Period', 'Forecast_Demand', 'Inventory_Level', 'Order_Placed']
        assert proj_df['Inventory_Level'].min() >= 0  # Floored at 0
        assert proj_df['Inventory_Level'].iloc[0] <= current_stock

    def test_forecast_inventory_levels_order_triggers(self):
        inv = InventoryOptimization(service_level=0.95)
        # Low initial stock, high demand -> should trigger orders
        forecast = np.array([100, 100, 100, 100, 100])
        current_stock = 50
        lead_time = 2

        demand_hist = np.array([80] * 50)
        recs = inv.generate_inventory_recommendations(demand_hist, lead_time=lead_time)

        proj_df = inv.forecast_inventory_levels(
            current_stock=current_stock,
            forecast_demand=forecast,
            reorder_point=recs['reorder_point'],
            lead_time=lead_time,
            safety_stock=recs['safety_stock']
        )

        # Should have placed at least one order
        assert proj_df['Order_Placed'].any()

    def test_generate_optimization_report(self):
        inv = InventoryOptimization(service_level=0.95)
        demand = np.array([50] * 100)
        recs = inv.generate_inventory_recommendations(demand, lead_time=7)

        report = inv.generate_optimization_report(recs)

        assert 'INVENTORY OPTIMIZATION REPORT' in report
        assert 'Safety Stock' in report
        assert 'Reorder Point' in report
        assert str(recs['safety_stock']) in report
        assert str(recs['reorder_point']) in report
        assert 'Z × σ_demand × √(Lead Time)' in report
        assert 'Avg Demand × Lead Time' in report
