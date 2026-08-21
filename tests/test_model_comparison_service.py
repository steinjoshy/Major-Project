"""Tests for ModelComparisonService."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.services.model_comparison_service import (
    ModelComparisonService,
    ModelMetrics,
    ComparisonResult,
)


class TestModelComparisonService:
    """Test ModelComparisonService functionality."""
    
    @pytest.fixture
    def service(self):
        """Create ModelComparisonService instance."""
        return ModelComparisonService()
    
    # ------------------------------------------------------------------
    # Metric Calculations
    # ------------------------------------------------------------------
    
    def test_rmse_basic(self):
        """Test RMSE calculation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 33, 37, 52])
        
        rmse = ModelComparisonService.rmse(y_true, y_pred)
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert rmse == pytest.approx(expected, rel=1e-10)
    
    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        
        assert ModelComparisonService.rmse(y_true, y_pred) == 0.0
    
    def test_mae_basic(self):
        """Test MAE calculation."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        
        mae = ModelComparisonService.mae(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        assert mae == pytest.approx(expected, rel=1e-10)
    
    def test_mape_basic(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 330])
        
        mape = ModelComparisonService.mape(y_true, y_pred)
        expected = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        assert mape == pytest.approx(expected, rel=1e-6)
    
    def test_mape_zero_handling(self):
        """Test MAPE handles zeros with epsilon."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 190])
        
        mape = ModelComparisonService.mape(y_true, y_pred)
        assert np.isfinite(mape)
    
    def test_mape_negative_values(self):
        """Test MAPE with negative values."""
        y_true = np.array([-100, -200, -300])
        y_pred = np.array([-110, -190, -330])
        
        mape = ModelComparisonService.mape(y_true, y_pred)
        expected = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
        assert mape == pytest.approx(expected, rel=1e-6)
    
    # ------------------------------------------------------------------
    # Model Evaluation
    # ------------------------------------------------------------------
    
    def test_evaluate_model(self, service):
        """Test single model evaluation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 33, 37, 52])
        
        metrics = service.evaluate_model(y_true, y_pred, "TestModel")
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.model_name == 'TestModel'
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape >= 0
        assert metrics.n_samples == 5
        assert 'TestModel' in service.get_results()
    
    def test_evaluate_model_length_mismatch(self, service):
        """Test evaluation with length mismatch raises error."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18])
        
        with pytest.raises(ValueError, match="Length mismatch"):
            service.evaluate_model(y_true, y_pred, "TestModel")
    
    # ------------------------------------------------------------------
    # Model Comparison
    # ------------------------------------------------------------------
    
    def test_compare_models(self, service):
        """Test comparing multiple models."""
        y_true = np.array([10, 20, 30, 40, 50])
        
        preds = {
            'ModelA': np.array([12, 18, 33, 37, 52]),
            'ModelB': np.array([10, 20, 30, 40, 50]),  # Perfect
            'ModelC': np.array([15, 25, 25, 45, 55]),
        }
        
        result = service.compare_models(y_true, preds)
        
        assert isinstance(result, ComparisonResult)
        assert len(result.metrics_table) == 3
        # Should be sorted by RMSE: ModelB (0), ModelA, ModelC
        assert list(result.metrics_table['Model']) == ['ModelB', 'ModelA', 'ModelC']
        assert result.best_model == 'ModelB'
        assert result.best_metrics.rmse == 0.0
        assert result.improvement_pct == 100.0
    
    def test_compare_models_different_metric(self, service):
        """Test comparison with different sorting metric."""
        y_true = np.array([10, 20, 30, 40, 50])
        
        preds = {
            'ModelA': np.array([12, 18, 33, 37, 52]),
            'ModelB': np.array([10, 20, 30, 40, 50]),
            'ModelC': np.array([15, 25, 25, 45, 55]),
        }
        
        # Sort by MAE
        result = service.compare_models(y_true, preds, metric='MAE')
        assert list(result.metrics_table['Model']) == ['ModelB', 'ModelA', 'ModelC']
        
        # Sort by MAPE
        result = service.compare_models(y_true, preds, metric='MAPE')
        assert list(result.metrics_table['Model']) == ['ModelB', 'ModelA', 'ModelC']
    
    def test_compare_models_invalid_metric(self, service):
        """Test comparison with invalid metric defaults to RMSE."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}
        
        result = service.compare_models(y_true, preds, metric='INVALID')
        assert result.metric_used == 'RMSE'
    
    def test_compare_models_empty(self, service):
        """Test comparison with empty predictions raises error."""
        y_true = np.array([10, 20, 30])
        preds = {}
        
        with pytest.raises(ValueError, match="No models to compare"):
            service.compare_models(y_true, preds)
    
    def test_compare_models_length_mismatch(self, service):
        """Test comparison with mismatched lengths."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20])}
        
        with pytest.raises(ValueError, match="Length mismatch"):
            service.compare_models(y_true, preds)
    
    # ------------------------------------------------------------------
    # Best Model
    # ------------------------------------------------------------------
    
    def test_get_best_model(self, service):
        """Test getting best model from last comparison."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}
        
        service.compare_models(y_true, preds)
        best = service.get_best_model('RMSE')
        
        assert best['best_model'] == 'A'
        assert best['metrics']['RMSE'] == 0.0
        assert best['improvement_pct'] == 100.0
    
    def test_get_best_model_no_comparison(self, service):
        """Test getting best model without comparison raises error."""
        with pytest.raises(ValueError, match="No comparison performed"):
            service.get_best_model()
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    
    def test_generate_summary(self, service):
        """Test full comparison summary."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}
        
        summary = service.generate_summary(y_true, preds)
        
        assert 'comparison_table' in summary
        assert 'best_model_info' in summary
        assert summary['best_model_info']['best_model'] == 'A'
    
    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    
    def test_export_results(self, service, tmp_path):
        """Test exporting results to CSV."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}
        
        service.compare_models(y_true, preds)
        
        filepath = tmp_path / "comparison.csv"
        service.export_results(str(filepath))
        
        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 2
    
    def test_export_results_custom_df(self, service, tmp_path):
        """Test exporting custom DataFrame."""
        df = pd.DataFrame({
            'Model': ['A', 'B'],
            'MAE': [1.0, 2.0],
            'RMSE': [1.5, 2.5],
            'MAPE (%)': [5.0, 10.0],
        })
        
        filepath = tmp_path / "custom.csv"
        service.export_results(str(filepath), df)
        
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 2
    
    def test_export_results_no_data(self, service):
        """Test export without data raises error."""
        with pytest.raises(ValueError, match="No results to export"):
            service.export_results("test.csv")
    
    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------
    
    def test_get_results(self, service):
        """Test getting all results."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}
        
        service.compare_models(y_true, preds)
        results = service.get_results()
        
        assert len(results) == 2
        assert 'A' in results
        assert 'B' in results
    
    def test_get_last_comparison(self, service):
        """Test getting last comparison."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30]), 'B': np.array([15, 25, 35])}
        
        result = service.compare_models(y_true, preds)
        last = service.get_last_comparison()
        
        assert last is result
        assert last.best_model == 'A'
    
    def test_clear_results(self, service):
        """Test clearing results."""
        y_true = np.array([10, 20, 30])
        preds = {'A': np.array([10, 20, 30])}
        
        service.compare_models(y_true, preds)
        assert len(service.get_results()) == 1
        
        service.clear_results()
        assert len(service.get_results()) == 0
        assert service.get_last_comparison() is None
    
    # ------------------------------------------------------------------
    # Edge Cases
    # ------------------------------------------------------------------
    
    def test_evaluate_model_numpy_arrays(self, service):
        """Test evaluation with numpy arrays."""
        y_true = [10, 20, 30]
        y_pred = [12, 18, 33]
        
        metrics = service.evaluate_model(y_true, y_pred, "Test")
        
        assert isinstance(metrics, ModelMetrics)
    
    def test_metrics_dataclass(self):
        """Test ModelMetrics dataclass."""
        metrics = ModelMetrics(
            model_name='Test',
            mae=1.5,
            rmse=2.0,
            mape=5.0,
            n_samples=10,
        )
        
        d = metrics.to_dict()
        assert d['Model'] == 'Test'
        assert d['MAE'] == 1.5
        assert d['RMSE'] == 2.0
        assert d['MAPE (%)'] == 5.0
        assert d['N Samples'] == 10
    
    def test_comparison_result_dataclass(self):
        """Test ComparisonResult dataclass."""
        df = pd.DataFrame({'Model': ['A', 'B'], 'MAE': [1.0, 2.0], 'RMSE': [1.5, 2.5], 'MAPE (%)': [5.0, 10.0]})
        metrics = ModelMetrics('A', 1.0, 1.5, 5.0, 5)
        
        result = ComparisonResult(
            metrics_table=df,
            best_model='A',
            best_metrics=metrics,
            improvement_pct=50.0,
        )
        
        d = result.to_dict()
        assert d['best_model'] == 'A'
        assert d['improvement_pct'] == 50.0