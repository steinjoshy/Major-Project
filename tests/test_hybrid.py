"""Tests for Hybrid ARIMA + XGBoost model."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.models.arima_xgboost import HybridArimaXGBoost


class TestHybridModelConstruction:
    """Test Hybrid model initialization."""

    def test_default_initialization(self):
        hybrid = HybridArimaXGBoost()

        assert hybrid.arima_order == (1, 1, 1)
        assert hybrid.residual_lags == 10
        assert hybrid.xgb_model is not None
        assert hybrid.residual_scaler is not None
        assert hybrid.arima_fitted is False
        assert hybrid.xgb_trained is False

    def test_custom_initialization(self):
        hybrid = HybridArimaXGBoost(arima_order=(2, 1, 2), residual_lags=5)

        assert hybrid.arima_order == (2, 1, 2)
        assert hybrid.residual_lags == 5

    def test_xgb_default_params(self):
        hybrid = HybridArimaXGBoost()

        assert hybrid.xgb_model.n_estimators == 100
        assert hybrid.xgb_model.learning_rate == 0.05
        assert hybrid.xgb_model.max_depth == 5
        assert hybrid.xgb_model.random_state == 42


class TestHybridARIMAFitting:
    """Test ARIMA fitting with fallback."""

    def test_fit_arima_succeeds(self, train_data):
        hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
        hybrid.fit(train_data[:100])  # Use subset for speed

        assert hybrid.arima_fitted is True
        assert hybrid.arima_model is not None
        assert hasattr(hybrid.arima_model, 'resid')

    def test_fit_arima_fallback(self, train_data):
        """Test fallback works with problematic order."""
        hybrid = HybridArimaXGBoost(arima_order=(5, 5, 5))  # Likely to fail
        hybrid.fit(train_data[:100])

        assert hybrid.arima_fitted is True
        # Order may have changed due to fallback
        assert hybrid.arima_order in [(1,1,1), (0,1,1), (1,1,0), (0,1,0), (1,0,0), (5,5,5)]

    def test_fit_computes_residuals(self, train_data):
        hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
        hybrid.fit(train_data[:100])

        assert hybrid._residuals is not None
        assert len(hybrid._residuals) == len(train_data[:100])

    def test_fit_trains_xgb_on_residuals(self, train_data):
        hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1), residual_lags=5)
        hybrid.fit(train_data[:100])

        # XGBoost should be trained if enough residuals
        if len(hybrid._residuals) > hybrid.residual_lags + 5:
            assert hybrid.xgb_trained is True


class TestHybridEvaluation:
    """Test evaluation on held-out test set."""

    def test_evaluate_on_test_requires_fit(self, train_data):
        test_data = train_data[100:150]
        hybrid = HybridArimaXGBoost()

        with pytest.raises(ValueError, match="fit.*before"):
            hybrid.evaluate_on_test(train_data[:100], test_data)

    def test_evaluate_on_test_shape(self, train_data):
        test_data = train_data[100:150]
        hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
        hybrid.fit(train_data[:100])

        preds = hybrid.evaluate_on_test(train_data[:100], test_data)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(test_data)

    def test_evaluate_on_test_positive(self, train_data):
        test_data = train_data[100:150]
        hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
        hybrid.fit(train_data[:100])

        preds = hybrid.evaluate_on_test(train_data[:100], test_data)

        assert (preds >= 0).all()

    def test_evaluate_on_test_arima_component(self, train_data):
        """Verify ARIMA forecast component is present."""
        test_data = train_data[100:150]
        hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
        hybrid.fit(train_data[:100])

        preds = hybrid.evaluate_on_test(train_data[:100], test_data)

        # ARIMA forecast alone
        arima_fc = hybrid.arima_model.get_forecast(steps=len(test_data)).predicted_mean
        arima_fc = arima_fc.values if hasattr(arima_fc, 'values') else np.asarray(arima_fc)

        # Hybrid should differ from pure ARIMA (unless XGB failed)
        if hybrid.xgb_trained:
            # At least first correction should be non-zero
            diff = preds - arima_fc
            assert not np.allclose(diff, 0), "Hybrid should differ from ARIMA when XGB trained"


class TestHybridForecastFuture:
    """Test multi-step future forecasting."""

    def test_forecast_future_requires_fit(self):
        hybrid = HybridArimaXGBoost()

        with pytest.raises(ValueError, match="fit.*before"):
            hybrid.forecast_future(steps=10)

    def test_forecast_future_shape(self, fitted_hybrid, forecast_steps):
        forecast = fitted_hybrid.forecast_future(steps=forecast_steps)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == forecast_steps

    def test_forecast_future_positive(self, fitted_hybrid, forecast_steps):
        forecast = fitted_hybrid.forecast_future(steps=forecast_steps)
        assert (forecast >= 0).all()

    def test_forecast_future_multi_step_corrections(self, fitted_hybrid, forecast_steps):
        """BUG B2 TEST: Verify XGB corrections applied to ALL steps, not just step 0."""
        forecast = fitted_hybrid.forecast_future(steps=forecast_steps)

        # Get ARIMA-only forecast
        arima_fc = fitted_hybrid.arima_model.get_forecast(steps=forecast_steps).predicted_mean
        arima_fc = arima_fc.values if hasattr(arima_fc, 'values') else np.asarray(arima_fc)

        # Corrections = Hybrid - ARIMA
        corrections = forecast - arima_fc

        if fitted_hybrid.xgb_trained:
            # At least some corrections beyond index 0 should be non-zero
            # (rolling autoregressive on residuals)
            non_zero_corrections = np.count_nonzero(np.abs(corrections) > 1e-10)
            assert non_zero_corrections > 1, \
                f"Expected corrections at multiple steps, got {non_zero_corrections} non-zero"


class TestHybridResidualLagFeatures:
    """Test internal lag feature creation."""

    def test_create_lag_features(self):
        hybrid = HybridArimaXGBoost(residual_lags=5)
        series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        X, y = hybrid._create_lag_features(series)

        # 10 values, 5 lags -> 5 samples
        assert X.shape == (5, 5)
        assert y.shape == (5,)
        # First sample: [1,2,3,4,5] -> y=6
        np.testing.assert_array_equal(X[0], [1, 2, 3, 4, 5])
        assert y[0] == 6
        # Last sample: [5,6,7,8,9] -> y=10
        np.testing.assert_array_equal(X[-1], [5, 6, 7, 8, 9])
        assert y[-1] == 10

    def test_create_lag_features_insufficient(self):
        hybrid = HybridArimaXGBoost(residual_lags=5)
        series = np.array([1, 2, 3], dtype=float)

        X, y = hybrid._create_lag_features(series)

        assert len(X) == 0
        assert len(y) == 0


class TestHybridLegacyAPI:
    """Test backward compatibility methods."""

    def test_fit_arima_legacy(self, train_data):
        hybrid = HybridArimaXGBoost()
        hybrid.fit_arima(train_data[:100])
        assert hybrid.arima_fitted is True

    def test_predict_legacy(self, fitted_hybrid):
        forecast = fitted_hybrid.predict(steps=5)
        assert len(forecast) == 5

    def test_get_arima_params(self, fitted_hybrid):
        params = fitted_hybrid.get_arima_params()

        assert params is not None
        assert 'order' in params
        assert 'aic' in params
        assert 'bic' in params
