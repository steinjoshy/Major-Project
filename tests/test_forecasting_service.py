"""Tests for ForecastingService."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.preprocessing import DataPreprocessor
from src.services.forecasting_service import (
    ForecastingService,
    TrainingResult,
)


class TestForecastingService:
    """Test ForecastingService functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time-series data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        # Add trend and seasonality
        trend = np.linspace(0, 20, 200)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 30)
        noise = np.random.normal(0, 5, 200)
        sales = 50 + trend + seasonal + noise
        sales = np.maximum(sales, 0)  # No negative sales

        df = pd.DataFrame({
            'Date': dates,
            'Sales': sales,
        })
        return df

    @pytest.fixture
    def clean_data(self, sample_data):
        """Clean the sample data."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_data, 'Date', 'Sales')
        return df_clean, preprocessor

    @pytest.fixture
    def service(self):
        """Create ForecastingService instance."""
        return ForecastingService(
            seq_length=20,
            lstm_epochs=5,  # Minimal for testing
            lstm_batch_size=16,
            arima_order=(1, 1, 1),
        )

    def test_train_lstm(self, service, clean_data):
        """Test LSTM training."""
        df_clean, preprocessor = clean_data

        # Prepare data
        X_tr, X_te, y_tr, y_te, scaler = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length=20, test_size=0.2
        )

        # Train
        result = service.train_lstm(X_tr, y_tr, X_te, y_te, verbose=0)

        assert isinstance(result, TrainingResult)
        assert result.model_name == 'LSTM'
        assert result.model_type == 'lstm'
        assert result.train_loss is not None
        assert result.val_loss is not None
        assert result.epochs_trained is not None
        assert result.epochs_trained <= 5  # We set 5 epochs
        assert service.is_trained() is True

    def test_predict_lstm(self, service, clean_data):
        """Test LSTM prediction."""
        df_clean, preprocessor = clean_data

        X_tr, X_te, y_tr, y_te, scaler = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length=20, test_size=0.2
        )

        service.train_lstm(X_tr, y_tr, X_te, y_te, verbose=0)
        preds = service.predict_lstm(X_te)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X_te), 1)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0  # Scaled

    def test_forecast_lstm_future(self, service, clean_data):
        """Test LSTM future forecasting."""
        df_clean, preprocessor = clean_data

        X_tr, X_te, y_tr, y_te, scaler = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length=20, test_size=0.2
        )

        service.train_lstm(X_tr, y_tr, X_te, y_te, verbose=0)

        # Get last sequence
        scaled_all = preprocessor.scale_data(
            df_clean['Sales'].values.reshape(-1, 1), fit=False
        )
        last_seq = scaled_all[-20:].flatten()

        # Forecast
        forecast = service.forecast_lstm_future(last_seq, steps=10, scaler=preprocessor.scaler)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 10
        assert forecast.dtype in (float, np.float32, np.float64)
        assert (forecast >= 0).all()

    def test_get_lstm_history(self, service, clean_data):
        """Test getting LSTM training history."""
        df_clean, preprocessor = clean_data
        X_tr, X_te, y_tr, y_te, _ = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length=20, test_size=0.2
        )

        service.train_lstm(X_tr, y_tr, X_te, y_te, verbose=0)
        history = service.get_lstm_history()

        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) > 0

    def test_train_hybrid(self, service, clean_data):
        """Test Hybrid ARIMA+XGBoost training."""
        df_clean, _ = clean_data
        train_data = df_clean['Sales'].values.astype(float)

        result = service.train_hybrid(train_data[:100])  # Use subset for speed

        assert isinstance(result, TrainingResult)
        assert result.model_name == 'Hybrid ARIMA+XGBoost'
        assert result.model_type == 'hybrid'
        assert result.metadata['arima_order'] is not None
        assert 'xgb_trained' in result.metadata
        assert service.is_trained() is True

    def test_evaluate_hybrid_on_test(self, service, clean_data):
        """Test Hybrid evaluation on test set."""
        df_clean, _ = clean_data
        train_data = df_clean['Sales'].values.astype(float)

        service.train_hybrid(train_data[:100])
        test_data = train_data[100:130]

        preds = service.evaluate_hybrid_on_test(train_data[:100], test_data)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(test_data)
        assert (preds >= 0).all()

    def test_forecast_hybrid_future(self, service, clean_data):
        """Test Hybrid future forecasting."""
        df_clean, _ = clean_data
        train_data = df_clean['Sales'].values.astype(float)

        service.train_hybrid(train_data[:100])

        forecast = service.forecast_hybrid_future(steps=10)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 10
        assert (forecast >= 0).all()

    def test_get_hybrid_params(self, service, clean_data):
        """Test getting Hybrid ARIMA parameters."""
        df_clean, _ = clean_data
        train_data = df_clean['Sales'].values.astype(float)

        service.train_hybrid(train_data[:100])
        params = service.get_hybrid_params()

        assert params is not None
        assert 'order' in params
        assert 'aic' in params
        assert 'bic' in params

    def test_train_all_models(self, service, clean_data):
        """Test full pipeline training both models."""
        df_clean, preprocessor = clean_data

        results = service.train_all_models(df_clean, 'Sales', test_size=0.2, preprocessor=preprocessor)

        assert 'LSTM' in results
        assert 'Hybrid ARIMA+XGBoost' in results
        assert len(results) == 2
        assert service.is_trained()

    def test_generate_future_forecast(self, service, clean_data):
        """Test future forecast generation."""
        df_clean, preprocessor = clean_data

        service.train_all_models(df_clean, 'Sales', test_size=0.2, preprocessor=preprocessor)

        forecasts = service.generate_future_forecast(df_clean, 'Sales', forecast_steps=15)

        assert 'LSTM' in forecasts
        assert 'Hybrid ARIMA+XGBoost' in forecasts
        assert 'Ensemble' in forecasts
        assert len(forecasts['LSTM']) == 15
        assert len(forecasts['Hybrid ARIMA+XGBoost']) == 15
        assert len(forecasts['Ensemble']) == 15

    def test_get_test_predictions(self, service, clean_data):
        """Test getting stored test predictions."""
        df_clean, preprocessor = clean_data

        service.train_all_models(df_clean, 'Sales', test_size=0.2, preprocessor=preprocessor)

        preds = service.get_test_predictions()

        assert 'LSTM' in preds
        assert 'Hybrid ARIMA+XGBoost' in preds
        assert 'y_test' in preds
        assert len(preds['LSTM']) == len(preds['y_test'])

    def test_get_test_dates(self, service, clean_data):
        """Test getting test set dates."""
        df_clean, preprocessor = clean_data

        service.train_all_models(df_clean, 'Sales', test_size=0.2, preprocessor=preprocessor)

        dates = service.get_test_dates(df_clean, 'Date')

        assert len(dates) == len(service.get_test_predictions()['y_test'])
        assert isinstance(dates, np.ndarray)

    def test_save_load_lstm_model(self, service, clean_data, tmp_path):
        """Test LSTM model persistence."""
        df_clean, preprocessor = clean_data
        X_tr, X_te, y_tr, y_te, _ = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length=20, test_size=0.2
        )

        service.train_lstm(X_tr, y_tr, X_te, y_te, verbose=0)

        model_path = tmp_path / "test_lstm.keras"
        service.save_lstm_model(str(model_path))

        assert model_path.exists()

        # Load into new service
        new_service = ForecastingService(seq_length=20)
        new_service.load_lstm_model(str(model_path))

        assert new_service.is_trained()
        assert new_service.lstm_model is not None

    def test_get_training_results(self, service, clean_data):
        """Test getting training results."""
        df_clean, preprocessor = clean_data

        results = service.train_all_models(df_clean, 'Sales', test_size=0.2, preprocessor=preprocessor)
        training_results = service.get_training_results()

        assert len(training_results) == 2
        assert 'LSTM' in training_results
        assert 'Hybrid ARIMA+XGBoost' in training_results

    def test_untrained_errors(self, service):
        """Test that untrained models raise appropriate errors."""
        with pytest.raises(ValueError, match="not trained"):
            service.predict_lstm(np.random.rand(5, 20, 1))

        with pytest.raises(ValueError, match="not trained"):
            service.forecast_lstm_future(np.random.rand(20), steps=5)

        # train_hybrid should work without prior training (it trains the model)
        service.train_hybrid(np.random.rand(100))
        assert service.hybrid_model is not None

        # But forecasting without training should fail
        new_service = ForecastingService()
        with pytest.raises(ValueError, match="not trained"):
            new_service.forecast_hybrid_future(5)

        with pytest.raises(ValueError, match="not trained"):
            new_service.generate_future_forecast(
                pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=50),
                              'Sales': np.random.rand(50)}),
                'Sales', 10
            )


class TestForecastingServiceConfigs:
    """Test different configuration options."""

    def test_custom_config(self):
        """Test custom configuration."""
        service = ForecastingService(
            seq_length=15,
            lstm_epochs=3,
            lstm_batch_size=8,
            arima_order=(2, 1, 2),
        )

        assert service.seq_length == 15
        assert service.lstm_epochs == 3
        assert service.lstm_batch_size == 8
        assert service.arima_order == (2, 1, 2)

    def test_default_config(self):
        """Test default configuration."""
        service = ForecastingService()

        assert service.seq_length == 30
        assert service.lstm_epochs == 50
        assert service.lstm_batch_size == 32
        assert service.arima_order == (1, 1, 1)
