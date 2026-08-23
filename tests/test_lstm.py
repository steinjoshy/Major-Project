"""Tests for LSTM model: sequence generation, model construction, training, prediction, forecasting."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.models.lstm_model import LSTMForecaster


class TestLSTMModelConstruction:
    """Test LSTM model architecture and compilation."""

    def test_build_model_shapes(self):
        forecaster = LSTMForecaster(seq_length=30, lstm_units=64)
        model = forecaster.build_model((30, 1))

        assert model.input_shape == (None, 30, 1)
        assert model.output_shape == (None, 1)
        # Check layers exist
        layer_names = [layer.name for layer in model.layers]
        assert any('lstm' in name.lower() for name in layer_names)
        assert any('dense' in name.lower() for name in layer_names)

    def test_model_compilation(self):
        forecaster = LSTMForecaster(seq_length=30)
        model = forecaster.build_model((30, 1))

        assert model.optimizer is not None
        assert model.loss == 'mse'
        # metrics_names includes 'loss' and metric names; in newer Keras it may be 'compile_metrics'
        metrics = model.metrics_names
        assert 'loss' in metrics or metrics == ['loss', 'compile_metrics']

    def test_default_hyperparameters(self):
        forecaster = LSTMForecaster()
        assert forecaster.seq_length == 30
        assert forecaster.lstm_units == 64
        assert forecaster.epochs == 50
        assert forecaster.batch_size == 32

    def test_custom_hyperparameters(self):
        forecaster = LSTMForecaster(seq_length=15, lstm_units=32, epochs=10, batch_size=16)
        assert forecaster.seq_length == 15
        assert forecaster.lstm_units == 32
        assert forecaster.epochs == 10
        assert forecaster.batch_size == 16


class TestLSTMTraining:
    """Test LSTM training process."""

    def test_train_runs_without_error(self, lstm_data):
        X_tr, X_te, y_tr, y_te, preprocessor = lstm_data
        forecaster = LSTMForecaster(seq_length=30, epochs=2, batch_size=16)  # Minimal epochs for test

        history = forecaster.train(X_tr, y_tr, X_te, y_te, verbose=0)

        assert forecaster.model is not None
        assert history is not None
        assert 'loss' in history.history
        assert len(history.history['loss']) <= 2  # epochs=2

    def test_train_creates_expected_layers(self, lstm_data):
        X_tr, X_te, y_tr, y_te, _ = lstm_data
        forecaster = LSTMForecaster(seq_length=30, lstm_units=32, epochs=1, batch_size=16)
        forecaster.train(X_tr, y_tr, X_te, y_te, verbose=0)

        # Check architecture: LSTM(32) -> Dropout -> LSTM(16) -> Dropout -> Dense(16) -> Dense(1)
        layer_types = [type(layer).__name__ for layer in forecaster.model.layers]
        assert layer_types.count('LSTM') == 2
        assert layer_types.count('Dropout') == 2
        assert layer_types.count('Dense') == 2


class TestLSTMPrediction:
    """Test LSTM prediction on test data."""

    def test_predict_shape(self, trained_lstm):
        forecaster, X_te, y_te, _ = trained_lstm
        preds = forecaster.predict(X_te)

        assert preds.shape == (len(X_te), 1)
        assert preds.ndim == 2

    def test_predict_scaled_output(self, trained_lstm):
        forecaster, X_te, y_te, _ = trained_lstm
        preds = forecaster.predict(X_te)

        # Predictions should be in scaled space (0-1)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_inverse_scale_predictions(self, trained_lstm):
        forecaster, X_te, y_te, preprocessor = trained_lstm
        preds_scaled = forecaster.predict(X_te)
        preds = preprocessor.inverse_scale(preds_scaled).flatten()

        # Should be back in original scale
        assert preds.min() >= 0
        assert preds.max() > 100  # Sample data has values > 100

    def test_predict_untrained_raises(self):
        forecaster = LSTMForecaster(seq_length=30)
        X_dummy = np.random.rand(5, 30, 1)

        with pytest.raises(ValueError, match="not trained"):
            forecaster.predict(X_dummy)


class TestLSTMForecastFuture:
    """Test autoregressive future forecasting."""

    def test_forecast_future_shape(self, trained_lstm_for_forecast, forecast_steps, seq_length):
        forecaster, preprocessor, df_clean = trained_lstm_for_forecast

        # Get last sequence (already scaled)
        scaled_all = preprocessor.scale_data(df_clean['Sales'].values.reshape(-1, 1), fit=False)
        last_seq = scaled_all[-seq_length:].flatten()

        forecast = forecaster.forecast_future(last_seq, steps=forecast_steps, scaler=preprocessor.scaler)

        assert forecast.shape == (forecast_steps,)
        assert forecast.dtype in (float, np.float32, np.float64)

    def test_forecast_future_positive(self, trained_lstm_for_forecast, forecast_steps, seq_length):
        forecaster, preprocessor, df_clean = trained_lstm_for_forecast

        scaled_all = preprocessor.scale_data(df_clean['Sales'].values.reshape(-1, 1), fit=False)
        last_seq = scaled_all[-seq_length:].flatten()

        forecast = forecaster.forecast_future(last_seq, steps=forecast_steps, scaler=preprocessor.scaler)

        assert (forecast >= 0).all()

    def test_forecast_future_wrong_sequence_length_raises(self, trained_lstm_for_forecast):
        forecaster, _, _ = trained_lstm_for_forecast
        wrong_seq = np.random.rand(15)  # Wrong length

        with pytest.raises(ValueError, match="must have length"):
            forecaster.forecast_future(wrong_seq, steps=10)

    def test_forecast_future_untrained_raises(self):
        forecaster = LSTMForecaster(seq_length=30)
        seq = np.random.rand(30)

        with pytest.raises(ValueError, match="not trained"):
            forecaster.forecast_future(seq, steps=10)

    def test_forecast_future_autoregressive(self, trained_lstm_for_forecast):
        """Verify forecast is autoregressive (each step feeds next)."""
        forecaster, preprocessor, df_clean = trained_lstm_for_forecast

        scaled_all = preprocessor.scale_data(df_clean['Sales'].values.reshape(-1, 1), fit=False)
        last_seq = scaled_all[-30:].flatten()

        # Get 1-step and 2-step forecasts
        fc_1 = forecaster.forecast_future(last_seq, steps=1, scaler=preprocessor.scaler)
        fc_2 = forecaster.forecast_future(last_seq, steps=2, scaler=preprocessor.scaler)

        # First step should be identical
        assert fc_1[0] == fc_2[0]


class TestLSTMHistory:
    """Test training history retrieval."""

    def test_get_training_history(self, lstm_data):
        X_tr, X_te, y_tr, y_te, _ = lstm_data
        forecaster = LSTMForecaster(seq_length=30, epochs=2, batch_size=16)
        forecaster.train(X_tr, y_tr, X_te, y_te, verbose=0)

        history = forecaster.get_training_history()

        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) == len(history['val_loss'])

    def test_get_training_history_untrained(self):
        forecaster = LSTMForecaster()
        history = forecaster.get_training_history()
        assert history == {}


class TestLSTMModelPersistence:
    """Test model save/load."""

    def test_save_load_model(self, lstm_data, tmp_path):
        X_tr, X_te, y_tr, y_te, _ = lstm_data
        forecaster = LSTMForecaster(seq_length=30, epochs=1, batch_size=16)
        forecaster.train(X_tr, y_tr, X_te, y_te, verbose=0)

        model_path = tmp_path / "test_lstm.keras"
        forecaster.save_model(str(model_path))

        assert model_path.exists()

        # Load into new forecaster
        new_forecaster = LSTMForecaster(seq_length=30)
        new_forecaster.load_model(str(model_path))

        assert new_forecaster.model is not None
        # Predictions should match
        preds_orig = forecaster.predict(X_te)
        preds_new = new_forecaster.predict(X_te)
        np.testing.assert_allclose(preds_orig, preds_new, rtol=1e-5)
