"""
Forecasting Service for AI Demand Forecasting.

Provides unified interface for LSTM and Hybrid ARIMA+XGBoost models.
Pure Python service with no Streamlit dependencies.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.lstm_model import LSTMForecaster
from src.preprocessing import DataPreprocessor


@dataclass
class ModelConfig:
    """Configuration for a forecasting model."""
    name: str
    model_type: str  # 'lstm' or 'hybrid'
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of model training."""
    model_name: str
    model_type: str
    train_loss: float | None = None
    val_loss: float | None = None
    epochs_trained: int | None = None
    history: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Result of forecasting."""
    model_name: str
    predictions: np.ndarray
    dates: pd.DatetimeIndex | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastingService:
    """
    Service for training and using forecasting models.
    
    Supports LSTM and Hybrid ARIMA+XGBoost models.
    No Streamlit dependencies.
    """

    def __init__(
        self,
        seq_length: int = 30,
        lstm_epochs: int = 50,
        lstm_batch_size: int = 32,
        arima_order: tuple[int, int, int] = (1, 1, 1),
    ):
        """
        Initialize the forecasting service.
        
        Args:
            seq_length: Sequence length for LSTM
            lstm_epochs: Max epochs for LSTM training
            lstm_batch_size: Batch size for LSTM training
            arima_order: ARIMA (p,d,q) order for Hybrid model
        """
        self.seq_length = seq_length
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.arima_order = arima_order

        # Model instances
        self.lstm_model: LSTMForecaster | None = None
        self.hybrid_model: HybridArimaXGBoost | None = None
        self.preprocessor: DataPreprocessor | None = None

        # Training state
        self._is_trained = False
        self._training_results: dict[str, TrainingResult] = {}
        self._test_predictions: dict[str, np.ndarray] = {}
        self._y_test: np.ndarray | None = None
        self._scaler: Any | None = None
        self._lstm_test_start_idx: int | None = None

    # ------------------------------------------------------------------
    # LSTM Methods
    # ------------------------------------------------------------------

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: int = 0,
    ) -> TrainingResult:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, seq_length, 1)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Keras verbosity
            
        Returns:
            TrainingResult with training metrics
        """
        self.lstm_model = LSTMForecaster(
            seq_length=self.seq_length,
            epochs=self.lstm_epochs,
            batch_size=self.lstm_batch_size,
        )
        self.lstm_model.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.lstm_model.train(X_train, y_train, X_val, y_val, verbose=verbose)

        result = TrainingResult(
            model_name='LSTM',
            model_type='lstm',
            train_loss=history.history['loss'][-1] if history.history.get('loss') else None,
            val_loss=history.history['val_loss'][-1] if history.history.get('val_loss') else None,
            epochs_trained=len(history.history.get('loss', [])),
            history=history.history,
            metadata={
                'seq_length': self.seq_length,
                'epochs': self.lstm_epochs,
                'batch_size': self.lstm_batch_size,
            },
        )
        self._training_results['LSTM'] = result
        self._is_trained = True
        return result

    def predict_lstm(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate LSTM predictions on test data.
        
        Args:
            X_test: Test sequences
            
        Returns:
            Scaled predictions
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Call train_lstm() first.")
        return self.lstm_model.predict(X_test)

    def forecast_lstm_future(
        self,
        last_scaled_sequence: np.ndarray,
        steps: int,
        scaler: Any | None = None,
    ) -> np.ndarray:
        """
        Generate LSTM future forecast.
        
        Args:
            last_scaled_sequence: Last sequence (scaled) of length seq_length
            steps: Number of future steps to forecast
            scaler: Scaler for inverse transform
            
        Returns:
            Forecast values (original scale if scaler provided)
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Call train_lstm() first.")
        return self.lstm_model.forecast_future(last_scaled_sequence, steps, scaler)

    def get_lstm_history(self) -> dict:
        """Get LSTM training history."""
        if self.lstm_model is None:
            return {}
        return self.lstm_model.get_training_history()

    # ------------------------------------------------------------------
    # Hybrid ARIMA+XGBoost Methods
    # ------------------------------------------------------------------

    def train_hybrid(
        self,
        train_data: np.ndarray,
    ) -> TrainingResult:
        """
        Train the Hybrid ARIMA+XGBoost model.
        
        Args:
            train_data: 1-D array of training demand values
            
        Returns:
            TrainingResult with model info
        """
        self.hybrid_model = HybridArimaXGBoost(arima_order=self.arima_order)
        self.hybrid_model.fit(train_data)

        result = TrainingResult(
            model_name='Hybrid ARIMA+XGBoost',
            model_type='hybrid',
            metadata={
                'arima_order': self.hybrid_model.arima_order,
                'residual_lags': self.hybrid_model.residual_lags,
                'xgb_trained': self.hybrid_model.xgb_trained,
            },
        )
        self._training_results['Hybrid ARIMA+XGBoost'] = result
        self._is_trained = True
        return result

    def evaluate_hybrid_on_test(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate Hybrid model on test set.
        
        Args:
            train_data: Training data (same as used in fit)
            test_data: Held-out test data
            
        Returns:
            Hybrid predictions for test set
        """
        if self.hybrid_model is None:
            raise ValueError("Hybrid model not trained. Call train_hybrid() first.")
        return self.hybrid_model.evaluate_on_test(train_data, test_data)

    def forecast_hybrid_future(self, steps: int) -> np.ndarray:
        """
        Generate Hybrid future forecast.
        
        Args:
            steps: Number of future steps
            
        Returns:
            Forecast values
        """
        if self.hybrid_model is None:
            raise ValueError("Hybrid model not trained. Call train_hybrid() first.")
        return self.hybrid_model.forecast_future(steps)

    def get_hybrid_params(self) -> dict | None:
        """Get Hybrid ARIMA parameters."""
        if self.hybrid_model is None:
            return None
        return self.hybrid_model.get_arima_params()

    # ------------------------------------------------------------------
    # Full Pipeline Training
    # ------------------------------------------------------------------

    def train_all_models(
        self,
        df: pd.DataFrame,
        sales_col: str,
        test_size: float = 0.2,
        preprocessor: DataPreprocessor | None = None,
    ) -> dict[str, TrainingResult]:
        """
        Train both LSTM and Hybrid models on a dataset.
        
        Args:
            df: Cleaned dataframe
            sales_col: Sales column name
            test_size: Test split ratio
            preprocessor: Optional preprocessor (creates new if None)
            
        Returns:
            Dict of training results by model name
        """
        if preprocessor is None:
            preprocessor = DataPreprocessor()
        self.preprocessor = preprocessor

        # LSTM preparation (no leakage)
        X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_lstm_data(
            df, sales_col, self.seq_length, test_size
        )
        self._scaler = scaler
        self._lstm_test_start_idx = preprocessor.train_size + self.seq_length

        # Train LSTM
        self.train_lstm(X_train, y_train, X_test, y_test, verbose=0)
        lstm_preds_scaled = self.predict_lstm(X_test)
        lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Hybrid preparation - align test window with LSTM
        values = df[sales_col].values.astype(float)
        train_series = values[:self._lstm_test_start_idx]
        test_series = values[self._lstm_test_start_idx:]

        # Train Hybrid
        self.train_hybrid(train_series)
        hybrid_preds = self.evaluate_hybrid_on_test(train_series, test_series)

        # Align lengths
        min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))
        lstm_preds = lstm_preds[:min_len]
        hybrid_preds = hybrid_preds[:min_len]
        y_test_actual = y_test_actual[:min_len]

        # Store test predictions
        self._test_predictions = {
            'LSTM': lstm_preds,
            'Hybrid ARIMA+XGBoost': hybrid_preds,
            'y_test': y_test_actual,
        }

        return self._training_results

    # ------------------------------------------------------------------
    # Future Forecasting
    # ------------------------------------------------------------------

    def generate_future_forecast(
        self,
        df: pd.DataFrame,
        sales_col: str,
        forecast_steps: int,
    ) -> dict[str, np.ndarray]:
        """
        Generate future forecasts from both trained models.
        
        Uses the ALREADY-TRAINED models (no refitting).
        
        Args:
            df: Full cleaned dataframe
            sales_col: Sales column name
            forecast_steps: Number of future steps
            
        Returns:
            Dict of model_name -> forecast array
        """
        if not self._is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")

        # LSTM forecast
        scaled_all = self.preprocessor.scale_data(
            df[sales_col].values.reshape(-1, 1), fit=False
        )
        last_seq = scaled_all[-self.seq_length:].flatten()
        lstm_fc = self.forecast_lstm_future(last_seq, forecast_steps, self.preprocessor.scaler)

        # Hybrid forecast (uses already-trained model)
        hybrid_fc = self.forecast_hybrid_future(forecast_steps)

        # Ensemble
        ensemble_fc = (lstm_fc + hybrid_fc) / 2

        return {
            'LSTM': lstm_fc,
            'Hybrid ARIMA+XGBoost': hybrid_fc,
            'Ensemble': ensemble_fc,
        }

    def get_test_predictions(self) -> dict[str, np.ndarray]:
        """Get stored test set predictions."""
        return self._test_predictions.copy()

    def get_test_dates(
        self,
        df: pd.DataFrame,
        date_col: str,
    ) -> np.ndarray:
        """Get test set dates aligned with predictions."""
        if self._lstm_test_start_idx is None:
            raise ValueError("Models not trained yet.")
        test_len = len(self._test_predictions.get('y_test', []))
        return df[date_col].values[self._lstm_test_start_idx:self._lstm_test_start_idx + test_len]

    def is_trained(self) -> bool:
        """Check if models are trained."""
        return self._is_trained

    def get_training_results(self) -> dict[str, TrainingResult]:
        """Get all training results."""
        return self._training_results.copy()

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def save_lstm_model(self, filepath: str) -> None:
        """Save LSTM model to disk."""
        if self.lstm_model is not None:
            self.lstm_model.save_model(filepath)

    def load_lstm_model(self, filepath: str) -> None:
        """Load LSTM model from disk."""
        self.lstm_model = LSTMForecaster(seq_length=self.seq_length)
        self.lstm_model.load_model(filepath)
        self._is_trained = True

    # Note: Hybrid model persistence would require joblib for XGBoost + ARIMA
    # This can be added when ModelRegistry is implemented
