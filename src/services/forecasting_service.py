"""
Forecasting Service for AI Demand Forecasting.

Provides unified interface for LSTM and Hybrid ARIMA+XGBoost models.
Pure Python service with no Streamlit dependencies.
"""
from dataclasses import dataclass, field
from typing import Any

import logging
import time

import numpy as np
import pandas as pd

from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.lstm_model import LSTMForecaster
from src.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


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
        progress_callback=None,
        log_every: int = 5,
    ) -> TrainingResult:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, seq_length, 1)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Keras verbosity
            progress_callback: Optional callable(epoch, total_epochs) for UI progress
            log_every: Log a training line every N epochs
            
        Returns:
            TrainingResult with training metrics
        """
        started = time.perf_counter()
        self.lstm_model = LSTMForecaster(
            seq_length=self.seq_length,
            epochs=self.lstm_epochs,
            batch_size=self.lstm_batch_size,
        )
        self.lstm_model.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.lstm_model.train(
            X_train, y_train, X_val, y_val, verbose=verbose,
            progress_callback=progress_callback, log_every=log_every,
        )

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
                'train_seconds': round(time.perf_counter() - started, 3),
            },
        )
        self._training_results['LSTM'] = result
        self._is_trained = True
        logger.info(
            "LSTM result: train_loss=%s val_loss=%s epochs=%s",
            result.train_loss, result.val_loss, result.epochs_trained,
        )
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

    # ------------------------------------------------------------------
    # Panel Data Methods (store_id, sku_id)
    # ------------------------------------------------------------------

    def train_all_models_panel(
        self,
        train_df: pd.DataFrame,
        sales_col: str,
        test_size: float = 0.2,
        store_col: str = 'store_id',
        sku_col: str = 'sku_id',
        date_col: str = 'week',
        preprocessor: DataPreprocessor | None = None,
        progress_callback=None,
    ) -> dict[str, TrainingResult]:
        """
        Train both LSTM and Hybrid models on panel data (multiple store/SKU series).

        Args:
            train_df: Cleaned training dataframe with store_id, sku_id columns
            sales_col: Sales/target column name
            test_size: Validation split ratio within train_df
            store_col: Store identifier column
            sku_col: SKU identifier column
            date_col: Date column name
            preprocessor: Optional preprocessor (creates new if None)
            progress_callback: Optional callable(fraction, message) with fraction in
                [0, 1] for the whole run. Used to keep the UI/terminal informed
                during the long per-group ARIMA/XGBoost loop.

        Returns:
            Dict of training results by model name (contains both 'LSTM' and
            'Hybrid ARIMA+XGBoost').
        """
        if preprocessor is None:
            preprocessor = DataPreprocessor()
        self.preprocessor = preprocessor

        def _emit(fraction, message):
            if progress_callback is None:
                return
            try:
                progress_callback(float(fraction), message)
            except Exception:
                logger.debug("progress_callback failed", exc_info=True)

        pipeline_started = time.perf_counter()

        # LSTM preparation for panel data (no leakage)
        _emit(0.02, 'Preparing LSTM sequences (per store/SKU)...')
        t0 = time.perf_counter()
        X_train, X_test, y_train, y_test, panel_scalers, group_info = preprocessor.prepare_lstm_data_panel(
            train_df, sales_col, self.seq_length, test_size, store_col, sku_col
        )
        self._panel_scalers = panel_scalers
        self._panel_group_info = group_info
        logger.info(
            "Panel LSTM prep: %d train / %d test sequences from %d groups in %.1fs",
            len(X_train), len(X_test), len(group_info), time.perf_counter() - t0,
        )

        # Train LSTM on combined data from all groups

        def _lstm_progress(epoch, total_epochs):
            _emit(0.05 + 0.50 * (epoch / max(total_epochs, 1)),
                  f'Training LSTM epoch {epoch}/{total_epochs}...')

        self.train_lstm(
            X_train, y_train, X_test, y_test, verbose=0,
            progress_callback=_lstm_progress,
        )

        # Evaluate LSTM on test set
        lstm_preds_scaled = self.predict_lstm(X_test)

        # Inverse transform per group (using stored scalers)
        # For simplicity, we'll use the first group's scaler as representative
        # In production, you'd want per-group inverse transforms
        if panel_scalers:
            default_scaler = list(panel_scalers.values())[0]
        else:
            default_scaler = self.preprocessor.scaler
        self._scaler = default_scaler

        lstm_preds = default_scaler.inverse_transform(lstm_preds_scaled).flatten()
        y_test_actual = default_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Hybrid preparation for panel data
        _emit(0.58, 'Preparing hybrid (ARIMA+XGBoost) panel data...')
        t0 = time.perf_counter()
        panel_hybrid_data = preprocessor.prepare_hybrid_data_panel(
            train_df, sales_col, test_size, store_col, sku_col
        )
        logger.info(
            "Panel hybrid prep: %d groups in %.1fs",
            len(panel_hybrid_data), time.perf_counter() - t0,
        )

        # Train Hybrid on panel data
        self.hybrid_model = HybridArimaXGBoost(arima_order=self.arima_order)
        total_groups = len(panel_hybrid_data)

        def _hybrid_progress(done, total, group_key):
            _emit(0.60 + 0.35 * (done / max(total, 1)),
                  f'Fitting hybrid ARIMA+XGBoost: group {done}/{total} {group_key}')

        t_hybrid = time.perf_counter()
        hybrid_results = self.hybrid_model.fit_panel(
            panel_hybrid_data, progress_callback=_hybrid_progress
        )
        hybrid_secs = time.perf_counter() - t_hybrid

        n_fitted = sum(1 for r in hybrid_results.values() if not r.get('skipped'))
        if n_fitted == 0:
            raise RuntimeError(
                f"Hybrid ARIMA+XGBoost fitting failed for all {total_groups} groups. "
                "Check that the demand series are long enough and contain finite values."
            )
        logger.info(
            "Hybrid panel fit: %d/%d groups fitted in %.1fs",
            n_fitted, total_groups, hybrid_secs,
        )

        # Record a TrainingResult for the Hybrid model so callers receive both
        # models from the returned dict (previously only LSTM was recorded).
        self._training_results['Hybrid ARIMA+XGBoost'] = TrainingResult(
            model_name='Hybrid ARIMA+XGBoost',
            model_type='hybrid',
            metadata={
                'arima_order': self.hybrid_model.arima_order,
                'residual_lags': self.hybrid_model.residual_lags,
                'n_groups': total_groups,
                'n_groups_fitted': n_fitted,
                'fit_seconds': round(hybrid_secs, 3),
                'panel': True,
            },
        )

        # Get test predictions from Hybrid
        _emit(0.96, 'Collecting hybrid test predictions...')
        hybrid_test_preds_dict = self.hybrid_model.predict_panel_test(panel_hybrid_data)

        # Combine predictions from all groups for evaluation
        # We need to align them with the test set
        all_hybrid_preds = []
        all_y_test = []

        # Get actual test values from train_df for evaluation
        train_df_sorted = train_df.sort_values([store_col, sku_col, date_col]).reset_index(drop=True)
        for (store_id, sku_id), info in group_info.items():
            if info.get('skipped', False):
                continue
            split_idx = info.get('sequence_split', 0)
            if split_idx > 0:
                group_data = train_df_sorted[
                    (train_df_sorted[store_col] == store_id) &
                    (train_df_sorted[sku_col] == sku_id)
                ].sort_values(date_col).reset_index(drop=True)
                if len(group_data) > split_idx:
                    test_actual = group_data[sales_col].values[split_idx:]
                    all_y_test.extend(test_actual)

                    hybrid_preds = hybrid_test_preds_dict.get((store_id, sku_id))
                    if hybrid_preds is not None and len(hybrid_preds) > 0:
                        all_hybrid_preds.extend(hybrid_preds[:len(test_actual)])

        all_hybrid_preds = np.array(all_hybrid_preds)
        all_y_test = np.array(all_y_test)

        # Align lengths with LSTM
        min_len = min(len(lstm_preds), len(all_hybrid_preds), len(all_y_test))
        if min_len > 0:
            lstm_preds = lstm_preds[:min_len]
            all_hybrid_preds = all_hybrid_preds[:min_len]
            all_y_test = all_y_test[:min_len]

        # Store test predictions
        self._test_predictions = {
            'LSTM': lstm_preds,
            'Hybrid ARIMA+XGBoost': all_hybrid_preds,
            'y_test': all_y_test,
        }

        # Store hybrid results for reference
        self._hybrid_panel_results = hybrid_results

        _emit(1.0, 'Panel training complete.')
        logger.info(
            "Panel training complete in %.1fs: LSTM=%d preds, Hybrid=%d preds, y_test=%d",
            time.perf_counter() - pipeline_started,
            len(lstm_preds), len(all_hybrid_preds), len(all_y_test),
        )

        return self._training_results

    def generate_test_predictions(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        sales_col: str,
        store_col: str = 'store_id',
        sku_col: str = 'sku_id',
        date_col: str = 'week',
        record_id_col: str = 'record_ID',
        model_name: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for test.csv using trained models.

        Each (store_id, sku_id) group is forecast from the *end of its own*
        training series in ``train_df``; the models are NOT re-fitted here.

        Args:
            train_df: Training dataframe with target column
            test_df: Test dataframe (no target column)
            sales_col: Target column name (in train_df)
            store_col: Store identifier column
            sku_col: SKU identifier column
            date_col: Date column name
            record_id_col: Record ID column in test_df
            model_name: Which model's predictions to emit — 'LSTM',
                'Hybrid ARIMA+XGBoost', 'Ensemble', or None to auto-select the
                best model by validation RMSE (matching the dashboard).

        Returns:
            DataFrame with columns [record_ID, units_sold] matching sample_submission format

        Raises:
            ValueError: if a group in test_df has no usable training history (no
                fabricated/placeholder values are emitted).
        """
        if not self._is_trained:
            raise ValueError("Models not trained. Call train_all_models_panel() first.")
        if self.lstm_model is None:
            raise ValueError("LSTM model is not available for prediction.")
        if self.hybrid_model is None:
            raise ValueError("Hybrid ARIMA+XGBoost model is not available for prediction.")

        # Prepare per-group LSTM context sequences (one row per usable group).
        X_test_lstm, _, group_info = self.preprocessor.prepare_lstm_test_sequences(
            train_df, test_df, sales_col, self.seq_length, store_col, sku_col
        )

        # --- LSTM: one autoregressive forecast per group using that group's own
        # tail of scaled training values as context. X_test_lstm rows follow the
        # non-skipped group order, so advance an index in lock-step (previously
        # X_test_lstm[0] was reused for every group, giving identical forecasts).
        lstm_forecasts: dict[tuple, np.ndarray] = {}
        seq_index = 0
        for (store_id, sku_id), info in group_info.items():
            if info.get('skipped', False):
                continue
            context_seq = X_test_lstm[seq_index]
            seq_index += 1
            scaler = self._panel_scalers.get((store_id, sku_id), self._scaler)
            lstm_forecasts[(store_id, sku_id)] = self.forecast_lstm_future(
                context_seq.flatten(), info['n_test'], scaler
            )

        # --- Hybrid: the panel hybrid model was fitted on the training portion of
        # each group (validation holdout withheld). test.csv begins after ALL
        # training rows, so advance past the holdout gap before taking the window.
        hybrid_forecasts = self._aligned_hybrid_forecasts(
            train_df, group_info, store_col, sku_col
        )

        selected = model_name or self.get_best_model_name()

        # Build submission in a deterministic (store, sku, date) order.
        test_df_sorted = test_df.sort_values([store_col, sku_col, date_col]).reset_index(drop=True)
        submissions = []
        unpredicted = []

        for (store_id, sku_id), test_group in test_df_sorted.groupby([store_col, sku_col]):
            test_group = test_group.sort_values(date_col).reset_index(drop=True)
            record_ids = test_group[record_id_col].values
            n_test = len(test_group)

            lstm_pred = lstm_forecasts.get((store_id, sku_id))
            hybrid_pred = hybrid_forecasts.get((store_id, sku_id))
            if lstm_pred is not None:
                lstm_pred = np.asarray(lstm_pred)[:n_test]
            if hybrid_pred is not None:
                hybrid_pred = np.asarray(hybrid_pred)[:n_test]

            final_pred = self._select_predictions(selected, lstm_pred, hybrid_pred)
            if final_pred is None or len(final_pred) != n_test:
                unpredicted.append((store_id, sku_id))
                continue

            for rid, pred in zip(record_ids, final_pred):
                submissions.append({
                    record_id_col: rid,
                    'units_sold': max(0.0, float(pred)),  # demand is non-negative
                })

        if unpredicted:
            shown = unpredicted[:10]
            suffix = f" (+{len(unpredicted) - 10} more)" if len(unpredicted) > 10 else ""
            raise ValueError(
                f"Cannot generate test predictions for {len(unpredicted)} group(s) with no "
                f"usable training history: {shown}{suffix}. Providing placeholder values would "
                "fabricate demand; supply historical data for these (store_id, sku_id) pairs instead."
            )

        return pd.DataFrame(submissions, columns=[record_id_col, 'units_sold'])

    @staticmethod
    def _select_predictions(selected, lstm_pred, hybrid_pred):
        """Choose the prediction array for the requested model.

        Falls back to whichever single model is available when the requested one
        is missing, so a partially-fitted panel still yields real predictions.
        """
        if selected == 'LSTM':
            return lstm_pred if lstm_pred is not None else hybrid_pred
        if selected == 'Hybrid ARIMA+XGBoost':
            return hybrid_pred if hybrid_pred is not None else lstm_pred
        # 'Ensemble' / auto / unknown
        if lstm_pred is not None and hybrid_pred is not None:
            return (lstm_pred + hybrid_pred) / 2.0
        return lstm_pred if lstm_pred is not None else hybrid_pred

    def _aligned_hybrid_forecasts(self, train_df, group_info, store_col, sku_col):
        """Hybrid forecasts aligned to the test.csv period for each group.

        The panel hybrid model is fitted on the training portion of each group
        (a validation holdout is withheld). To predict test.csv — which follows
        the end of ``train_df`` — the ARIMA/XGB forecast must advance
        ``gap + n_test`` steps, where ``gap`` is the number of withheld training
        rows, and keep the final ``n_test`` values.

        Returns:
            Dict mapping (store_id, sku_id) -> aligned forecast array of len n_test.
        """
        panel_models = getattr(self.hybrid_model, '_panel_models', None) or {}
        if not panel_models:
            return {}

        train_sizes = train_df.groupby([store_col, sku_col]).size().to_dict()

        steps_needed = {}
        for key, info in group_info.items():
            if info.get('skipped', False):
                continue
            model_info = panel_models.get(key)
            if model_info is None:
                continue
            fitted_series = model_info.get('train_data')
            fitted_len = len(fitted_series) if fitted_series is not None else 0
            gap = max(int(train_sizes.get(key, 0)) - fitted_len, 0)
            steps_needed[key] = gap + int(info['n_test'])

        if not steps_needed:
            return {}

        raw = self.hybrid_model.forecast_panel_future(steps=max(steps_needed.values()))

        aligned = {}
        for key, steps in steps_needed.items():
            fc = raw.get(key)
            if fc is None or len(fc) < steps:
                continue
            n_test = int(group_info[key]['n_test'])
            aligned[key] = np.asarray(fc)[steps - n_test:steps]
        return aligned

    def get_best_model_name(self) -> str:
        """Name of the best model, by validation RMSE when available.

        Uses the stored panel test predictions — the same arrays the dashboard
        compares — so the submission is generated with the model the UI reports
        as best. Falls back to the lowest recorded validation loss, then to
        'Ensemble'.
        """
        tp = self._test_predictions or {}
        y_true = tp.get('y_test')
        candidates = {
            name: pred
            for name, pred in tp.items()
            if name in ('LSTM', 'Hybrid ARIMA+XGBoost')
            and pred is not None and len(pred) > 0
        }

        if y_true is not None and len(y_true) > 0 and candidates:
            best_name = None
            best_rmse = float('inf')
            y_true = np.asarray(y_true, dtype=float)
            for name, pred in candidates.items():
                pred = np.asarray(pred, dtype=float)
                n = min(len(y_true), len(pred))
                if n == 0:
                    continue
                rmse = float(np.sqrt(np.mean((y_true[:n] - pred[:n]) ** 2)))
                if rmse < best_rmse:
                    best_rmse, best_name = rmse, name
            if best_name is not None:
                return best_name

        # Fallback: lowest LSTM validation loss, if recorded.
        best_model = None
        best_loss = float('inf')
        for name, result in (self._training_results or {}).items():
            val = getattr(result, 'val_loss', None)
            if val is not None and val < best_loss:
                best_loss, best_model = val, name

        return best_model or 'Ensemble'

    def generate_future_forecast_panel(
        self,
        train_df: pd.DataFrame,
        sales_col: str,
        forecast_steps: int,
        store_col: str = 'store_id',
        sku_col: str = 'sku_id',
    ) -> dict[str, dict]:
        """
        Generate future forecasts for all panel groups.

        Returns:
            Dict mapping (store_id, sku_id) -> {model_name: forecast_array}
        """
        if not self._is_trained:
            raise ValueError("Models not trained. Call train_all_models_panel() first.")

        if not hasattr(self, '_panel_scalers') or not self._panel_scalers:
            raise ValueError("No panel scalers found. Train with panel data first.")

        results = {}

        for (store_id, sku_id), scaler in self._panel_scalers.items():
            # Get last seq_length values from training data for this group
            group_data = train_df[
                (train_df[store_col] == store_id) &
                (train_df[sku_col] == sku_id)
            ].sort_values(self.preprocessor.date_column)

            if len(group_data) < self.seq_length:
                continue

            last_values = group_data[sales_col].values[-self.seq_length:].reshape(-1, 1)
            last_scaled = scaler.transform(last_values).flatten()

            # LSTM forecast
            lstm_fc = self.forecast_lstm_future(last_scaled, forecast_steps, scaler)

            # Hybrid forecast
            hybrid_fc = self.hybrid_model.forecast_panel_future(forecast_steps).get((store_id, sku_id))
            if hybrid_fc is None:
                hybrid_fc = np.zeros(forecast_steps)

            # Ensemble
            ensemble_fc = (lstm_fc + hybrid_fc) / 2

            results[(store_id, sku_id)] = {
                'LSTM': lstm_fc,
                'Hybrid ARIMA+XGBoost': hybrid_fc,
                'Ensemble': ensemble_fc
            }

        return results
