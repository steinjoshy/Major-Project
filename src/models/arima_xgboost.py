import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

_FALLBACK_ARIMA_ORDERS = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (0, 1, 0), (1, 0, 0)]


class HybridArimaXGBoost:
    """
    Hybrid forecasting model combining ARIMA and XGBoost.

    Methodology (as per Phase-I report):
        Step 1 — Fit ARIMA on training demand to capture linear patterns.
        Step 2 — Compute ARIMA residuals = Actual − ARIMA_in_sample_fit.
        Step 3 — Train XGBoost on lag features of residuals to learn nonlinear error patterns.
        Step 4 — Hybrid Forecast = ARIMA_forecast + XGBoost_residual_correction.

    Supports both single series and panel data (multiple store/sku combinations).
    """

    def __init__(self, arima_order=(1, 1, 1), xgb_params=None, residual_lags=10):
        self.arima_order = arima_order
        self.residual_lags = residual_lags
        self.arima_model = None       # fitted ARIMAResultsWrapper (single series)
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=0,
            **(xgb_params or {})
        )
        self.residual_scaler = StandardScaler()
        self.arima_fitted = False
        self.xgb_trained = False
        self._train_data = None       # kept for future-forecast reference (single series)
        self._residuals = None        # ARIMA in-sample residuals (single series)

        # Panel data support
        self._panel_models = {}       # (store_id, sku_id) -> dict with arima_model, xgb_model, etc.
        self._is_panel = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fit_arima_with_fallback(self, data, order):
        """Try to fit ARIMA with given order; fall back through safe alternatives."""
        orders_to_try = [order] + [o for o in _FALLBACK_ARIMA_ORDERS if o != order]
        last_exc = None
        for o in orders_to_try:
            try:
                m = ARIMA(data, order=o)
                fitted = m.fit()
                self.arima_order = o
                return fitted
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"All ARIMA orders failed. Last error: {last_exc}")

    def _create_lag_features(self, series):
        """Create a feature matrix of lagged values from a 1-D series."""
        lags = self.residual_lags
        X, y = [], []
        for i in range(lags, len(series)):
            X.append(series[i - lags: i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def _generate_residual_corrections(self, steps, initial_residuals):
        """
        Generate XGBoost residual corrections for multiple steps using
        rolling autoregressive approach.

        Args:
            steps: Number of forecast steps
            initial_residuals: Array of in-sample residuals from training

        Returns:
            np.array of corrections of length `steps`
        """
        corrections = np.zeros(steps)
        if not self.xgb_trained:
            return corrections

        # Start with last `residual_lags` residuals from training
        residual_window = initial_residuals[-self.residual_lags:].copy()

        for i in range(steps):
            if len(residual_window) == self.residual_lags:
                try:
                    corr = self.xgb_model.predict(
                        self.residual_scaler.transform(residual_window.reshape(1, -1))
                    )[0]
                    corrections[i] = corr
                    # Roll window: drop oldest, append predicted correction
                    residual_window = np.append(residual_window[1:], corr)
                except Exception:
                    corrections[i] = 0.0
            else:
                corrections[i] = 0.0

        return corrections

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_data):
        """
        Full training pipeline on training data only.

        1. Fit ARIMA on train_data.
        2. Compute in-sample residuals.
        3. Train XGBoost on lagged residuals.

        Args:
            train_data: 1-D array of historical demand values (training split only)
        """
        train_data = np.asarray(train_data, dtype=float)
        self._train_data = train_data

        # Step 1 — Fit ARIMA
        self.arima_model = self._fit_arima_with_fallback(train_data, self.arima_order)
        self.arima_fitted = True

        # Step 2 — Residuals = actual − ARIMA in-sample fitted values
        _resid = self.arima_model.resid
        self._residuals = _resid.values if hasattr(_resid, 'values') else np.asarray(_resid)

        # Step 3 — Train XGBoost on lagged residuals
        X_r, y_r = self._create_lag_features(self._residuals)
        if len(X_r) >= 5:
            X_r_scaled = self.residual_scaler.fit_transform(X_r)
            self.xgb_model.fit(X_r_scaled, y_r)
            self.xgb_trained = True

    # ------------------------------------------------------------------
    # Evaluation on held-out test set (no leakage)
    # ------------------------------------------------------------------

    def evaluate_on_test(self, train_data, test_data):
        """
        Generate hybrid predictions for the test set without data leakage.

        The model must have been fitted on train_data via fit() first.
        ARIMA forecasts test_len steps ahead from the end of training.
        XGBoost correction uses rolling autoregressive approach on residuals.

        Args:
            train_data: Training demand array (same as passed to fit())
            test_data:  Held-out test demand array

        Returns:
            hybrid_predictions: np.array of length len(test_data)
        """
        if not self.arima_fitted:
            raise ValueError("Call fit(train_data) before evaluate_on_test().")

        test_len = len(test_data)

        # ARIMA forecast for test_len steps ahead
        _fc = self.arima_model.get_forecast(steps=test_len).predicted_mean
        arima_forecast = _fc.values if hasattr(_fc, 'values') else np.asarray(_fc)

        # XGBoost correction for all steps using rolling autoregressive approach
        xgb_corrections = self._generate_residual_corrections(test_len, self._residuals)

        hybrid_preds = arima_forecast + xgb_corrections
        return hybrid_preds

    # ------------------------------------------------------------------
    # Future Forecasting
    # ------------------------------------------------------------------

    def forecast_future(self, steps=30):
        """
        Generate future demand forecast using the hybrid model.
        Must have called fit() first (on full data).

        Returns:
            np.array of length `steps`
        """
        if not self.arima_fitted:
            raise ValueError("Call fit() before forecast_future().")

        _ff = self.arima_model.get_forecast(steps=steps).predicted_mean
        arima_forecast = _ff.values if hasattr(_ff, 'values') else np.asarray(_ff)

        # XGBoost correction for all steps using rolling autoregressive approach
        xgb_corrections = self._generate_residual_corrections(steps, self._residuals)

        return arima_forecast + xgb_corrections

    # ------------------------------------------------------------------
    # Legacy API (kept for backward compat)
    # ------------------------------------------------------------------

    def fit_arima(self, data):
        """Legacy: fit ARIMA only. Use fit() for full pipeline."""
        self.fit(data)

    def fit_xgb_on_residuals(self, data):
        """Legacy: no-op if fit() already called."""
        if not self.arima_fitted:
            self.fit(data)

    def predict(self, steps=1):
        """Legacy: forecast future steps. Use forecast_future() instead."""
        return self.forecast_future(steps=steps)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_arima_params(self):
        """Return ARIMA order and information criteria."""
        if self.arima_model is None:
            return None
        return {
            'order': self.arima_order,
            'aic': getattr(self.arima_model, 'aic', None),
            'bic': getattr(self.arima_model, 'bic', None),
        }

    # ------------------------------------------------------------------
    # Panel Data Support
    # ------------------------------------------------------------------

    def fit_panel(self, panel_data, arima_order=None, verbose=False):
        """
        Fit Hybrid model for each (store_id, sku_id) group in panel data.

        Args:
            panel_data: Dict mapping (store_id, sku_id) -> {
                'train_series': np.array,
                'test_series': np.array (optional),
                'split_idx': int
            }
            arima_order: Optional ARIMA order override
            verbose: Print progress

        Returns:
            Dict of results per group
        """
        self._is_panel = True
        self._panel_models = {}
        results = {}

        order = arima_order or self.arima_order

        for (store_id, sku_id), data in panel_data.items():
            if data.get('skipped', False):
                results[(store_id, sku_id)] = {
                    'skipped': True,
                    'reason': data.get('reason', 'Unknown')
                }
                continue

            train_series = data['train_series']
            test_series = data.get('test_series')

            if len(train_series) < 20:
                results[(store_id, sku_id)] = {
                    'skipped': True,
                    'reason': f'Insufficient training data: {len(train_series)}'
                }
                continue

            try:
                # Fit ARIMA with fallback
                arima_model = self._fit_arima_with_fallback(train_series, order)
                arima_fitted = True

                # Compute in-sample residuals
                _resid = arima_model.resid
                residuals = _resid.values if hasattr(_resid, 'values') else np.asarray(_resid)

                # Train XGBoost on lagged residuals
                X_r, y_r = self._create_lag_features(residuals)
                residual_scaler = StandardScaler()
                xgb_trained = False
                xgb_model = None

                if len(X_r) >= 5:
                    X_r_scaled = residual_scaler.fit_transform(X_r)
                    xgb_model = XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=5,
                        random_state=42,
                        verbosity=0
                    )
                    xgb_model.fit(X_r_scaled, y_r)
                    xgb_trained = True

                # Store model components
                self._panel_models[(store_id, sku_id)] = {
                    'arima_model': arima_model,
                    'arima_order': self.arima_order,
                    'residuals': residuals,
                    'residual_scaler': residual_scaler,
                    'xgb_model': xgb_model,
                    'xgb_trained': xgb_trained,
                    'train_data': train_series
                }

                # Evaluate on test if provided
                test_preds = None
                if test_series is not None and len(test_series) > 0:
                    test_preds = self._predict_panel_group(
                        (store_id, sku_id), len(test_series)
                    )

                results[(store_id, sku_id)] = {
                    'skipped': False,
                    'arima_order': self.arima_order,
                    'xgb_trained': xgb_trained,
                    'n_train': len(train_series),
                    'n_test': len(test_series) if test_series is not None else 0,
                    'test_predictions': test_preds
                }

                if verbose:
                    print(f"  Fitted {store_id}-{sku_id}: ARIMA{self.arima_order}, XGB={xgb_trained}")

            except Exception as e:
                results[(store_id, sku_id)] = {
                    'skipped': True,
                    'reason': f'Fitting failed: {str(e)}'
                }
                if verbose:
                    print(f"  Failed {store_id}-{sku_id}: {e}")

        return results

    def _predict_panel_group(self, group_key, steps):
        """Generate predictions for a single panel group."""
        model_info = self._panel_models.get(group_key)
        if model_info is None:
            return None

        arima_model = model_info['arima_model']
        xgb_model = model_info['xgb_model']
        residual_scaler = model_info['residual_scaler']
        residuals = model_info['residuals']
        xgb_trained = model_info['xgb_trained']

        # ARIMA forecast
        _fc = arima_model.get_forecast(steps=steps).predicted_mean
        arima_forecast = _fc.values if hasattr(_fc, 'values') else np.asarray(_fc)

        # XGBoost correction
        xgb_corrections = np.zeros(steps)
        if xgb_trained and xgb_model is not None:
            residual_window = residuals[-self.residual_lags:].copy()
            for i in range(steps):
                if len(residual_window) == self.residual_lags:
                    try:
                        corr = xgb_model.predict(
                            residual_scaler.transform(residual_window.reshape(1, -1))
                        )[0]
                        xgb_corrections[i] = corr
                        residual_window = np.append(residual_window[1:], corr)
                    except Exception:
                        xgb_corrections[i] = 0.0
                else:
                    xgb_corrections[i] = 0.0

        return arima_forecast + xgb_corrections

    def predict_panel_test(self, panel_data):
        """
        Generate test set predictions for all groups in panel data.

        Args:
            panel_data: Dict mapping (store_id, sku_id) -> {
                'test_series': np.array
            }

        Returns:
            Dict mapping (store_id, sku_id) -> predictions array
        """
        if not self._is_panel:
            raise ValueError("Call fit_panel() first for panel data.")

        predictions = {}
        for (store_id, sku_id), data in panel_data.items():
            test_series = data.get('test_series')
            if test_series is not None and len(test_series) > 0:
                preds = self._predict_panel_group((store_id, sku_id), len(test_series))
                predictions[(store_id, sku_id)] = preds
            else:
                predictions[(store_id, sku_id)] = np.array([])
        return predictions

    def forecast_panel_future(self, steps=30):
        """
        Generate future forecasts for all fitted panel groups.

        Returns:
            Dict mapping (store_id, sku_id) -> forecast array of length `steps`
        """
        if not self._is_panel:
            raise ValueError("Call fit_panel() first for panel data.")

        forecasts = {}
        for group_key, model_info in self._panel_models.items():
            steps_forecast = self._predict_panel_group(group_key, steps)
            forecasts[group_key] = steps_forecast
        return forecasts


if __name__ == "__main__":
    print("Hybrid ARIMA + XGBoost module loaded successfully")
