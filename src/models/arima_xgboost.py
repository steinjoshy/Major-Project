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
    """

    def __init__(self, arima_order=(1, 1, 1), xgb_params=None, residual_lags=10):
        self.arima_order = arima_order
        self.residual_lags = residual_lags
        self.arima_model = None       # fitted ARIMAResultsWrapper
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
        self._train_data = None       # kept for future-forecast reference
        self._residuals = None        # ARIMA in-sample residuals

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


if __name__ == "__main__":
    print("Hybrid ARIMA + XGBoost module loaded successfully")
