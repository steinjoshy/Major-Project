"""Focused tests for training completion and actionable errors.

Regression coverage for the "apparently stuck" panel training report:
- the long per-group ARIMA+XGBoost loop must report progress,
- ``train_all_models_panel`` must return results for BOTH models,
- degenerate ARIMA input must raise a clear, actionable error,
- recoverable statsmodels warnings (e.g. non-invertible starting MA) must not
  be treated as fatal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.arima_xgboost import HybridArimaXGBoost
from src.preprocessing import DataPreprocessor


def _make_panel(n_stores=4, n_skus=3, n_days=70, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for st in range(n_stores):
        for sk in range(n_skus):
            base = rng.uniform(5, 40)
            vals = np.maximum(0, base + rng.normal(0, 4, n_days)).round()
            for d in range(n_days):
                rows.append({
                    'store_id': f'S{st}',
                    'sku_id': f'K{sk}',
                    'week': pd.Timestamp('2020-01-01') + pd.Timedelta(days=d),
                    'units_sold': float(vals[d]),
                })
    return pd.DataFrame(rows)


def test_panel_training_returns_all_models_and_predictions():
    """Training must return and report both models plus aligned predictions."""
    pytest.importorskip('tensorflow')
    from src.services.forecasting_service import ForecastingService

    df = _make_panel()
    pp = DataPreprocessor()
    dfc = pp.clean_data(df, date_col='week', sales_col='units_sold',
                        store_col='store_id', sku_col='sku_id')

    svc = ForecastingService(seq_length=10, lstm_epochs=1, lstm_batch_size=32,
                             arima_order=(1, 1, 1))
    progress = []

    def _cb(fraction, message):
        progress.append((fraction, message))

    results = svc.train_all_models_panel(
        train_df=dfc, sales_col='units_sold', test_size=0.25,
        store_col='store_id', sku_col='sku_id', date_col='week',
        preprocessor=pp, progress_callback=_cb,
    )

    # Both models must be reported back (this was missing for the panel path).
    assert 'LSTM' in results
    assert 'Hybrid ARIMA+XGBoost' in results

    preds = svc.get_test_predictions()
    assert set(preds) >= {'LSTM', 'Hybrid ARIMA+XGBoost', 'y_test'}
    n = len(preds['y_test'])
    assert n > 0
    assert len(preds['LSTM']) == n
    assert len(preds['Hybrid ARIMA+XGBoost']) == n

    # Progress callback fires, stays in range, and reaches completion.
    assert progress, 'progress_callback was never called'
    fractions = [f for f, _ in progress]
    assert fractions[0] <= 0.05
    assert fractions[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_arima_non_finite_input_raises_clear_error():
    """A NaN/inf series must fail fast with an actionable message, not an
    opaque XGBoost error deep inside residual training."""
    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    bad = np.r_[np.arange(30.0), [np.nan], np.arange(29.0)]
    with pytest.raises(ValueError, match='finite'):
        hybrid.fit(bad)


def test_arima_too_short_series_raises_clear_error():
    hybrid = HybridArimaXGBoost()
    with pytest.raises(ValueError, match='at least 3'):
        hybrid.fit(np.array([1.0, 2.0]))


def test_arima_constant_series_warns_but_still_fits():
    """Non-invertible / degenerate warnings are recoverable, not fatal."""
    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    hybrid.fit(np.full(50, 12.0))
    assert hybrid.arima_fitted is True
    assert len(hybrid.forecast_future(steps=5)) == 5
