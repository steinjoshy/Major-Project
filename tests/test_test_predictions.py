"""Regression tests for the test.csv prediction workflow.

Covers the bugs fixed in ``ForecastingService.generate_test_predictions``:

- every (store_id, sku_id) group must be forecast from ITS OWN training context
  (previously ``X_test_lstm[0]`` was reused for all groups, so every group got
  an identical — fully correlated — LSTM forecast);
- the hybrid ARIMA+XGBoost forecast must be advanced past the withheld
  validation holdout so it lines up with the test.csv period;
- the emitted model must be the same best model the dashboard reports (selected
  by validation RMSE);
- a group with no usable training history must fail loudly instead of receiving
  placeholder zeros.

``ForecastingService`` imports TensorFlow transitively, so it is imported inside
each test (collection-time TensorFlow import trips the project's
``-W error::DeprecationWarning`` via google.protobuf).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import DataPreprocessor


def _import_service():
    """Import ForecastingService at runtime (pulls in TensorFlow)."""
    pytest.importorskip('tensorflow')
    from src.services.forecasting_service import ForecastingService
    return ForecastingService


class _FakeHybrid:
    """Minimal stand-in that returns a known ramp for the requested horizon."""

    def __init__(self, train_lens):
        self._panel_models = {
            key: {'train_data': np.zeros(n)} for key, n in train_lens.items()
        }
        self.last_steps = None

    def forecast_panel_future(self, steps):
        self.last_steps = steps
        return {key: np.arange(steps, dtype=float) for key in self._panel_models}


def _make_panel(n_stores=3, n_skus=2, n_days=120, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for st in range(n_stores):
        for sk in range(n_skus):
            base = rng.uniform(10, 50)
            vals = np.maximum(0, base + rng.normal(0, 3, n_days)).round()
            for d in range(n_days):
                rows.append({
                    'store_id': f'S{st}',
                    'sku_id': f'K{sk}',
                    'week': pd.Timestamp('2021-01-04') + pd.Timedelta(days=d),
                    'units_sold': float(vals[d]),
                })
    return pd.DataFrame(rows)


def _split_train_test(dfc, train_frac=0.8):
    train_parts, test_parts = [], []
    for (s, k), g in dfc.groupby(['store_id', 'sku_id']):
        g = g.sort_values('week').reset_index(drop=True)
        cut = int(len(g) * train_frac)
        train_parts.append(g.iloc[:cut])
        test = g.iloc[cut:].copy()
        test['record_ID'] = [f'{s}_{k}_{i}' for i in range(len(test))]
        test_parts.append(test.drop(columns=['units_sold']))
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


# ----------------------------------------------------------------------
# Fast logic tests
# ----------------------------------------------------------------------

def test_hybrid_forecast_advanced_past_validation_gap():
    """The hybrid forecast must start at the end of train_df, not the end of the
    hybrid model's (shorter) fitting window."""
    ForecastingService = _import_service()
    keys = [('S0', 'K0'), ('S0', 'K1')]
    svc = ForecastingService(seq_length=10)
    svc.hybrid_model = _FakeHybrid({keys[0]: 12, keys[1]: 15})

    train_df = pd.DataFrame({
        'store_id': ['S0'] * 40,
        'sku_id': ['K0'] * 20 + ['K1'] * 20,   # 20 full rows per group
    })
    group_info = {
        keys[0]: {'skipped': False, 'n_test': 5},   # gap 8 -> steps 13
        keys[1]: {'skipped': False, 'n_test': 5},   # gap 5 -> steps 10
    }

    aligned = svc._aligned_hybrid_forecasts(train_df, group_info, 'store_id', 'sku_id')

    assert svc.hybrid_model.last_steps == 13
    # Window for each group is the LAST n_test values of its own ramp.
    assert np.array_equal(aligned[keys[0]], np.arange(8, 13, dtype=float))
    assert np.array_equal(aligned[keys[1]], np.arange(5, 10, dtype=float))


def test_best_model_selected_by_rmse():
    ForecastingService = _import_service()
    svc = ForecastingService(seq_length=10)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    svc._test_predictions = {
        'LSTM': np.array([1.1, 2.1, 3.1, 4.1]),
        'Hybrid ARIMA+XGBoost': np.array([2.0, 3.0, 4.0, 5.0]),
        'y_test': y,
    }
    assert svc.get_best_model_name() == 'LSTM'

    svc._test_predictions['LSTM'] = np.array([5.0, 6.0, 7.0, 8.0])
    assert svc.get_best_model_name() == 'Hybrid ARIMA+XGBoost'


def test_select_predictions_falls_back_to_available_model():
    ForecastingService = _import_service()
    sel = ForecastingService._select_predictions
    assert np.allclose(sel('LSTM', None, np.array([1.0, 2.0])), [1.0, 2.0])
    assert np.allclose(sel('Hybrid ARIMA+XGBoost', np.array([3.0, 4.0]), None), [3.0, 4.0])
    assert np.allclose(sel('Ensemble', np.array([2.0, 2.0]), np.array([4.0, 4.0])), [3.0, 3.0])
    assert sel('LSTM', None, None) is None


def test_unusable_group_raises_instead_of_zero_fill():
    """A group present in test.csv but absent from training must not silently
    receive zeros."""
    ForecastingService = _import_service()
    svc = ForecastingService(seq_length=10)
    svc._is_trained = True

    class _NoopLSTM:
        def forecast_future(self, *a, **k):
            raise AssertionError("should not be called for a skipped group")

    svc.lstm_model = _NoopLSTM()
    svc.hybrid_model = _FakeHybrid({})

    pp = DataPreprocessor()
    pp.date_column = 'week'
    pp._panel_scalers = {('S0', 'K0'): None}
    svc.preprocessor = pp

    # train_df has only S0/K0; test adds S9/K9 with no history.
    train_df = pd.DataFrame({
        'store_id': ['S0'] * 20,
        'sku_id': ['K0'] * 20,
        'week': pd.date_range('2021-01-04', periods=20),
        'units_sold': np.arange(20.0),
    })
    test_df = pd.DataFrame({
        'store_id': ['S9'] * 3,
        'sku_id': ['K9'] * 3,
        'week': pd.date_range('2021-01-04', periods=3),
        'record_ID': ['S9_K9_0', 'S9_K9_1', 'S9_K9_2'],
    })

    with pytest.raises(ValueError, match='no usable training history'):
        svc.generate_test_predictions(
            train_df=train_df, test_df=test_df, sales_col='units_sold',
            store_col='store_id', sku_col='sku_id', date_col='week',
        )


# ----------------------------------------------------------------------
# Integration test with a real trained panel
# ----------------------------------------------------------------------

def test_generate_test_predictions_is_per_group_and_complete():
    ForecastingService = _import_service()

    pp = DataPreprocessor()
    dfc = pp.clean_data(_make_panel(), date_col='week', sales_col='units_sold',
                        store_col='store_id', sku_col='sku_id')
    train_df, test_df = _split_train_test(dfc, 0.8)

    svc = ForecastingService(seq_length=10, lstm_epochs=1, lstm_batch_size=32,
                             arima_order=(1, 1, 1))
    svc.train_all_models_panel(
        train_df=train_df, sales_col='units_sold', test_size=0.25,
        store_col='store_id', sku_col='sku_id', date_col='week', preprocessor=pp,
    )

    sub = svc.generate_test_predictions(
        train_df=train_df, test_df=test_df, sales_col='units_sold',
        store_col='store_id', sku_col='sku_id', date_col='week',
        record_id_col='record_ID', model_name='LSTM',
    )

    # Complete, unique, non-negative submission covering every test record.
    assert len(sub) == len(test_df)
    assert set(sub['record_ID']) == set(test_df['record_ID'])
    assert (sub['units_sold'] >= 0).all()

    # Core regression: per-group contexts => forecasts must NOT all be identical.
    sub['grp'] = sub['record_ID'].str.rsplit('_', n=1).str[0]
    series = {g: grp['units_sold'].to_numpy() for g, grp in sub.groupby('grp')}
    names = list(series)
    assert len(names) >= 2
    max_corr = np.nanmax([
        abs(np.corrcoef(series[a], series[b])[0, 1])
        for i, a in enumerate(names) for b in names[i + 1:]
    ])
    assert max_corr < 0.999, f"group forecasts are identical (max corr={max_corr:.4f})"
