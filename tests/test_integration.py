"""Integration tests: full pipeline and golden fixture generation."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.inventory.optimization import InventoryOptimization
from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.lstm_model import LSTMForecaster
from src.models.model_comparison import ModelComparison

FIXTURE_DIR = ROOT / "tests" / "fixtures"


def test_lstm_training_pipeline(pipeline_data):
    """Train LSTM and verify outputs."""
    X_tr, X_te, y_tr, y_te = (
        pipeline_data['X_tr'], pipeline_data['X_te'],
        pipeline_data['y_tr'], pipeline_data['y_te']
    )
    preprocessor = pipeline_data['preprocessor']

    lstm = LSTMForecaster(seq_length=30, epochs=3, batch_size=16)
    lstm.build_model((X_tr.shape[1], X_tr.shape[2]))
    lstm.train(X_tr, y_tr, X_te, y_te, verbose=0)

    # Predict
    preds_scaled = lstm.predict(X_te)
    preds = preprocessor.inverse_scale(preds_scaled).flatten()
    actual = preprocessor.inverse_scale(y_te.reshape(-1, 1)).flatten()

    assert len(preds) == len(actual)
    assert (preds >= 0).all()


def test_hybrid_training_pipeline(pipeline_data):
    """Train Hybrid and verify outputs."""
    train_series = pipeline_data['train_series']
    test_series = pipeline_data['test_series']

    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    hybrid.fit(train_series)

    preds = hybrid.evaluate_on_test(train_series, test_series)

    assert len(preds) == len(test_series)
    assert (preds >= 0).all()


def test_model_comparison_pipeline(pipeline_data):
    """Compare both models."""
    preprocessor = pipeline_data['preprocessor']
    X_tr, X_te, y_tr, y_te = (
        pipeline_data['X_tr'], pipeline_data['X_te'],
        pipeline_data['y_tr'], pipeline_data['y_te']
    )
    train_series = pipeline_data['train_series']
    test_series = pipeline_data['test_series']

    # LSTM
    lstm = LSTMForecaster(seq_length=30, epochs=3, batch_size=16)
    lstm.build_model((X_tr.shape[1], X_tr.shape[2]))
    lstm.train(X_tr, y_tr, X_te, y_te, verbose=0)
    lstm_preds_scaled = lstm.predict(X_te)
    lstm_preds = preprocessor.inverse_scale(lstm_preds_scaled).flatten()

    # Hybrid
    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    hybrid.fit(train_series)
    hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

    # Align (current behavior)
    y_test_actual = preprocessor.inverse_scale(y_te.reshape(-1, 1)).flatten()
    min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))

    comparator = ModelComparison()
    comp_df = comparator.compare_models(
        y_test_actual[:min_len],
        {'LSTM': lstm_preds[:min_len], 'Hybrid ARIMA+XGBoost': hybrid_preds[:min_len]}
    )

    assert len(comp_df) == 2
    assert 'LSTM' in comp_df['Model'].values
    assert 'Hybrid ARIMA+XGBoost' in comp_df['Model'].values


def test_inventory_optimization_pipeline(pipeline_data):
    """Full inventory optimization from forecasts."""
    df_clean = pipeline_data['df_clean']

    # Mock forecasts with some variability
    forecast_steps = 30
    mean_demand = df_clean['Sales'].mean()
    # Add some variation to trigger safety stock
    lstm_fc = np.random.normal(mean_demand, mean_demand * 0.1, forecast_steps)
    hybrid_fc = np.random.normal(mean_demand, mean_demand * 0.1, forecast_steps)
    ensemble_fc = (lstm_fc + hybrid_fc) / 2
    ensemble_fc = np.maximum(ensemble_fc, 0)  # No negative demand

    inv_opt = InventoryOptimization(service_level=0.95)
    recs = inv_opt.generate_inventory_recommendations(
        demand_data=df_clean['Sales'].values,
        lead_time=7,
        forecast_data=ensemble_fc
    )

    assert recs['safety_stock'] > 0
    assert recs['reorder_point'] > recs['safety_stock']
    assert recs['safety_stock_basis'] == 'forecast variability'

    # Projection
    proj_df = inv_opt.forecast_inventory_levels(
        current_stock=1000,
        forecast_demand=ensemble_fc,
        reorder_point=recs['reorder_point'],
        lead_time=7,
        safety_stock=recs['safety_stock']
    )

    assert len(proj_df) == forecast_steps
    assert (proj_df['Inventory_Level'] >= 0).all()


def test_generate_baseline_fixtures(pipeline_data):
    """Run pipeline and save baseline outputs for regression testing."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline_data['preprocessor']
    df_clean = pipeline_data['df_clean']
    seq_length = pipeline_data['seq_length']
    test_size = pipeline_data['test_size']

    # Train both models (minimal epochs for speed)
    X_tr, X_te, y_tr, y_te = (
        pipeline_data['X_tr'], pipeline_data['X_te'],
        pipeline_data['y_tr'], pipeline_data['y_te']
    )
    train_series = pipeline_data['train_series']
    test_series = pipeline_data['test_series']

    # LSTM
    lstm = LSTMForecaster(seq_length=seq_length, epochs=2, batch_size=16)
    lstm.build_model((X_tr.shape[1], X_tr.shape[2]))
    lstm.train(X_tr, y_tr, X_te, y_te, verbose=0)
    lstm_preds_scaled = lstm.predict(X_te)
    lstm_preds = preprocessor.inverse_scale(lstm_preds_scaled).flatten()
    y_test_actual = preprocessor.inverse_scale(y_te.reshape(-1, 1)).flatten()

    # Hybrid
    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    hybrid.fit(train_series)
    hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

    # Align
    min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))

    # Metrics
    comparator = ModelComparison()
    comp_df = comparator.compare_models(
        y_test_actual[:min_len],
        {'LSTM': lstm_preds[:min_len], 'Hybrid ARIMA+XGBoost': hybrid_preds[:min_len]}
    )

    # Save fixtures
    fixtures = {
        'data_shape': {'n_rows': len(df_clean), 'n_train': len(X_tr), 'n_test': len(X_te)},
        'seq_length': seq_length,
        'test_size': test_size,
        'min_eval_len': min_len,
        'lstm_preds': lstm_preds[:min_len].tolist(),
        'hybrid_preds': hybrid_preds[:min_len].tolist(),
        'y_test_actual': y_test_actual[:min_len].tolist(),
        'metrics': comp_df.to_dict('records'),
    }

    fixture_path = FIXTURE_DIR / "baseline_pipeline.json"
    with open(fixture_path, 'w') as f:
        json.dump(fixtures, f, indent=2)

    # Verify saved
    assert fixture_path.exists()

    with open(fixture_path) as f:
        loaded = json.load(f)

    assert loaded['data_shape']['n_rows'] == len(df_clean)
    assert len(loaded['lstm_preds']) == min_len
    assert len(loaded['metrics']) == 2


def test_verify_fixtures_loadable():
    """Verify fixtures can be loaded and have expected structure."""
    fixture_path = FIXTURE_DIR / "baseline_pipeline.json"

    if not fixture_path.exists():
        pytest.skip("Run test_generate_baseline_fixtures first")

    with open(fixture_path) as f:
        fixtures = json.load(f)

    assert 'data_shape' in fixtures
    assert 'lstm_preds' in fixtures
    assert 'hybrid_preds' in fixtures
    assert 'metrics' in fixtures
    assert len(fixtures['metrics']) == 2


def test_inventory_fixtures(pipeline_data):
    """Generate inventory optimization baseline fixtures."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = pipeline_data['df_clean']
    forecast_steps = 30

    # Simple ensemble forecast with variability
    mean_demand = df_clean['Sales'].mean()
    ensemble_fc = np.random.normal(mean_demand, mean_demand * 0.1, forecast_steps)
    ensemble_fc = np.maximum(ensemble_fc, 0)

    inv_opt = InventoryOptimization(service_level=0.95)
    recs = inv_opt.generate_inventory_recommendations(
        demand_data=df_clean['Sales'].values,
        lead_time=7,
        forecast_data=ensemble_fc
    )

    proj_df = inv_opt.forecast_inventory_levels(
        current_stock=1000,
        forecast_demand=ensemble_fc,
        reorder_point=recs['reorder_point'],
        lead_time=7,
        safety_stock=recs['safety_stock']
    )

    fixtures = {
        'recs': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in recs.items()},
        'projection': proj_df.to_dict('records'),
    }

    fixture_path = FIXTURE_DIR / "baseline_inventory.json"
    with open(fixture_path, 'w') as f:
        json.dump(fixtures, f, indent=2, default=str)

    assert fixture_path.exists()

    with open(fixture_path) as f:
        loaded = json.load(f)

    assert 'safety_stock' in loaded['recs']
    assert 'reorder_point' in loaded['recs']
    assert len(loaded['projection']) == forecast_steps


def test_pipeline_regression(pipeline_data):
    """Compare current outputs to baseline fixtures."""
    fixture_path = FIXTURE_DIR / "baseline_pipeline.json"

    if not fixture_path.exists():
        pytest.skip("Baseline fixtures not generated yet")

    with open(fixture_path) as f:
        baseline = json.load(f)

    # Run current pipeline
    preprocessor = pipeline_data['preprocessor']
    X_tr, X_te, y_tr, y_te = (
        pipeline_data['X_tr'], pipeline_data['X_te'],
        pipeline_data['y_tr'], pipeline_data['y_te']
    )
    train_series = pipeline_data['train_series']
    test_series = pipeline_data['test_series']

    # LSTM
    lstm = LSTMForecaster(seq_length=30, epochs=2, batch_size=16)
    lstm.build_model((X_tr.shape[1], X_tr.shape[2]))
    lstm.train(X_tr, y_tr, X_te, y_te, verbose=0)
    lstm_preds_scaled = lstm.predict(X_te)
    lstm_preds = preprocessor.inverse_scale(lstm_preds_scaled).flatten()
    y_test_actual = preprocessor.inverse_scale(y_te.reshape(-1, 1)).flatten()

    # Hybrid
    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    hybrid.fit(train_series)
    hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

    min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))

    # Compare predictions (allow small numerical differences)
    np.testing.assert_allclose(
        lstm_preds[:min_len], baseline['lstm_preds'],
        rtol=0.05, atol=1.0,  # 5% or 1 unit tolerance
        err_msg="LSTM predictions diverged from baseline"
    )
    np.testing.assert_allclose(
        hybrid_preds[:min_len], baseline['hybrid_preds'],
        rtol=0.05, atol=1.0,
        err_msg="Hybrid predictions diverged from baseline"
    )

    # Compare metrics
    comparator = ModelComparison()
    comp_df = comparator.compare_models(
        y_test_actual[:min_len],
        {'LSTM': lstm_preds[:min_len], 'Hybrid ARIMA+XGBoost': hybrid_preds[:min_len]}
    )

    for i, baseline_metric in enumerate(baseline['metrics']):
        current = comp_df.iloc[i].to_dict()
        for key in ['MAE', 'RMSE', 'MAPE (%)']:
            if key in baseline_metric and key in current:
                assert current[key] == pytest.approx(baseline_metric[key], rel=0.1), \
                    f"{key} for {baseline_metric['Model']} diverged"
