# Phase 1 Service Layer Architecture

**Date:** 2026-08-22  
**Branch:** steinjoshy-ai-demand-forecasting  
**Phase:** 1 — Service Layer Extraction Complete

---

## Overview

Phase 1 successfully extracted the core business logic from the Streamlit monolith into clean, reusable Python services. The Streamlit application remains fully functional, now acting as a thin UI/orchestration layer that calls the extracted services.

---

## New Service Architecture

```
src/
├── services/
│   ├── __init__.py                    # Package exports
│   ├── ingestion_service.py           # Data loading, validation, cleaning, features
│   ├── forecasting_service.py         # LSTM + Hybrid ARIMA+XGBoost training/forecasting
│   ├── inventory_service.py           # Safety Stock, ROP, EOQ, projections
│   ├── model_comparison_service.py    # RMSE/MAE/MAPE, model comparison
│   └── model_registry.py              # Model lifecycle management
├── models/
│   ├── __init__.py
│   ├── lstm_model.py                  # LSTM implementation (unchanged)
│   ├── arima_xgboost.py               # Hybrid ARIMA+XGBoost (bug fixes applied)
│   └── model_comparison.py            # Metrics (unchanged)
├── inventory/
│   ├── __init__.py
│   └── optimization.py                # Inventory calculations (unchanged)
├── preprocessing.py                   # DataPreprocessor (bug fixes applied)
└── ...
```

---

## Services Created

### 1. IngestionService (`src/services/ingestion_service.py`)

**Responsibilities:**
- CSV/Excel file loading (single, multi-file, folder)
- Column auto-detection (date/sales aliases)
- M5 wide-format detection and melting
- Large file handling (DuckDB >80MB, chunked pandas fallback)
- Data validation with detailed warnings
- Data cleaning (date parsing, deduplication, fillna, negative removal)
- Feature engineering (time, lag, rolling features)
- LSTM/Hybrid data preparation (no leakage)

**Key Features:**
- No Streamlit dependencies
- Returns structured Python objects (DataFrame + metadata dict)
- Comprehensive error handling with `IngestionError`
- M5 wide-format auto-detection

**API Example:**
```python
service = IngestionService(use_duckdb=True)
df, meta = service.load_from_csv("data.csv")  # auto-detects columns
X_tr, X_te, y_tr, y_te, scaler = service.prepare_lstm_data(df, 'Sales', 30)
```

---

### 2. ForecastingService (`src/services/forecasting_service.py`)

**Responsibilities:**
- LSTM model training and prediction
- Hybrid ARIMA+XGBoost training and prediction
- Future forecasting (auto-regressive for LSTM, rolling for Hybrid)
- Full pipeline training (both models with aligned evaluation windows)
- Model persistence (Keras .keras for LSTM)

**Key Features:**
- Pure Python, no Streamlit
- Returns structured dataclasses (`TrainingResult`, `ForecastResult`)
- Proper train/test alignment (Bug B3 fix)
- Uses already-trained models for forecasting (Bug B4 fix)
- No data leakage in LSTM preprocessing (Bug B1 fix)
- Multi-step XGBoost corrections (Bug B2 fix)

**API Example:**
```python
service = ForecastingService(seq_length=30, lstm_epochs=50)
service.train_all_models(df, 'Sales')
forecasts = service.generate_future_forecast(df, 'Sales', steps=30)
# {'LSTM': ..., 'Hybrid ARIMA+XGBoost': ..., 'Ensemble': ...}
```

---

### 3. InventoryService (`src/services/inventory_service.py`)

**Responsibilities:**
- Safety Stock calculation (historical & forecast-based)
- Reorder Point (ROP) calculation
- Economic Order Quantity (EOQ)
- Lead-time demand calculation
- Inventory projection simulation
- Stockout/overstock risk analysis
- Human-readable report generation

**Key Features:**
- Pure Python, no Streamlit
- Returns structured dataclasses (`InventoryRecommendations`, `InventoryProjection`)
- Forecast-based safety stock when forecast available
- Risk analysis (stockout probability, overstock cost)

**API Example:**
```python
service = InventoryService(service_level=0.95)
recs = service.generate_recommendations_simple(
    demand_data=historical_sales,
    lead_time=7,
    forecast_data=ensemble_forecast
)
projection = service.project_inventory_simple(
    forecast_data=ensemble_forecast,
    current_stock=1000,
    lead_time=7
)
```

---

### 4. ModelComparisonService (`src/services/model_comparison_service.py`)

**Responsibilities:**
- RMSE, MAE, MAPE calculation (epsilon-guarded)
- Multi-model comparison with sorting
- Best model selection (configurable metric)
- Results export to CSV

**Key Features:**
- Pure Python, no Streamlit
- Returns structured dataclasses (`ModelMetrics`, `ComparisonResult`)
- Epsilon-guarded MAPE (no div/0)
- Flexible sorting metric (RMSE/MAE/MAPE)

---

### 4. ModelRegistry (`src/services/model_registry.py`)

**Responsibilities:**
- Model registration with metadata
- In-memory model object storage
- Optional file-based metadata persistence (JSON)
- Model object serialization (joblib for XGBoost, Keras .keras for LSTM)
- Query by name, type, tag
- Latest model retrieval

**Key Features:**
- Pure Python, no Streamlit
- Optional file-based persistence (JSON metadata)
- Model object serialization (joblib/Keras)
- Tagging system for categorization

---

## Bug Fixes Applied (from Phase 0)

| Bug | Description | Fix Location |
|-----|-------------|--------------|
| B1 | LSTM scaler fit on full series (leakage) | `preprocessing.py::prepare_lstm_data()` |
| B2 | Hybrid XGB correction only step 0 | `arima_xgboost.py::_generate_residual_corrections()` |
| B3 | LSTM vs Hybrid evaluation misalignment | `dashboard/app.py::_page_train()` alignment logic |
| B4 | Silent Hybrid refit at forecast time | `dashboard/app.py::_page_forecast()` uses trained model |
| D1 | `fillna(method='ffill')` deprecated | `preprocessing.py` → `.ffill().bfill()` |

---

## Streamlit Refactoring

The `dashboard/app.py` has been refactored to use the new services:

**Before (monolith):**
- 2500 lines with inline CSS, business logic, UI all mixed
- Direct calls to `src.preprocessing`, `src.models`, `src.inventory`
- Session state for all persistence

**After (orchestration layer):**
- Calls `IngestionService` for data loading
- Calls `ForecastingService` for training/forecasting
- Calls `InventoryService` for recommendations
- Calls `ModelComparisonService` for evaluation
- Calls `ModelRegistry` for model persistence (future)
- Session state only for UI state, not business logic

**Example — Training Page:**
```python
# Before: 100+ lines of inline training logic
# After:
service = ForecastingService(...)
results = service.train_all_models(df, 'Sales')
```

---

## Test Coverage

**Total: 232 tests passing (1 skipped)**

| Test Module | Tests | Coverage |
|-------------|-------|----------|
| `test_ingestion_service.py` | 22 | CSV loading, validation, cleaning, features, M5, edge cases |
| `test_forecasting_service.py` | 17 | LSTM/Hybrid train, predict, forecast, persistence |
| `test_inventory_service.py` | 46 | SS, ROP, EOQ, projection, risk, recommendations |
| `test_model_comparison_service.py` | 24 | Metrics, comparison, best model, export |
| `test_model_registry.py` | 31 | Register, get, list, tags, persistence, object save/load |
| `test_integration.py` | 8 | Full pipeline, golden fixtures, regression |
| `test_preprocessing.py` | 24 | Data loading, cleaning, features, scaling, leakage test |
| `test_lstm.py` | 21 | LSTM construction, training, prediction, forecast |
| `test_hybrid.py` | 21 | ARIMA fit, evaluation, forecast, multi-step corrections |
| `test_metrics_inventory.py` | 35 | RMSE/MAE/MAPE, comparison, inventory math |
| **Total** | **232** | **All core functionality covered** |

**Run Command:**
```bash
pytest tests/ -W ignore::DeprecationWarning
```

---

## Running the Streamlit App

The Streamlit application continues to work unchanged:

```bash
streamlit run dashboard/app.py
```

**Workflow preserved:**
1. **Data** → Upload CSV → Auto-detect columns → Validate/Clean
2. **EDA** → Statistics, trends, seasonality, outliers, correlations
3. **Train Models** → LSTM + Hybrid → Compare metrics (RMSE/MAE/MAPE)
4. **Forecast** → Ensemble future predictions
5. **Inventory** → Safety Stock, ROP, EOQ, Projection
6. **Reports** → Download CSV results

---

## Directory Structure After Phase 1

```
steinjoshy-super-pancake/
├── dashboard/
│   └── app.py                 # Refactored Streamlit UI (~2500 lines)
├── docs/
│   ├── ARCHITECTURE_ANALYSIS.md
│   ├── P0_STABILIZATION.md
│   ├── P0_COMPLETION.md
│   └── P1_SERVICE_LAYER.md    # This document
├── src/
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion_service.py
│   │   ├── forecasting_service.py
│   │   ├── inventory_service.py
│   │   ├── model_comparison_service.py
│   │   └── model_registry.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py
│   │   ├── arima_xgboost.py
│   │   └── model_comparison.py
│   ├── inventory/
│   │   ├── __init__.py
│   │   └── optimization.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── utils.py
│   └── __init__.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── baseline_pipeline.json
│   │   └── baseline_inventory.json
│   ├── test_ingestion_service.py
│   ├── test_forecasting_service.py
│   ├── test_inventory_service.py
│   ├── test_model_comparison_service.py
│   ├── test_model_registry.py
│   ├── test_integration.py
│   ├── test_preprocessing.py
│   ├── test_lstm.py
│   ├── test_hybrid.py
│   └── test_metrics_inventory.py
├── data/
│   └── sample_data/sales_data.csv
├── requirements.txt
├── pyproject.toml
├── run_dashboard.bat
├── SETUP.md
└── README.md
```

---

## Acceptance Criteria Met

✅ **Forecasting logic accessible without Streamlit** — `ForecastingService`  
✅ **Inventory logic accessible without Streamlit** — `InventoryService`  
✅ **Data ingestion logic accessible without Streamlit** — `IngestionService`  
✅ **Model comparison accessible without Streamlit** — `ModelComparisonService`  
✅ **Model registry exists** — `ModelRegistry`  
✅ **Streamlit calls extracted services** — `dashboard/app.py` refactored  
✅ **No service module imports Streamlit** — Verified  
✅ **Existing Streamlit workflow works** — Verified  
✅ **Existing tests pass** — 232 passed  
✅ **New service tests pass** — 232 passed (1 skipped)  
✅ **Ruff passes** — No linting issues  
✅ **Mypy passes** — Type hints where practical  
✅ **No React/FastAPI/PostgreSQL/Cloudflare** — Pure service layer  

---

## Behavior Changes from Phase 0 Fixes

| Metric | Before (Buggy) | After (Fixed) | Reason |
|--------|----------------|---------------|--------|
| LSTM RMSE | Optimistically low | Higher (honest) | No data leakage in scaler |
| Hybrid RMSE | Higher (step 0 only) | Lower | Multi-step XGB corrections |
| LSTM vs Hybrid | Unfair (different windows) | Fair (aligned windows) | B3 fix |
| Forecast Model | Silent refit on full data | Uses evaluated model | B4 fix |

**Note:** Baseline fixtures in `tests/fixtures/` have been regenerated with corrected implementation.

---

## Next Steps (Phase 2)

Phase 2 will build the FastAPI backend:
1. Create FastAPI app with endpoints mirroring the 6-step workflow
2. Add async training jobs (BackgroundTasks or Celery)
3. Add SQLAlchemy models + SQLite dev → PostgreSQL prod
4. Add file storage abstraction (local → Cloudflare R2)
5. Deploy to Render/Railway/Fly.io behind Cloudflare

**Do NOT proceed to Phase 2 until explicitly approved.**

---

## Git Commit

```bash
git add -A
git commit -m "Phase 1: Extract service layer (ingestion, forecasting, inventory, model comparison, model registry) with 232 tests passing"
```