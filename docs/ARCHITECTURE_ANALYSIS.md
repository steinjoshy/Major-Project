# Architecture Analysis — AI Demand Forecasting & Inventory Optimization

**Repository:** steinjoshy-super-pancake  
**Branch:** steinjoshy-ai-demand-forecasting  
**Date:** 2026-08-21  
**Phase:** 0 Stabilization (Pre-Migration Audit)

---

## 1. Current Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Monolith (~2500 LOC)              │
├─────────────────────────────────────────────────────────────────┤
│  Pages: Dashboard | Data | EDA | Train | Compare | Forecast    │
│         | Inventory | Reports | Settings                        │
├─────────────────────────────────────────────────────────────────┤
│  Inline CSS (~500 lines)  │  Session State (all persistence)   │
├─────────────────────────────────────────────────────────────────┤
│  src/                                                     ┌────┐  │
│  ├── preprocessing.py          Data pipeline            │    │  │
│  ├── eda.py                    EDA visualizations       │    │  │
│  ├── utils.py                  Helpers                  │    │  │
│  ├── models/                                                          
│  │   ├── lstm_model.py         LSTM (TF/Keras)          │ ML │  │
│  │   ├── arima_xgboost.py      Hybrid ARIMA + XGB       │Lib │  │
│  │   └── model_comparison.py   RMSE/MAE/MAPE            │    │  │
│  └── inventory/                                                          
│      └── optimization.py       SS/ROP/EOQ/Projection    │Inv │  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    data/sample_data/sales_data.csv (730 rows)
                    No database, no model persistence, no CI/CD
```

### 1.2 Technology Stack

| Layer | Technology | Version (pinned) | Version (SETUP.md claimed) |
|-------|------------|------------------|----------------------------|
| Python | CPython | 3.12.10 (venv) | 3.12.x |
| Deep Learning | TensorFlow / Keras | 2.16.2 | 2.16.2 (Keras 3.15.0) |
| Gradient Boosting | XGBoost | 2.1.1 | 3.3.0 |
| Statistics | statsmodels | 0.14.4 | — |
| Data | pandas, numpy | 2.2.3 / 1.26.4 | 3.0.3 / 1.26.4 |
| ML Utils | scikit-learn | 1.5.2 | 1.9.0 |
| Web UI | Streamlit | 1.38.0 | 1.60.0 |
| Viz | Plotly, Matplotlib, Seaborn | 5.24.1 / 3.9.2 / 0.13.2 | — |
| DB/Storage | duckdb (optional) | 1.1.3 | — |

**Critical mismatch:** `requirements.txt` pins conflict with `SETUP.md` documented versions (pandas 3.x doesn't exist; scikit-learn 1.9.0 doesn't exist; XGBoost 3.3.0 doesn't exist; Streamlit 1.60.0 doesn't exist as of 2026-08-21).

---

## 2. Current Data Flow

```
CSV Upload / Local Folder
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ _page_data() — dashboard/app.py:1226                         │
│ • Auto-detect date/demand columns via alias lists            │
│ • M5 wide-format detection (d_* columns → melt + sum)        │
│ • Multi-file concat on selected columns                      │
│ • DuckDB SQL aggregation for >80MB files (temp file)         │
│ • Chunked pandas fallback (250k rows/chunk, cached)          │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ DataPreprocessor.clean_data() — src/preprocessing.py:94      │
│ • Parse dates, coerce numeric                                │
│ • Deduplicate by date (sum)                                  │
│ • Sort by date                                               │
│ • fillna(method='ffill') → DEPRECATED in pandas ≥3          │
│ • Drop negative sales                                        │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ Feature Engineering (called by EDA, not by models directly)  │
│ • Time features: Year, Month, Week, Day, DOW, Quarter, Wkend│
│ • Lag features: 1, 7, 14, 30 (dropped NaN)                  │
│ • Rolling mean/std: 7, 14, 30                                │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ Train/Test Split (chronological, 80/20, NO shuffling)        │
│ • LSTM: prepare_lstm_data() → sequences → train_test_split() │
│ • Hybrid: prepare_hybrid_data() → raw series split           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Current ML Pipeline

### 3.1 LSTM Forecaster (`src/models/lstm_model.py`)

**Architecture:**
```
Input (seq_length=30, 1 feature)
    │
    ▼
LSTM(64, tanh, return_sequences=True) → Dropout(0.2)
    │
    ▼
LSTM(32, tanh) → Dropout(0.2)
    │
    ▼
Dense(16, ReLU)
    │
    ▼
Dense(1)  →  Output (scaled prediction)
```

**Training:**
- Optimizer: Adam(lr=0.001)
- Loss: MSE
- EarlyStopping: monitor=val_loss (patience=8, restore_best_weights=True)
- Validation data = test split (mild leakage into early stopping)

**Forecasting:**
- Autoregressive: feed prediction back as next input
- `forecast_future(last_scaled_sequence, steps, scaler)` → inverse transform if scaler provided

### 3.2 Hybrid ARIMA + XGBoost (`src/models/arima_xgboost.py`)

**Methodology (per docstring):**
1. Fit ARIMA on training data → captures linear trends/seasonality
2. Compute residuals = Actual − ARIMA in-sample fitted values
3. Train XGBoost on lagged residuals (10 lags) → learns nonlinear error patterns
4. Hybrid Forecast = ARIMA_forecast + XGBoost_residual_correction

**ARIMA Fallback Chain:** `(1,1,1) → (0,1,1) → (1,1,0) → (0,1,0) → (1,0,0)`

**XGBoost Config:** n_estimators=100, lr=0.05, max_depth=5, random_state=42

### 3.3 Model Evaluation (`src/models/model_comparison.py`)

**Metrics:**
- RMSE: √mean((y_true - y_pred)²) — penalizes large errors
- MAE: mean(|y_true - y_pred|) — average absolute deviation
- MAPE: mean(|(y_true - y_pred) / (|y_true| + ε)|) × 100 — ε=1e-8 guards div/0

**Comparison:** `compare_models()` returns DataFrame sorted by RMSE ascending; `get_best_model()` returns best model + improvement % over worst.

---

## 4. Current Inventory Optimization Pipeline

### 4.1 Safety Stock (`src/inventory/optimization.py:60`)

```
SS = Z × σ_demand × √(Lead Time)
```

- **Z-score:** `scipy.stats.norm.ppf(service_level)` (e.g., 1.645 for 95%)
- **σ_demand:** 
  - Historical: `std(demand_data)` 
  - **Forecast-based (preferred when forecast exists):** `std(forecast_array)`
- **Lead Time:** User input (sidebar, 1–90 days, default 7)

### 4.2 Reorder Point (`:74`)

```
ROP = (μ_demand × Lead Time) + Safety Stock
```

### 4.3 Economic Order Quantity (`:88`)

```
EOQ = √(2 × D × S / H)
```
- D = Annual demand, S = Ordering cost, H = Holding cost/unit/year
- **Not exposed in UI** — API exists but no input fields

### 4.4 Inventory Projection (`:189`)

Day-by-day simulation:
- Start with `current_stock`
- For each forecast day: receive pending orders → check ROP → place order (arrives after lead_time) → consume demand (floor at 0)
- Order quantity = max(30-day avg demand, 2×ROP)

---

## 5. Important Files

| File | Purpose | Lines | Key Issues |
|------|---------|-------|------------|
| `dashboard/app.py` | Streamlit monolith (9 pages, CSS, orchestration) | 2496 | God object, tight coupling, no persistence |
| `src/preprocessing.py` | Data pipeline, LSTM prep, Hybrid prep | 259 | **Scaler fit on full series (leakage)** |
| `src/models/lstm_model.py` | LSTM model, training, forecasting | 173 | Val data = test split; no persistence used |
| `src/models/arima_xgboost.py` | Hybrid ARIMA+XGB | 212 | **XGB correction only step 0**; silent refit |
| `src/models/model_comparison.py` | Metrics, comparison | 129 | Clean |
| `src/inventory/optimization.py` | SS, ROP, EOQ, projection | 291 | Clean math; EOQ unused in UI |
| `src/eda.py` | EDA visualizations | 242 | ACF/PACF uses matplotlib (not Plotly) |
| `src/utils.py` | Helpers | 68 | Minimal use |
| `requirements.txt` | Pinned dependencies | 16 | Conflicts with SETUP.md |
| `SETUP.md` | Installation guide | 170 | Version claims unrealistic |
| `data/sample_data/sales_data.csv` | Sample dataset | 730 rows | Date, Sales, Price, Promotion |

---

## 6. Dependencies

**Actual (requirements.txt):**
```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
tensorflow==2.16.2
xgboost==2.1.1
statsmodels==0.14.4
streamlit==1.38.0
plotly==5.24.1
matplotlib==3.9.2
seaborn==0.13.2
python-dateutil==2.9.0.post0
pytz==2024.2
openpyxl==3.1.5
scipy==1.13.1
pyarrow==17.0.0
duckdb==1.1.3
```

**Issues:**
- `fillna(method='ffill')` deprecated since pandas 2.1.0, removed in 3.0 → use `.ffill()`
- No `pyproject.toml`, no `pip-tools`, no `pip-audit`
- TensorFlow 2.16.2 is last TF 2.x; TF 2.17+ drops Python 3.12 support

---

## 7. Problems & Technical Debt

### 7.1 Critical Correctness Bugs

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| **B1** | **LSTM scaler fit on full series before split** | `preprocessing.py:200-201` | Data leakage → overoptimistic metrics |
| **B2** | **Hybrid XGB correction only step 0** | `arima_xgboost.py:131-145` | Hybrid = ARIMA after step 1 |
| **B3** | **LSTM vs Hybrid evaluation windows misaligned** | `app.py:1828-1832` | Different date ranges compared via `min_len` truncation |
| **B4** | **Silent Hybrid refit on full data at forecast time** | `app.py:2040-2044` | Evaluated model ≠ deployed model |

### 7.2 Dependency & Compatibility

| # | Issue | Location |
|---|-------|----------|
| D1 | `fillna(method='ffill')` deprecated | `preprocessing.py:129` |
| D2 | `requirements.txt` vs `SETUP.md` version mismatch | Both files |
| D3 | No `pyproject.toml`, no lock file | Root |
| D4 | TensorFlow 2.16.2 is end-of-line for Python 3.12 | requirements.txt |

### 7.3 Architecture & Maintainability

| # | Issue | Location |
|---|-------|----------|
| A1 | 2500-line God object (`app.py`) | `dashboard/app.py` |
| A2 | ~500 lines inline CSS in Python string | `dashboard/app.py:50-514` |
| A3 | All state in `st.session_state` — no persistence | `dashboard/app.py` |
| A4 | Models stored in session_state → per-tab memory duplication | `app.py:1849-1856` |
| A5 | No model persistence despite `save_model()`/`load_model()` existing | `lstm_model.py:161-169` |
| A6 | No tests beyond 2 smoke scripts | `test_imports.py`, `test_setup.py` |
| A7 | No CI/CD, no linting, no type hints | Root |
| A8 | Hard-coded local path in `run_dashboard.bat:11` | `run_dashboard.bat` |
| A9 | Undefined CSS class `.sb-z-box` used in sidebar | `app.py:817` |
| A10 | Dead code: `lstm_status`, `hyb_status` vars | `app.py:1722-1725` |
| A11 | Fragile `'inv_proj' in dir()` check | `app.py:2323` |
| A12 | Global `warnings.filterwarnings('ignore')` in modules | `arima_xgboost.py:8`, `lstm_model.py:9` |
| A13 | README references nonexistent `notebooks/exploratory.ipynb`, `data/README.md` | `README.md` |
| A14 | Untracked empty `data/datasets/favorita-grocery-sales-forecasting/` | `data/datasets/` |
| A15 | EOQ API exists but no UI exposure | `optimization.py:88`, `app.py` |

---

## 8. Recommended Target Architecture (Post-Migration)

> **Note:** This is the Phase 1+ target. Phase 0 makes NO architectural changes.

```
┌────────────────────────────────────────────────────────────────────┐
│                     Cloudflare (DNS · SSL · WAF · R2)             │
└────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │ Next.js/React │               │  FastAPI      │
            │ Frontend      │◄──── REST ────►│  Backend      │
            │ (Vercel)      │               │  (Render/     │
            └───────────────┘               │   Railway/    │
                                            │   Fly.io)     │
                                            └───────┬───────┘
                                                    │
                              ┌─────────────────────┼─────────────────────┐
                              ▼                     ▼                     ▼
                       ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
                       │PostgreSQL   │      │ Cloudflare  │      │  Job Queue  │
                       │ (datasets,  │      │ R2 (model   │      │ (training,  │
                       │  runs,      │      │  artifacts, │      │  forecasts) │
                       │  forecasts) │      │  datasets)  │      │             │
                       └─────────────┘      └─────────────┘      └─────────────┘
```

**Backend hosts Python-capable services** (Render, Railway, Fly.io all support background workers for long-running training). Cloudflare provides DNS/SSL/WAF/R2 but **cannot run Python application servers** (Workers Python is too limited for TF/XGB).

**Migration strategy: Strangler Fig** — Streamlit remains runnable at every phase.

---

## 9. Migration Phases

### Phase 0 — Stabilize (THIS PHASE)
- Fix 4 correctness bugs (B1–B4)
- Fix pandas deprecation (D1)
- Align dependency versions (D2)
- Create pytest test harness + golden fixtures
- Clean safe technical debt (A9–A12, A15)
- **Zero architectural changes** — Streamlit must remain fully functional

### Phase 1 — Extract Service Layer
- Pure-Python domain services: `ForecastingService`, `InventoryService`, `IngestionService`
- Model registry with persistence (joblib for XGB/Sklearn, Keras `.keras` for LSTM)
- Streamlit refactored to call services — behavior identical
- Contract tests on golden fixtures

### Phase 2 — FastAPI Backend
- REST endpoints mirroring 6-step workflow: upload → preprocess → train → compare → forecast → inventory
- Async training jobs (Celery/RQ or FastAPI BackgroundTasks)
- SQLAlchemy models; SQLite dev → PostgreSQL prod
- File storage abstraction (local → R2)

### Phase 3 — Persistence & Database
- PostgreSQL schema: datasets, runs, metrics, forecasts, inventory_params
- R2 for model artifacts and large datasets
- Alembic migrations

### Phase 4 — Next.js Frontend
- Rebuild 9 pages against API
- Streamlit kept in parallel until feature parity
- Shared TypeScript types from OpenAPI spec

### Phase 5 — Infrastructure & Cutover
- Cloudflare DNS/SSL/WAF/R2 configuration
- Backend deploy (Render/Railway/Fly.io)
- Domain cutover, Streamlit deprecation

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| TF model size/startup latency in containers | High | Medium | CPU-only TF; consider ONNX export in P2 |
| Training time > HTTP timeout | High | High | Async job queue from P2; progress polling |
| Metric regression after bug fixes | Certain | High | Golden fixtures + contract tests in P0 |
| Feature parity drift (Streamlit vs Next.js) | Medium | High | Automated E2E tests against API; manual checklist |
| Cloudflare R2 cost at scale | Low | Low | Free tier generous; monitor |
| XGBoost/TF version compatibility | Medium | Medium | Pin exact versions; test matrix in CI |

---

## 11. Recommended First Implementation Step

**Phase 0, Step 1:** Create `docs/ARCHITECTURE_ANALYSIS.md` (this document) and `docs/P0_STABILIZATION.md` with detailed bug fix specifications.

**Phase 0, Step 2:** Create `tests/` directory with pytest harness covering:
- Data pipeline (load, validate, clean, split)
- LSTM (sequence gen, scaling, train, predict, forecast)
- Hybrid (ARIMA fit, residuals, XGB correction, forecast)
- Metrics (RMSE, MAE, MAPE, edge cases)
- Inventory (SS, ROP, EOQ, projection)

**Phase 0, Step 3:** Run current implementation on sample data → save baseline outputs to `tests/fixtures/`.

**Phase 0, Step 4–7:** Implement fixes B1–B4 with regression tests proving corrected behavior.

**Phase 0, Step 8–9:** Dependency alignment + safe cleanup.

**Phase 0, Step 10:** Full Streamlit smoke test.

**Phase 0, Step 11:** `P0_COMPLETION.md` + Git commit.

**Do not proceed to Phase 1 until all Phase 0 tests pass and Streamlit verified.**