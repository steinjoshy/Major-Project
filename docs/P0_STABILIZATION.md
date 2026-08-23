# Phase 0 Stabilization Plan

**Objective:** Fix 4 correctness bugs, resolve pandas deprecation, align dependencies, add test coverage — **without changing architecture or breaking Streamlit**.

---

## Bug B1: LSTM Data Leakage — Scaler Fit on Full Series

### Location
`src/preprocessing.py:190-208` — `prepare_lstm_data()`

### Current Behavior
```python
def prepare_lstm_data(self, df, sales_col='Sales', seq_length=30):
    data = df[sales_col].values.reshape(-1, 1)
    data_scaled = self.scale_data(data, fit=True)  # FITS ON ENTIRE SERIES
    # ... create sequences from scaled data ...
    return np.array(X), np.array(y)
```
Then `train_test_split_data()` splits sequences chronologically.

### Why Incorrect
- Scaler learns min/max from **test + future** data
- Test sequences are scaled using statistics they shouldn't know
- Inverse transform during evaluation uses contaminated scaler
- Metrics (RMSE/MAE/MAPE) are optimistically biased

### Correct Behavior
1. Split raw series chronologically FIRST (train/test)
2. Fit scaler ONLY on training portion
3. Transform train + test using training-fitted scaler
4. Create sequences from scaled train and scaled test independently
5. Future forecasting uses same training-fitted scaler

### Planned Fix
**Option A — Modify `prepare_lstm_data` signature (preferred):**
```python
def prepare_lstm_data(self, df, sales_col='Sales', seq_length=30, test_size=0.2):
    """Split first, then scale, then sequence. Returns (X_train, X_test, y_train, y_test, scaler)."""
    values = df[sales_col].values.reshape(-1, 1)
    split = int(len(values) * (1 - test_size))
    train_vals, test_vals = values[:split], values[split:]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled = scaler.transform(test_vals)
    
    # Create sequences from each partition
    X_tr, y_tr = _create_sequences(train_scaled, seq_length)
    X_te, y_te = _create_sequences(test_scaled, seq_length)
    
    return X_tr, X_te, y_tr, y_te, scaler
```

**Option B — Add new method, deprecate old:**
- Add `prepare_lstm_data_no_leakage()` with correct behavior
- Update dashboard to use new method
- Keep old for backward compat (marked deprecated)

**Decision:** Option A — single source of truth. Update all callers.

### Expected Impact
- Test metrics will likely **increase** (worse) — this is correct
- Training may be slightly harder (less data for scaler fitting)
- Forecast scale may shift slightly

### Tests Required
- `test_scaler_not_fit_on_test_data()` — verify scaler min/max from train only
- `test_inverse_transform_consistency()` — round-trip train → scale → inverse = original
- `test_forecast_uses_training_scaler()` — future forecast uses same scaler

---

## Bug B2: Hybrid XGBoost Correction Only Step 0

### Location
`src/models/arima_xgboost.py:107-146` — `evaluate_on_test()`  
`src/models/arima_xgboost.py:152-177` — `forecast_future()`

### Current Behavior
```python
def evaluate_on_test(self, train_data, test_data):
    test_len = len(test_data)
    arima_forecast = self.arima_model.get_forecast(steps=test_len).predicted_mean
    
    xgb_corrections = np.zeros(test_len)
    if self.xgb_trained:
        recent_residuals = self._residuals[-self.residual_lags:]
        if len(recent_residuals) == self.residual_lags:
            xgb_corrections[0] = self.xgb_model.predict(...)[0]  # ONLY INDEX 0
            # Rest remain zero
    
    return arima_forecast + xgb_corrections
```

### Why Incorrect
- Hybrid methodology promises: ARIMA forecast + XGBoost residual correction for **each step**
- XGBoost trained on lagged residuals → should predict next residual given previous residuals
- Current code predicts one correction, applies to step 0, zeros for steps 1..N
- Result: Hybrid = ARIMA + tiny one-step tweak → defeats the purpose

### Correct Behavior
**Rolling residual correction:**
1. Start with last `residual_lags` residuals from training
2. For each forecast step `t`:
   - XGBoost predicts residual `r_t` given last `residual_lags` residuals
   - Hybrid prediction = ARIMA_forecast[t] + r_t
   - Append `r_t` to residual history, drop oldest, repeat
2. This is autoregressive on the residual space

### Planned Fix
```python
def _generate_residual_corrections(self, steps, initial_residuals):
    """Generate XGBoost corrections for multiple steps autoregressively."""
    corrections = np.zeros(steps)
    residual_window = initial_residuals[-self.residual_lags:].copy()
    
    for i in range(steps):
        if len(residual_window) == self.residual_lags:
            try:
                corr = self.xgb_model.predict(
                    self.residual_scaler.transform(residual_window.reshape(1, -1))
                )[0]
                corrections[i] = corr
                # Roll window
                residual_window = np.append(residual_window[1:], corr)
            except Exception:
                corrections[i] = 0.0
        else:
            corrections[i] = 0.0
    
    return corrections

def evaluate_on_test(self, train_data, test_data):
    # ... ARIMA forecast ...
    xgb_corrections = self._generate_residual_corrections(
        len(test_data), self._residuals
    )
    return arima_forecast + xgb_corrections

def forecast_future(self, steps=30):
    # ... ARIMA forecast ...
    xgb_corrections = self._generate_residual_corrections(
        steps, self._residuals
    )
    return arima_forecast + xgb_corrections
```

### Expected Impact
- Hybrid forecasts now genuinely multi-step corrected
- Test metrics for Hybrid should **improve** (lower RMSE/MAE)
- Forecast variability increases (more realistic)

### Tests Required
- `test_hybrid_corrections_all_steps_nonzero()` — corrections array not all zeros after index 0
- `test_hybrid_forecast_length()` — output length == requested steps
- `test_residual_autoregression_consistency()` — manual roll matches model output

---

## Bug B3: LSTM vs Hybrid Evaluation Window Misalignment

### Location
`dashboard/app.py:1799-1832` — `_page_train()`

### Current Behavior
```python
# LSTM: sequences created from full scaled series, then split
X, y = preprocessor.prepare_lstm_data(df, sc, seq_length)
X_tr, X_te, y_tr, y_te = preprocessor.train_test_split_data(X, y, test_size=0.2)

# Hybrid: raw series split directly
train_series, test_series, _ = preprocessor.prepare_hybrid_data(df, sc, test_size=0.2)

# ... train both ...

# Alignment hack
min_len = min(len(lstm_preds), len(hybrid_preds), len(y_test_actual))
lstm_preds   = lstm_preds[:min_len]
hybrid_preds = hybrid_preds[:min_len]
y_test_actual = y_test_actual[:min_len]
```

### Why Incorrect
- LSTM sequences: each sample uses `seq_length` prior points → first test sample corresponds to date at `train_size + seq_length`
- Hybrid raw split: test series starts at `train_size` (no sequence warm-up)
- `min_len` truncation compares **different date ranges**:
  - LSTM: dates[train_size + seq_length : train_size + seq_length + min_len]
  - Hybrid: dates[train_size : train_size + min_len]
- Metrics not comparable

### Correct Behavior
Both models evaluated on **exact same actual-demand dates**.

**Approach:** Align Hybrid test start to match LSTM test start date.
- LSTM test indices: `split` to `len(X)` where `split = int(len(X) * 0.8)`
- Corresponding raw dates: `split + seq_length` to `len(df)`
- Hybrid test series should be `df[split + seq_length:]`

### Planned Fix
In `_page_train()`:
```python
# LSTM prep (unchanged)
X, y = preprocessor.prepare_lstm_data(df, sc, seq_length)
X_tr, X_te, y_tr, y_te = preprocessor.train_test_split_data(X, y, test_size=0.2)
lstm_test_start_idx = preprocessor.train_size + seq_length

# Hybrid prep — align test start to LSTM
values = df[sc].values.astype(float)
train_series = values[:lstm_test_start_idx]
test_series = values[lstm_test_start_idx:]

hybrid = HybridArimaXGBoost(arima_order=arima_tuple)
hybrid.fit(train_series)
hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

# LSTM eval
lstm_preds_scaled = lstm.predict(X_te)
lstm_preds = preprocessor.inverse_scale(lstm_preds_scaled).flatten()
y_test_actual = preprocessor.inverse_scale(y_te.reshape(-1, 1)).flatten()

# Now lengths should match naturally (or be off by ≤1 due to rounding)
# No min_len truncation needed
```

Also update `DataPreprocessor.prepare_hybrid_data()` to accept explicit `split_idx` for alignment.

### Expected Impact
- Fair comparison: same dates, same number of points
- Hybrid test set slightly smaller (loses first `seq_length` test points)
- Metrics comparable; no silent truncation

### Tests Required
- `test_evaluation_dates_align()` — LSTM and Hybrid test predictions correspond to same date indices
- `test_no_silent_truncation()` — no `min()` truncation in comparison logic

---

## Bug B4: Train/Future Forecast Divergence

### Location
`dashboard/app.py:2038-2044` — `_page_forecast()`

### Current Behavior
```python
# During training: hybrid fitted on train_series (80%)
# During forecast: NEW Hybrid fitted on FULL data silently
hybrid_full = HybridArimaXGBoost(arima_order=arima_tuple)
hybrid_full.fit(df[sc].values.astype(float))  # RE-FIT ON ALL DATA
hybrid_fc = hybrid_full.forecast_future(steps=forecast_steps)
```

### Why Incorrect
- Evaluated model: trained on 80%, tested on 20%
- Deployed forecast model: trained on 100%
- Different model parameters → different forecasts
- User sees metrics for Model A, gets forecasts from Model B
- Silent, no UI indication

### Correct Behavior
**Explicit two-phase workflow:**
1. **Evaluation phase** — train on train split, evaluate on test split, select best model
2. **Production phase** — retrain selected model on **all available data**, then forecast
3. UI must make this explicit: "Retraining selected model on full dataset for production forecast"

### Planned Fix
Add method to `HybridArimaXGBoost` for retraining on full data, and update forecast page:

```python
# In HybridArimaXGBoost
def refit_full(self, full_data):
    """Explicitly retrain on full dataset for production forecasting."""
    return self.fit(full_data)  # fit() already handles full pipeline

# In _page_forecast()
status.info('Retraining Hybrid on full dataset for production forecast...')
prog.progress(50)
hybrid_obj = st.session_state.predictions['hybrid_obj']
hybrid_obj.refit_full(df[sc].values.astype(float))
hybrid_fc = hybrid_obj.forecast_future(steps=forecast_steps)
```

Same for LSTM — retrain on full data (or at least use the already-trained model which saw 80%; for true production, retrain on 100%).

**Simpler approach for P0:** Use the already-trained objects for forecasting (they're in session_state). Only retrain if explicitly requested. This avoids the silent divergence.

### Expected Impact
- Forecasts now come from the **same model** that was evaluated
- Slightly different forecasts (trained on 80% vs 100%) — but transparent
- Future: add "Retrain on Full Data" button in UI

### Tests Required
- `test_forecast_uses_evaluated_model()` — forecast model parameters match evaluated model
- `test_explicit_refit_option()` — optional refit produces different (documented) model

---

## Bug D1: pandas fillna Deprecation

### Location
`src/preprocessing.py:129` — `clean_data()`

### Current Behavior
```python
df[sales_col] = df[sales_col].fillna(method='ffill').fillna(method='bfill')
```

### Issue
`method` parameter deprecated since pandas 2.1.0, removed in 3.0.

### Fix
```python
df[sales_col] = df[sales_col].ffill().bfill()
```

### Tests Required
- `test_fillna_no_deprecation_warning()` — run with `-W error::FutureWarning`

---

## Dependency Stabilization (D2)

### Actions
1. **Audit actual working versions** — run `pip freeze` in venv
2. **Update `requirements.txt`** to match verified working versions
3. **Update `SETUP.md`** to match `requirements.txt` (remove fictional versions)
4. **Add `pyproject.toml`** with `[project]` and `[tool.pytest.ini_options]`
5. **Pin TensorFlow to 2.16.2** (last supporting Python 3.12)

### Expected Versions (from venv)
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
scipy==1.13.1
duckdb==1.1.3
```

---

## Safe Technical Debt Cleanup

| Item | Location | Fix |
|------|----------|-----|
| Undefined `.sb-z-box` CSS | `app.py:817` | Add `.sb-z-box { font-family: monospace; padding: 2px 6px; background: #1e293b; color: #93c5fd; border-radius: 4px; }` |
| Dead `lstm_status`/`hyb_status` vars | `app.py:1722-1725` | Remove |
| `'inv_proj' in dir()` | `app.py:2323` | Use `locals().get('inv_proj')` or flag variable |
| Global `warnings.filterwarnings('ignore')` | `arima_xgboost.py:8`, `lstm_model.py:9` | Remove or scope to specific warnings |
| Hard-coded path in `run_dashboard.bat` | `run_dashboard.bat:11` | Use `%~dp0` for script directory |
| README nonexistent refs | `README.md` | Remove references to `notebooks/exploratory.ipynb` and `data/README.md` |
| Untracked empty favorita dir | `data/datasets/` | Remove or add `.gitkeep` |
| EOQ unused in UI | `optimization.py:88`, `app.py` | Add optional holding/ordering cost inputs to Inventory page (or remove EOQ from API if not needed) |

---

## Implementation Order

1. **Create test harness** (tests/ directory, pytest config)
2. **Generate golden fixtures** — run current code, save outputs
3. **Fix B1** — LSTM scaler (preprocessing.py + dashboard)
4. **Fix B3** — Evaluation alignment (dashboard + preprocessing)
5. **Fix B2** — Hybrid multi-step correction (arima_xgboost.py)
6. **Fix B4** — Train/forecast divergence (dashboard + model)
7. **Fix D1** — pandas fillna (preprocessing.py)
8. **Dependency audit** — requirements.txt, SETUP.md, pyproject.toml
9. **Safe cleanup** — CSS, dead code, warnings, paths, README
10. **Full regression test** — pytest + Streamlit smoke test

---

## Acceptance Criteria

- [ ] All pytest tests pass
- [ ] Golden fixtures match (or documented expected changes)
- [ ] Streamlit: upload → validate → clean → EDA → train → compare → forecast → inventory → reports — all pages work
- [ ] No deprecation warnings on pandas operations
- [ ] requirements.txt matches venv; SETUP.md matches requirements.txt
- [ ] Git commit with message: "Phase 0: Fix B1-B4, D1, dependency alignment, safe cleanup"