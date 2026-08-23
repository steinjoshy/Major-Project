# AI-Based Demand Forecasting System - Setup & Installation Guide

## Quick Start

### 1. Clone & Navigate
```bash
cd steinjoshy-super-pancake
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will start at `http://localhost:8501`

---

## System Requirements

- **Python**: 3.12 or 3.11 (Note: Python 3.14+ may have compatibility issues with TensorFlow)
- **OS**: Windows, macOS, Linux
- **RAM**: 4GB+ recommended (TensorFlow requires significant memory)
- **Disk**: ~2GB for dependencies

---

## Tested Versions

These versions are known to work together (from `requirements.txt`):

```
Python 3.12.10
TensorFlow 2.16.2
NumPy 1.26.4
Pandas 2.2.3
Scikit-learn 1.5.2
XGBoost 2.1.1
Streamlit 1.38.0
SciPy 1.13.1
Statsmodels 0.14.4
Plotly 5.24.1
Matplotlib 3.9.2
Seaborn 0.13.2
DuckDB 1.1.3
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution**: Run `pip install tensorflow==2.16.2`

### Issue: `AttributeError: module 'numpy' has no attribute 'long'`
**Solution**: This occurs with numpy 2.x + scipy 1.18.x incompatibility
- Fix: Install scipy 1.13.1 with numpy 1.26.4
```bash
pip install scipy==1.13.1
pip install numpy==1.26.4 --force-reinstall
```

### Issue: Streamlit won't start
**Solution**: Ensure you're running with `streamlit run` (not regular python):
```bash
streamlit run dashboard/app.py --logger.level=warning
```

### Issue: TensorFlow initialization takes a long time
**Solution**: This is normal - TensorFlow initializes oneDNN operations on first import. Wait 30-60 seconds.

---

## Project Structure

```
├── src/
│   ├── preprocessing.py       # Data cleaning & feature engineering
│   ├── eda.py                 # Exploratory data analysis
│   ├── models/
│   │   ├── lstm_model.py      # LSTM forecasting model
│   │   ├── arima_xgboost.py   # Hybrid ARIMA+XGBoost model
│   │   └── model_comparison.py # Metrics (RMSE, MAE, MAPE)
│   ├── inventory/
│   │   └── optimization.py     # Safety Stock & Reorder Point
│   └── utils.py               # Utilities
├── dashboard/
│   └── app.py                 # Streamlit web interface
├── data/
│   └── sample_data/
│       └── sales_data.csv     # Sample dataset
├── requirements.txt
├── README.md
└── SETUP.md (this file)
```

---

## Dashboard Features

The Streamlit dashboard provides 5 main tabs:

1. **📤 Data Upload** - Upload CSV with columns: Date, Quantity
2. **📊 EDA Analysis** - Visualize trends, seasonality, distributions
3. **🤖 Model Training** - Train LSTM & Hybrid ARIMA+XGBoost models
4. **📈 Forecasts** - Compare predictions & view confidence intervals
5. **📦 Inventory Optimization** - Safety Stock, Reorder Point, EOQ

---

## Sample Data Format

```csv
Date,Product_ID,Quantity,Price,Store_ID
2023-01-01,P001,100,29.99,S001
2023-01-02,P001,105,29.99,S001
2023-01-03,P001,98,29.99,S001
```

Minimum required columns: `Date`, `Quantity`

---

## Model Details

### LSTM Model
- 2 LSTM layers (64 → 32 units)
- Dropout regularization (0.2)
- Adam optimizer
- Sequences: 30-day windows
- Forecasting: 7-30 days ahead

### Hybrid ARIMA + XGBoost
- ARIMA captures linear trends & seasonality
- XGBoost corrects residual errors
- Combined forecast = ARIMA output + XGBoost correction

### Evaluation Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

---

## Notes

- First run of LSTM training may take 2-5 minutes (TensorFlow compilation)
- Sample dataset has ~100 records - use production data (500+ records) for better models
- Forecasts are more accurate with seasonal data (1+ year of history)

---

## Support

For issues, check the README.md for detailed documentation or review the module docstrings.
