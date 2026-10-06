# AI-Based Demand Forecasting for Inventory Optimization

An intelligent demand forecasting and inventory optimization system using advanced AI techniques including LSTM deep learning and Hybrid ARIMA + XGBoost models.

## 🎯 Overview

This project develops a comprehensive system that combines multiple forecasting approaches to:
- **Predict future demand** with high accuracy
- **Compare forecasting models** using RMSE, MAE, and MAPE metrics
- **Optimize inventory levels** through Safety Stock and Reorder Point calculations
- **Provide real-time insights** via an interactive Streamlit dashboard

## 🚀 Features

### Forecasting Models
- **LSTM (Long Short-Term Memory)**: Deep learning model for capturing sequential patterns, seasonality, and long-term dependencies
- **Hybrid ARIMA + XGBoost**: Combines ARIMA for linear trends with XGBoost for nonlinear error correction

### Data Processing
- Automated data cleaning (missing values, duplicates, formatting)
- Time-based feature engineering (day, week, month, quarter)
- Lag feature creation for temporal patterns
- Data scaling for neural networks
- 80-20 train-test split with **no data leakage**

### Exploratory Data Analysis (EDA)
- Time-series trend visualization
- Seasonal pattern analysis
- Correlation heatmaps
- Distribution analysis
- Outlier detection
- Moving average analysis

### Model Evaluation
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **MAE** (Mean Absolute Error): Average absolute difference
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error metric

### Inventory Optimization
- **Safety Stock Calculation**: Extra inventory to handle demand uncertainty
- **Reorder Point**: Optimal inventory level to trigger new orders
- **Economic Order Quantity (EOQ)**: Cost-optimized order size
- **Inventory Forecasting**: Projected stock levels based on demand forecast

### Interactive Dashboard
- CSV dataset upload (single or multi-file)
- Data preprocessing and validation
- Model training interface
- Real-time visualization
- Forecast comparison charts
- Inventory optimization recommendations
- Results download capability

## 📋 Project Structure

```
demand-forecasting/
├── .streamlit/
│   └── config.toml              # Streamlit configuration for cloud deployment
├── data/
│   ├── sample_data/
│   │   └── sales_data.csv       # Sample dataset for testing
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Data pipeline (leakage-free)
│   ├── eda.py                   # Exploratory analysis
│   ├── utils.py                 # Helper functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py       # LSTM implementation
│   │   ├── arima_xgboost.py    # Hybrid model
│   │   └── model_comparison.py # Evaluation metrics
│   ├── inventory/
│   │   ├── __init__.py
│   │   └── optimization.py     # Inventory calculations
│   └── services/                # Service layer (optional backend)
├── dashboard/
│   └── app.py                   # Streamlit web interface (entry point)
├── tests/                       # Unit and integration tests
├── requirements.txt             # Python dependencies (pinned)
├── pyproject.toml              # Project metadata
├── run_dashboard.bat           # Windows launcher script
├── test_imports.py             # Import validation script
├── test_setup.py               # Setup validation script
└── README.md                    # This file
```

## 🔧 Installation

### Prerequisites
- **Python 3.10, 3.11, or 3.12** (3.13+ not yet supported by TensorFlow)
- **pip** (Python package manager)
- **Git** (for cloning)

### Windows CMD Local Setup

```cmd
:: 1. Clone the repository
git clone <repository-url>
cd demand-forecasting

:: 2. Create virtual environment (recommended)
py -3.12 -m venv venv

:: 3. Activate virtual environment
venv\Scripts\activate

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install dependencies
pip install -r requirements.txt

:: 6. Verify installation
python test_setup.py

:: 7. Run the dashboard
streamlit run dashboard/app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Alternative: Using the Launcher Script (Windows)

```cmd
:: Double-click run_dashboard.bat or run from CMD:
run_dashboard.bat
```

The launcher script:
- Checks for virtual environment
- Validates all dependencies and project imports
- Starts Streamlit on port 8501

## ☁️ Streamlit Community Cloud Deployment

### Prerequisites
- GitHub account
- Repository pushed to GitHub (public or private)

### Deployment Steps

1. **Push to GitHub**
   ```cmd
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **"New app"**
   - Connect your GitHub account if not already connected
   - Select your repository: `your-username/demand-forecasting`
   - Branch: `main`
   - **Main file path**: `dashboard/app.py`
   - Click **"Deploy!"**

3. **Configuration** (automatic from `.streamlit/config.toml`)
   - Port: 8501
   - Headless mode: enabled
   - CORS/XSRF: disabled for cloud compatibility
   - Theme: configured via config.toml

4. **Python Version**
   - Streamlit Cloud auto-detects from `pyproject.toml` (`requires-python = ">=3.10,<3.13"`)
   - Uses Python 3.12 by default (compatible with TensorFlow 2.16+)

### Important Notes for Cloud Deployment

| Requirement | How It's Handled |
|-------------|------------------|
| Entry point | `dashboard/app.py` (configured in deploy UI) |
| Dependencies | `requirements.txt` at repo root |
| Config | `.streamlit/config.toml` at repo root |
| Python version | `pyproject.toml` specifies `>=3.10,<3.13` |
| Sample data | `data/sample_data/sales_data.csv` included in repo |
| No Docker/Node.js | Pure Python + Streamlit |

**No additional configuration needed** - the app is ready for Streamlit Community Cloud as-is.

## 📊 Usage

### Dashboard Navigation

The sidebar provides access to all pages:

| Page | Purpose |
|------|---------|
| **Dashboard** | Overview KPIs, data quality, quick actions |
| **Data** | Upload CSV, configure columns, preprocess |
| **EDA** | Trend, seasonality, distribution, outliers, correlation |
| **Train Models** | Configure & train LSTM + Hybrid ARIMA+XGBoost |
| **Model Comparison** | MAE/RMAPE/MAPE metrics, actual vs predicted charts |
| **Forecast** | Generate future demand forecasts (ensemble) |
| **Inventory** | Safety stock, reorder point, EOQ, projection simulation |
| **Reports** | Download processed data, model results, forecasts |
| **Settings** | Advanced model parameters (epochs, ARIMA order, etc.) |

### Quick Start Workflow

1. **Data** → Upload `data/sample_data/sales_data.csv` (or your own)
2. **Data** → Click "Load & Preprocess"
3. **Train Models** → Click "Train Both Models"
4. **Model Comparison** → Review metrics, note best model
5. **Forecast** → Click "Generate Forecast"
6. **Inventory** → Set current stock, review recommendations
7. **Reports** → Download CSV results

## 🔄 Methodology

### Data Flow

```
1. Data Collection (CSV Upload)
         ↓
2. Data Preprocessing (Cleaning, Feature Engineering)
         ↓
3. Exploratory Data Analysis (Understanding Patterns)
         ↓
4. Train-Test Split (80% Train, 20% Test) — CHRONOLOGICAL
         ↓
5. Model Training
     ├─ LSTM Training (scaler fit on TRAIN only — no leakage)
     └─ Hybrid ARIMA+XGBoost Training (on train split)
         ↓
6. Performance Comparison (RMSE, MAE, MAPE on aligned test window)
         ↓
7. Future Forecasting (models refit on FULL data for forecasting)
         ↓
8. Inventory Optimization (Safety Stock, Reorder Point, EOQ)
         ↓
9. Dashboard Visualization & Recommendations
```

### LSTM Model Architecture

```
Input Sequence (30 timesteps, 1 feature)
    ↓
LSTM(64 units) + Dropout(0.2)
    ↓
LSTM(32 units) + Dropout(0.2)
    ↓
Dense(16 units, ReLU)
    ↓
Output (1 prediction)
```

**Key**: Scaler fit ONLY on training data → prevents data leakage.

### Hybrid ARIMA + XGBoost

```
Historical Data
    ├─ ARIMA Models Linear Trends & Seasonality
    │   └─ Generates Base Forecast
    │
    └─ Calculate ARIMA Residuals
        └─ XGBoost Models Residual Patterns (lag features)
            └─ Generates Residual Corrections
                ↓
            Combines Forecast + Correction = Final Prediction
```

**Forecasting**: Model refit on FULL dataset before generating future predictions.

## 📈 Performance Metrics

- **RMSE**: Lower is better, penalizes large errors
- **MAE**: Average absolute difference in original units
- **MAPE**: Percentage error, easier to interpret

## 🛒 Inventory Optimization Calculations

### Safety Stock Formula
```
SS = Z-score × σ_demand × √(Lead Time)
```
Where:
- Z-score = Statistical value for desired service level (e.g., 1.645 for 95%)
- σ_demand = Standard deviation of demand (forecast variability preferred)
- Lead Time = Procurement lead time in days

### Reorder Point Formula
```
ROP = (Average Demand × Lead Time) + Safety Stock
```

### Economic Order Quantity (EOQ)
```
EOQ = √(2 × D × S / H)
```
Where:
- D = Annual demand
- S = Ordering cost per order
- H = Holding cost per unit per year

## 📦 Sample Data

A sample dataset (`data/sample_data/sales_data.csv`) is included with:
- **Date**: Transaction date (2022-01-01 to 2023-12-31, 731 days)
- **Sales**: Units sold (demand)
- **Price**: Product price
- **Promotion**: Whether promotion was active (0/1)

This data shows realistic demand patterns:
- Upward trend over time
- Seasonal variations
- Random fluctuations
- Promotional effects

## 🎓 Key Advantages

✅ **Multiple Models**: Combines LSTM and Hybrid approach for better accuracy  
✅ **No Data Leakage**: Strict chronological split, scaler fit on train only  
✅ **Aligned Evaluation**: LSTM and Hybrid test windows match exactly  
✅ **Business Focused**: Converts predictions into actionable inventory decisions  
✅ **User-Friendly**: Interactive dashboard requires no coding  
✅ **Real-Time**: Live visualization and instant recommendations  
✅ **Flexible**: Works with various CSV formats and datasets  
✅ **Cloud Ready**: Deploys to Streamlit Community Cloud with zero config  

## 🔬 Technical Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow/Keras 2.16 |
| **Machine Learning** | XGBoost 2.1, scikit-learn 1.5 |
| **Statistical** | statsmodels 0.14 |
| **Data Processing** | pandas 2.2, numpy 1.26 |
| **Visualization** | Plotly 5.24, Matplotlib 3.9, Seaborn 0.13 |
| **Web Interface** | Streamlit 1.40 |
| **Optimization** | scipy 1.13 |

## 📋 Requirements

See `requirements.txt` for complete pinned list:
- pandas==2.2.3
- numpy==1.26.4
- scikit-learn==1.5.2
- tensorflow==2.16.2
- xgboost==2.1.1
- statsmodels==0.14.4
- streamlit==1.40.1
- plotly==5.24.1
- matplotlib==3.9.2
- seaborn==0.13.2
- python-dateutil==2.9.0.post0
- pytz==2024.2
- openpyxl==3.1.5
- scipy==1.13.1
- pyarrow==17.0.0
- duckdb==1.1.3

## 🚀 Getting Started

### Quick Start Example (Python API)

```python
from src.preprocessing import DataPreprocessor
from src.models.lstm_model import LSTMForecaster
from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.model_comparison import ModelComparison
from src.inventory.optimization import InventoryOptimization

# Load and prepare data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/sample_data/sales_data.csv')
df = preprocessor.clean_data(df, 'Date', 'Sales')

# Prepare LSTM data (leakage-free)
X_tr, X_te, y_tr, y_te, scaler = preprocessor.prepare_lstm_data(df, 'Sales', seq_length=30)

# Train LSTM
lstm = LSTMForecaster(seq_length=30, epochs=50)
lstm.build_model((X_tr.shape[1], X_tr.shape[2]))
lstm.train(X_tr, y_tr, X_te, y_te, verbose=0)

# Prepare Hybrid data (aligned test window)
train_series, test_series, _ = preprocessor.prepare_hybrid_data(df, 'Sales')

# Train Hybrid
hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
hybrid.fit(train_series)
hybrid_preds = hybrid.evaluate_on_test(train_series, test_series)

# LSTM predictions
lstm_preds = scaler.inverse_transform(lstm.predict(X_te)).flatten()
y_test = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

# Compare
comparison = ModelComparison()
comp_df = comparison.compare_models(y_test, {'LSTM': lstm_preds, 'Hybrid': hybrid_preds})
print(comp_df)

# Forecast future (refit on full data)
hybrid_full = HybridArimaXGBoost(arima_order=(1, 1, 1))
hybrid_full.fit(df['Sales'].values)
future_fc = hybrid_full.forecast_future(steps=30)

# Inventory optimization
inv = InventoryOptimization(service_level=0.95)
recs = inv.generate_inventory_recommendations(df['Sales'].values, lead_time=7, forecast_data=future_fc)
print(inv.generate_optimization_report(recs))
```

## 📚 Module Documentation

### preprocessing.py
Data cleaning, feature engineering, train-test splitting (leakage-free)

### eda.py
Statistical analysis and visualization utilities

### lstm_model.py
LSTM neural network for time-series forecasting

### arima_xgboost.py
Hybrid model combining statistical and ML approaches

### model_comparison.py
Performance evaluation and model comparison (MAE, RMSE, MAPE)

### optimization.py
Inventory optimization calculations and recommendations

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: "Insufficient data for LSTM training"
**Solution**: Ensure dataset has at least 100+ records. Reduce Look-back window in sidebar if needed.

### Issue: "Streamlit app not opening"
**Solution**: Check if port 8501 is available. Use `streamlit run dashboard/app.py --server.port 8502`

### Issue: TensorFlow import error on Python 3.13
**Solution**: Use Python 3.10, 3.11, or 3.12. TensorFlow 2.16 does not support 3.13+ yet.

### Issue: ARIMA convergence warning
**Solution**: Normal - fallback orders are applied automatically. Check Model Comparison for actual order used.

## 📝 Configuration Parameters

### Sidebar Settings (Runtime)
- **Forecast Period**: 1-90 days ahead
- **Lead Time**: 1-30 days for inventory calculations
- **Service Level**: 80-99% for safety stock calculation
- **Look-back (seq_length)**: 10-60 days for LSTM sequence length

### Settings Page (Advanced)
- **LSTM Epochs**: Training iterations (10-300)
- **LSTM Batch Size**: Batch size during training (8-128)
- **ARIMA Order**: (p,d,q) parameters as comma-separated values
- **DuckDB**: Enable for large files >80 MB

## 🎯 Future Enhancements

- [ ] Multi-step ahead forecasting with confidence intervals
- [ ] Prophet model integration
- [ ] Real-time data streaming
- [ ] Database integration (PostgreSQL, SQLite)
- [ ] API endpoints for predictions (FastAPI)
- [ ] Model persistence and versioning (MLflow)
- [ ] Demand clustering by product/location
- [ ] Automated hyperparameter tuning (Optuna)

## 📄 License

This project is part of an academic research initiative on AI-based inventory optimization.

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on the repository
- Include sample data and error messages
- Describe expected vs actual behavior

## 📚 References

- **LSTM**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **ARIMA**: Box & Jenkins (1970) - "Time Series Analysis: Forecasting and Control"
- **XGBoost**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
- **Inventory Theory**: Harris (1913) - "Economic Order Quantity Model"

---

**Version**: 1.0  
**Last Updated**: 2026  
**Status**: Production Ready  
**Python**: 3.10 - 3.12  
**Deploy Target**: Streamlit Community Cloud  

Developed as part of AI-based demand forecasting and inventory optimization research.