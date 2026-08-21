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
- 80-20 train-test split

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
- CSV dataset upload
- Data preprocessing and validation
- Model training interface
- Real-time visualization
- Forecast comparison charts
- Inventory optimization recommendations
- Results download capability

## 📋 Project Structure

```
demand-forecasting/
├── data/
│   ├── sample_data/
│   │   └── sales_data.csv          # Sample dataset for testing
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Data pipeline
│   ├── eda.py                       # Exploratory analysis
│   ├── utils.py                     # Helper functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # LSTM implementation
│   │   ├── arima_xgboost.py        # Hybrid model
│   │   └── model_comparison.py     # Evaluation metrics
│   └── inventory/
│       ├── __init__.py
│       └── optimization.py          # Inventory calculations
├── dashboard/
│   └── app.py                       # Streamlit web interface
├── notebooks/                       # (Empty - reserved for future development)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd demand-forecasting
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Running the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Dashboard Tabs

#### 1. **Data Upload** 📤
- Upload CSV file with sales data
- Configure date and sales columns
- Automatic data cleaning and preprocessing
- Create time-based and lag features

#### 2. **EDA Analysis** 📊
- View summary statistics
- Visualize sales trends over time
- Analyze seasonal patterns
- Detect outliers
- Display correlation heatmaps

#### 3. **Model Training** 🤖
- Configure LSTM epochs and batch size
- Set ARIMA order (p,d,q)
- Train both LSTM and Hybrid models
- View model comparison metrics (RMSE, MAE, MAPE)

#### 4. **Forecasts** 📈
- Compare actual vs predicted values on test set
- Generate future demand predictions
- Visualize forecast for next N days
- Compare LSTM and Hybrid model forecasts

#### 5. **Inventory Optimization** 📦
- Calculate Safety Stock
- Determine Reorder Point
- View Economic Order Quantity (EOQ)
- Project inventory levels
- Generate optimization report

## 🔄 Methodology

### Data Flow

```
1. Data Collection (CSV Upload)
         ↓
2. Data Preprocessing (Cleaning, Feature Engineering)
         ↓
3. Exploratory Data Analysis (Understanding Patterns)
         ↓
4. Train-Test Split (80% Train, 20% Test)
         ↓
5. Model Training
    ├─ LSTM Training
    └─ Hybrid ARIMA+XGBoost Training
         ↓
6. Performance Comparison (RMSE, MAE, MAPE)
         ↓
7. Future Forecasting (Next N Days)
         ↓
8. Inventory Optimization (Safety Stock, Reorder Point)
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

### Hybrid ARIMA + XGBoost

```
Historical Data
    ├─ ARIMA Models Linear Trends & Seasonality
    │   └─ Generates Base Forecast
    │
    └─ Calculate ARIMA Residuals
        └─ XGBoost Models Residual Patterns
            └─ Generates Residual Corrections
                ↓
            Combines Forecast + Correction = Final Prediction
```

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
- σ_demand = Standard deviation of demand
- Lead Time = Procurement lead time

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

A sample dataset (`data/sample_data/sales_data.csv`) is included with the project containing:
- **Date**: Transaction date
- **Sales**: Units sold (demand)
- **Price**: Product price
- **Promotion**: Whether promotion was active (0/1)

This data shows realistic demand patterns with:
- Upward trend over time
- Seasonal variations
- Random fluctuations
- Promotional effects

## 🎓 Key Advantages

✅ **Multiple Models**: Combines LSTM and Hybrid approach for better accuracy
✅ **Handles Complexity**: Captures both linear and nonlinear demand patterns
✅ **Business Focused**: Converts predictions into actionable inventory decisions
✅ **User-Friendly**: Interactive dashboard requires no coding
✅ **Real-Time**: Live visualization and instant recommendations
✅ **Flexible**: Works with various CSV formats and datasets

## 🔬 Technical Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow/Keras |
| **Machine Learning** | XGBoost, scikit-learn |
| **Statistical** | statsmodels |
| **Data Processing** | pandas, numpy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Interface** | Streamlit |
| **Optimization** | scipy.stats |

## 📋 Requirements

See `requirements.txt` for complete list:
- pandas
- numpy
- scikit-learn
- tensorflow
- xgboost
- statsmodels
- streamlit
- plotly
- matplotlib
- seaborn

## 🚀 Getting Started

### Quick Start Example

```python
from src.preprocessing import DataPreprocessor
from src.models.lstm_model import LSTMForecaster
from src.models.model_comparison import ModelComparison

# Load and prepare data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/sample_data/sales_data.csv')
df = preprocessor.clean_data(df)
df = preprocessor.create_time_features(df)

# Train LSTM
X_lstm, y_lstm = preprocessor.prepare_lstm_data(df)
lstm = LSTMForecaster(epochs=50)
lstm.train(X_train, y_train)

# Make predictions
predictions = lstm.predict(X_test)

# Evaluate
comparison = ModelComparison()
metrics = comparison.evaluate_model(y_test, predictions, "LSTM")
print(metrics)
```

## 📚 Module Documentation

### preprocessing.py
Data cleaning, feature engineering, and train-test splitting

### eda.py
Statistical analysis and visualization utilities

### lstm_model.py
LSTM neural network for time-series forecasting

### arima_xgboost.py
Hybrid model combining statistical and ML approaches

### model_comparison.py
Performance evaluation and model comparison

### optimization.py
Inventory optimization calculations and recommendations

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: "Insufficient data for LSTM training"
**Solution**: Ensure dataset has at least 100+ records for reliable training

### Issue: "Streamlit app not opening"
**Solution**: Check if port 8501 is available. Use `streamlit run dashboard/app.py --server.port 8502`

## 📝 Configuration Parameters

### Sidebar Settings
- **Forecast Period**: 1-90 days ahead
- **Lead Time**: 1-30 days for inventory calculations
- **Service Level**: 80-99% for safety stock calculation
- **LSTM Epochs**: Training iterations (10-100)
- **LSTM Batch Size**: Batch size during training (8-64)
- **ARIMA Order**: (p,d,q) parameters as comma-separated values

## 🎯 Future Enhancements

- [ ] Multi-step ahead forecasting
- [ ] Prophet model integration
- [ ] Real-time data streaming
- [ ] Database integration
- [ ] API endpoints for predictions
- [ ] Model persistence and versioning
- [ ] Demand clustering by product/location
- [ ] Confidence intervals for forecasts

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
**Last Updated**: 2024
**Status**: Production Ready

Developed as part of AI-based demand forecasting and inventory optimization research.