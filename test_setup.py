#!/usr/bin/env python3
"""Test that all required modules are installed and working."""

try:
    import tensorflow
    print(f'[OK] TensorFlow {tensorflow.__version__}')
except ImportError as e:
    print(f'[ERROR] TensorFlow: {e}')

try:
    import pandas
    print(f'[OK] Pandas {pandas.__version__}')
except ImportError as e:
    print(f'[ERROR] Pandas: {e}')

try:
    import numpy
    print(f'[OK] NumPy {numpy.__version__}')
except ImportError as e:
    print(f'[ERROR] NumPy: {e}')

try:
    import xgboost
    print(f'[OK] XGBoost {xgboost.__version__}')
except ImportError as e:
    print(f'[ERROR] XGBoost: {e}')

try:
    import statsmodels
    print(f'[OK] Statsmodels installed')
except ImportError as e:
    print(f'[ERROR] Statsmodels: {e}')

try:
    import streamlit
    print(f'[OK] Streamlit {streamlit.__version__}')
except ImportError as e:
    print(f'[ERROR] Streamlit: {e}')

try:
    import plotly
    print(f'[OK] Plotly installed')
except ImportError as e:
    print(f'[ERROR] Plotly: {e}')

try:
    import matplotlib
    print(f'[OK] Matplotlib installed')
except ImportError as e:
    print(f'[ERROR] Matplotlib: {e}')

try:
    import seaborn
    print(f'[OK] Seaborn installed')
except ImportError as e:
    print(f'[ERROR] Seaborn: {e}')

try:
    import sklearn
    print(f'[OK] Scikit-learn installed')
except ImportError as e:
    print(f'[ERROR] Scikit-learn: {e}')

print('\n--- Testing project modules ---')
try:
    from src.preprocessing import DataProcessor
    print('[OK] preprocessing module')
except ImportError as e:
    print(f'[ERROR] preprocessing: {e}')

try:
    from src.eda import EDA
    print('[OK] eda module')
except ImportError as e:
    print(f'[ERROR] eda: {e}')

try:
    from src.models.lstm_model import LSTMForecaster
    print('[OK] lstm_model module')
except ImportError as e:
    print(f'[ERROR] lstm_model: {e}')

try:
    from src.models.arima_xgboost import HybridForecaster
    print('[OK] arima_xgboost module')
except ImportError as e:
    print(f'[ERROR] arima_xgboost: {e}')

try:
    from src.models.model_comparison import ModelComparison
    print('[OK] model_comparison module')
except ImportError as e:
    print(f'[ERROR] model_comparison: {e}')

try:
    from src.inventory.optimization import InventoryOptimizer
    print('[OK] inventory optimization module')
except ImportError as e:
    print(f'[ERROR] inventory optimization: {e}')

print('\n[SUCCESS] All dependencies and modules ready!')
