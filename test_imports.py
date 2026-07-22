#!/usr/bin/env python
"""Test script to verify all modules load correctly."""

print("Testing AI Demand Forecasting System\n")

try:
    print("Testing imports...")
    from src.preprocessing import DataPreprocessor
    print("[OK] Preprocessing module loaded")
    
    from src.eda import ExploratoryAnalysis
    print("[OK] EDA module loaded")
    
    from src.models.lstm_model import LSTMForecaster
    print("[OK] LSTM Forecaster loaded (using sklearn backend)")
    
    from src.models.arima_xgboost import HybridArimaXGBoost
    print("[OK] Hybrid ARIMA+XGBoost model loaded")
    
    from src.models.model_comparison import ModelComparison
    print("[OK] Model Comparison module loaded")
    
    from src.inventory.optimization import InventoryOptimization
    print("[OK] Inventory Optimization module loaded")
    
    print("\n[SUCCESS] ALL MODULES LOADED SUCCESSFULLY!\n")
    print("System is ready to run!\n")
    print("Next step: streamlit run dashboard/app.py")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
