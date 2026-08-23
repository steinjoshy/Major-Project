"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.arima_xgboost import HybridArimaXGBoost
from src.models.lstm_model import LSTMForecaster
from src.preprocessing import DataPreprocessor


@pytest.fixture(scope="session")
def sample_data():
    """Load the sample dataset."""
    df = pd.read_csv(ROOT / "data" / "sample_data" / "sales_data.csv")
    return df

@pytest.fixture(scope="session")
def sample_data_clean():
    """Cleaned sample dataset using DataPreprocessor."""
    df = pd.read_csv(ROOT / "data" / "sample_data" / "sales_data.csv")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df, date_col='Date', sales_col='Sales')
    return df_clean, preprocessor

@pytest.fixture(scope="session")
def train_test_split(sample_data_clean):
    """Standard 80/20 chronological split for testing."""
    df_clean, preprocessor = sample_data_clean
    test_size = 0.2
    split = int(len(df_clean) * (1 - test_size))
    train_df = df_clean.iloc[:split].copy()
    test_df = df_clean.iloc[split:].copy()
    return train_df, test_df, split

@pytest.fixture(scope="session")
def seq_length():
    return 30

@pytest.fixture(scope="session")
def forecast_steps():
    return 30

@pytest.fixture(scope="session")
def train_data(sample_data_clean):
    """Training data for Hybrid model."""
    df_clean, _ = sample_data_clean
    return df_clean['Sales'].values.astype(float)

@pytest.fixture(scope="session")
def fitted_hybrid(train_data):
    """Pre-fitted Hybrid model for testing."""
    hybrid = HybridArimaXGBoost(arima_order=(1, 1, 1))
    hybrid.fit(train_data[:200])  # Use subset for speed
    return hybrid

@pytest.fixture(scope="session")
def lstm_data(sample_data_clean, seq_length):
    """LSTM training data fixture - uses NEW leakage-free method."""
    df_clean, preprocessor = sample_data_clean
    X_tr, X_te, y_tr, y_te, _ = preprocessor.prepare_lstm_data(
        df_clean, 'Sales', seq_length, test_size=0.2
    )
    return X_tr, X_te, y_tr, y_te, preprocessor

@pytest.fixture(scope="session")
def trained_lstm(lstm_data):
    """Pre-trained LSTM model for testing."""
    X_tr, X_te, y_tr, y_te, preprocessor = lstm_data
    forecaster = LSTMForecaster(seq_length=30, epochs=2, batch_size=16)
    forecaster.train(X_tr, y_tr, X_te, y_te, verbose=0)
    return forecaster, X_te, y_te, preprocessor

@pytest.fixture(scope="session")
def trained_lstm_for_forecast(sample_data_clean, seq_length):
    """Pre-trained LSTM model on full data for forecasting tests."""
    df_clean, preprocessor = sample_data_clean
    # Use legacy method for full-data training (for forecasting tests)
    X, y = preprocessor.prepare_lstm_data_legacy(df_clean, 'Sales', seq_length)
    forecaster = LSTMForecaster(seq_length=seq_length, epochs=2, batch_size=16)
    forecaster.train(X, y, verbose=0)
    return forecaster, preprocessor, df_clean

@pytest.fixture
def pipeline_data():
    """Full pipeline data fixture for integration tests (function-scoped for isolation)."""
    preprocessor = DataPreprocessor()
    df = pd.read_csv(ROOT / "data" / "sample_data" / "sales_data.csv")
    df_clean = preprocessor.clean_data(df, 'Date', 'Sales')

    seq_length = 30
    test_size = 0.2

    # Use NEW leakage-free method
    X_tr, X_te, y_tr, y_te, _ = preprocessor.prepare_lstm_data(
        df_clean, 'Sales', seq_length, test_size
    )

    train_series, test_series, _ = preprocessor.prepare_hybrid_data(
        df_clean, 'Sales', test_size
    )

    return {
        'df_clean': df_clean,
        'preprocessor': preprocessor,
        'seq_length': seq_length,
        'test_size': test_size,
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'train_series': train_series, 'test_series': test_series,
    }

# Suppress specific known warnings for cleaner test output
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

@pytest.fixture(scope="session")
def service():
    """Create IngestionService instance for testing."""
    from src.services.ingestion_service import IngestionService
    return IngestionService(use_duckdb=False, min_rows_lstm=50)
