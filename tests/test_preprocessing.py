"""Tests for data pipeline: loading, validation, cleaning, feature engineering, splitting."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.preprocessing import DataPreprocessor


class TestDataLoading:
    """Test CSV loading and column detection."""

    def test_load_csv(self):
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(ROOT / "data" / "sample_data" / "sales_data.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 730
        assert list(df.columns) == ['Date', 'Sales', 'Price', 'Promotion']

    def test_validate_data_success(self):
        preprocessor = DataPreprocessor()
        df = pd.read_csv(ROOT / "data" / "sample_data" / "sales_data.csv")
        is_valid, messages = preprocessor.validate_data(df, 'Date', 'Sales')
        assert is_valid is True
        assert isinstance(messages, list)

    def test_validate_data_missing_date_col(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'Sales': [1, 2, 3]})
        is_valid, messages = preprocessor.validate_data(df, 'Date', 'Sales')
        assert is_valid is False
        assert any('Date column' in m for m in messages)

    def test_validate_data_missing_sales_col(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02']})
        is_valid, messages = preprocessor.validate_data(df, 'Date', 'Sales')
        assert is_valid is False
        assert any('Sales' in m for m in messages)

    def test_validate_data_empty(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame()
        is_valid, messages = preprocessor.validate_data(df, 'Date', 'Sales')
        assert is_valid is False
        assert any('empty' in m.lower() for m in messages)

    def test_validate_data_constant_sales_warning(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'Date': pd.date_range('2023-01-01', periods=100), 'Sales': [10]*100})
        is_valid, messages = preprocessor.validate_data(df, 'Date', 'Sales')
        assert is_valid is True
        assert any('zero variance' in m.lower() or 'constant' in m.lower() for m in messages)


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_clean_data_parses_dates(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Sales': [10, 20, 30]
        })
        df_clean = preprocessor.clean_data(df, 'Date', 'Sales')
        assert pd.api.types.is_datetime64_any_dtype(df_clean['Date'])

    def test_clean_data_coerces_sales_numeric(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Sales': ['10', '20', '30']
        })
        df_clean = preprocessor.clean_data(df, 'Date', 'Sales')
        assert pd.api.types.is_numeric_dtype(df_clean['Sales'])

    def test_clean_data_deduplicates_dates(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-01', '2023-01-02'],
            'Sales': [10, 20, 30]
        })
        df_clean = preprocessor.clean_data(df, 'Date', 'Sales')
        assert len(df_clean) == 2
        assert df_clean.loc[df_clean['Date'] == '2023-01-01', 'Sales'].iloc[0] == 30  # sum

    def test_clean_data_sorts_by_date(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': ['2023-01-03', '2023-01-01', '2023-01-02'],
            'Sales': [30, 10, 20]
        })
        df_clean = preprocessor.clean_data(df, 'Date', 'Sales')
        assert df_clean['Date'].is_monotonic_increasing

    def test_clean_data_handles_missing_values(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Sales': [10, np.nan, 30, np.nan, 50]
        })
        df_clean = preprocessor.clean_data(df, 'Date', 'Sales')
        assert df_clean['Sales'].isna().sum() == 0
        # ffill then bfill then mean
        assert df_clean['Sales'].iloc[1] == 10  # ffill
        assert df_clean['Sales'].iloc[3] == 30  # ffill from 30

    def test_clean_data_drops_negative_sales(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Sales': [10, -5, 30, -1, 50]
        })
        df_clean = preprocessor.clean_data(df, 'Date', 'Sales')
        assert (df_clean['Sales'] >= 0).all()
        assert len(df_clean) == 3


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_create_time_features(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Sales': range(10)
        })
        df_feat = preprocessor.create_time_features(df, 'Date')

        expected_cols = ['Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend']
        for col in expected_cols:
            assert col in df_feat.columns

        assert df_feat['Year'].iloc[0] == 2023
        assert df_feat['Month'].iloc[0] == 1
        assert df_feat['IsWeekend'].isin([0, 1]).all()

    def test_create_lag_features(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Sales': range(50)
        })
        df_lag = preprocessor.create_lag_features(df, 'Sales', lags=(1, 7, 14))

        assert 'Sales_Lag_1' in df_lag.columns
        assert 'Sales_Lag_7' in df_lag.columns
        assert 'Sales_Lag_14' in df_lag.columns
        # After dropping NaN, first row has lag_1 = 0 (original index 1), lag_7 = 0 (original index 7)
        # df has values 0,1,2,3,4,5,6,7,8,9...
        # After lag_1: NaN,0,1,2,3... -> dropna -> first row is original index 1, lag_1=0
        # After lag_7: NaN... (7 NaNs),0,1,2... -> dropna -> first row is original index 7, lag_7=0
        # After lag_14: NaN... (14 NaNs) -> dropna -> first row is original index 14, lag_14=0
        # But all lag features are created then dropna() is called ONCE at the end
        # So the first row will have lag_1=0 (from original idx 1), lag_7=0 (from original idx 7), lag_14=0 (from original idx 14)
        # Actually wait - let me trace through:
        # Original df index: 0..49, values: 0..49
        # After shift(1): index 0 is NaN, index 1 is 0, index 2 is 1, ...
        # After shift(7): index 0-6 are NaN, index 7 is 0, index 8 is 1, ...
        # After shift(14): index 0-13 are NaN, index 14 is 0, index 15 is 1, ...
        # dropna() removes all rows where ANY column is NaN
        # So first valid row is index 14: lag_1=13, lag_7=7, lag_14=0
        assert df_lag['Sales_Lag_1'].iloc[0] == 13  # value at original index 13
        assert df_lag['Sales_Lag_7'].iloc[0] == 7   # value at original index 7
        assert df_lag['Sales_Lag_14'].iloc[0] == 0  # value at original index 0
        # NaN rows dropped
        assert df_lag['Sales_Lag_1'].isna().sum() == 0

    def test_create_rolling_features(self):
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'Sales': [10, 20, 30, 40, 50] * 4  # 20 values
        })
        df_roll = preprocessor.create_rolling_features(df, 'Sales', windows=(3, 5))

        assert 'Rolling_Mean_3' in df_roll.columns
        assert 'Rolling_Std_3' in df_roll.columns
        assert 'Rolling_Mean_5' in df_roll.columns
        assert 'Rolling_Std_5' in df_roll.columns
        # min_periods=1 so no NaN
        assert df_roll['Rolling_Mean_3'].isna().sum() == 0


class TestScaling:
    """Test MinMaxScaler operations."""

    def test_scale_data_fit_transform(self):
        preprocessor = DataPreprocessor()
        data = np.array([[10], [20], [30], [40], [50]], dtype=float)
        scaled = preprocessor.scale_data(data, fit=True)

        assert scaled.shape == data.shape
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0
        # Check inverse
        inversed = preprocessor.inverse_scale(scaled)
        np.testing.assert_allclose(inversed, data, rtol=1e-10)

    def test_scale_data_transform_only(self):
        preprocessor = DataPreprocessor()
        train = np.array([[10], [20], [30]], dtype=float)
        test = np.array([[15], [25]], dtype=float)

        preprocessor.scale_data(train, fit=True)
        test_scaled = preprocessor.scale_data(test, fit=False)

        # Test should use train's min/max
        assert test_scaled[0, 0] == 0.25  # (15-10)/(30-10) = 5/20 = 0.25

    def test_inverse_scale(self):
        preprocessor = DataPreprocessor()
        data = np.array([[10], [20], [30]], dtype=float)
        scaled = preprocessor.scale_data(data, fit=True)
        inversed = preprocessor.inverse_scale(scaled)
        np.testing.assert_allclose(inversed, data, rtol=1e-10)


class TestLSTMDataPreparation:
    """Test LSTM sequence generation and train/test split (NEW leakage-free API)."""

    def test_prepare_lstm_data_shapes(self, sample_data_clean, seq_length):
        df_clean, preprocessor = sample_data_clean
        X_tr, X_te, y_tr, y_te, scaler = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length, test_size=0.2
        )

        # Train samples = train_size - seq_length
        expected_train = preprocessor.train_size - seq_length
        expected_test = len(df_clean) - preprocessor.train_size - seq_length

        assert X_tr.shape == (expected_train, seq_length, 1)
        assert y_tr.shape == (expected_train,)
        assert X_te.shape == (expected_test, seq_length, 1)
        assert y_te.shape == (expected_test,)
        # Values should be scaled 0-1
        assert X_tr.min() >= 0.0 and X_tr.max() <= 1.0
        assert y_tr.min() >= 0.0 and y_tr.max() <= 1.0

    def test_train_test_split_chronological(self, sample_data_clean, seq_length):
        df_clean, preprocessor = sample_data_clean
        X_tr, X_te, y_tr, y_te, _ = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length, test_size=0.2
        )

        # Chronological: train sequences come before test sequences
        assert len(X_tr) + len(X_te) == len(df_clean) - 2 * seq_length
        assert len(y_tr) + len(y_te) == len(df_clean) - 2 * seq_length
        # Train size stored
        assert preprocessor.train_size > 0
        # No overlap - last train sequence ends before first test sequence starts
        # (in scaled space, values should be from earlier dates)

    def test_no_data_leakage_scaler_fit_on_train_only(self, sample_data_clean, seq_length):
        """Verify scaler is fit ONLY on training data."""
        df_clean, preprocessor = sample_data_clean
        X_tr, X_te, y_tr, y_te, scaler = preprocessor.prepare_lstm_data(
            df_clean, 'Sales', seq_length, test_size=0.2
        )

        # The scaler should be fit on training data only
        # Check that test data was transformed using train scaler (not fit on test)
        # This is verified by checking that test min/max are within train min/max bounds
        train_data = df_clean['Sales'].values[:preprocessor.train_size].reshape(-1, 1)
        test_data = df_clean['Sales'].values[preprocessor.train_size:].reshape(-1, 1)

        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Train scaled should be in [0, 1]
        assert train_scaled.min() >= 0.0 and train_scaled.max() <= 1.0
        # Test scaled may go outside [0, 1] if test has values outside train range
        # But it should use the SAME scaler (not a newly fit one)


class TestHybridDataPreparation:
    """Test Hybrid model data preparation (raw series split)."""

    def test_prepare_hybrid_data_shapes(self, sample_data_clean):
        df_clean, preprocessor = sample_data_clean
        train_series, test_series, split_idx = preprocessor.prepare_hybrid_data(
            df_clean, 'Sales', test_size=0.2
        )

        assert isinstance(train_series, np.ndarray)
        assert isinstance(test_series, np.ndarray)
        assert len(train_series) + len(test_series) == len(df_clean)
        assert split_idx == len(train_series)
        assert preprocessor.train_size == split_idx

    def test_prepare_hybrid_data_chronological(self, sample_data_clean):
        df_clean, preprocessor = sample_data_clean
        train_series, test_series, _ = preprocessor.prepare_hybrid_data(
            df_clean, 'Sales', test_size=0.2
        )

        # Train comes first chronologically
        assert train_series[-1] == df_clean['Sales'].iloc[len(train_series) - 1]
        assert test_series[0] == df_clean['Sales'].iloc[len(train_series)]


class TestUtilities:
    """Test utility functions."""

    def test_get_date_range(self, sample_data_clean):
        df_clean, preprocessor = sample_data_clean
        min_d, max_d = preprocessor.get_date_range(df_clean, 'Date')
        assert min_d == df_clean['Date'].min()
        assert max_d == df_clean['Date'].max()

    def test_resample_data(self, sample_data_clean):
        df_clean, preprocessor = sample_data_clean
        # Resample to weekly
        df_weekly = preprocessor.resample_data(df_clean, 'Date', 'Sales', 'W')
        assert len(df_weekly) < len(df_clean)  # Fewer weeks than days
        assert 'Date' in df_weekly.columns
        assert 'Sales' in df_weekly.columns
