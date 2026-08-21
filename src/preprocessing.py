import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Minimum rows required for LSTM with default seq_length=30 and 80/20 split
_MIN_ROWS_LSTM = 100


class DataPreprocessor:
    """Handles data preprocessing and feature engineering for time-series forecasting."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.date_column = None
        self.sales_column = None
        self.train_size = None  # set after train/test split

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_data(self, filepath):
        """Load CSV data from a file path."""
        return pd.read_csv(filepath)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_data(self, df, date_col, sales_col):
        """
        Validate the dataset before processing.

        Returns:
            (is_valid: bool, messages: list[str])
        """
        errors = []
        warnings = []

        if df is None or len(df) == 0:
            errors.append("Dataset is empty.")
            return False, errors + warnings

        if date_col not in df.columns:
            errors.append(f"Date column '{date_col}' not found in dataset.")

        if sales_col not in df.columns:
            errors.append(f"Sales/demand column '{sales_col}' not found in dataset.")

        if errors:
            return False, errors

        # Attempt date parse
        try:
            test_dates = pd.to_datetime(df[date_col].dropna().head(5), errors='raise')
        except Exception:
            errors.append(f"Column '{date_col}' does not appear to contain valid dates.")

        # Check numeric sales
        non_numeric = pd.to_numeric(df[sales_col], errors='coerce').isna().sum()
        if non_numeric == len(df):
            errors.append(f"Column '{sales_col}' contains no numeric values.")
        elif non_numeric > 0:
            warnings.append(f"{non_numeric} non-numeric values found in '{sales_col}' — they will be dropped.")

        if errors:
            return False, errors + warnings

        # Row count
        if len(df) < _MIN_ROWS_LSTM:
            warnings.append(
                f"Only {len(df)} rows found. LSTM requires at least {_MIN_ROWS_LSTM} records "
                f"for reliable training. Results may be unreliable."
            )

        # Constant demand
        numeric_vals = pd.to_numeric(df[sales_col], errors='coerce').dropna()
        if numeric_vals.std() == 0:
            warnings.append("Demand column has zero variance (constant value). Forecasting will not be meaningful.")

        # High missing rate
        missing_rate = df[sales_col].isna().mean()
        if missing_rate > 0.3:
            warnings.append(f"{missing_rate:.0%} of sales values are missing. Results may be unreliable.")

        return True, warnings

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_data(self, df, date_col='Date', sales_col='Sales'):
        """
        Clean dataset: handle missing values, duplicates, format dates.

        Args:
            df: Input dataframe (should already be aggregated by date)
            date_col: Name of date column
            sales_col: Name of sales/quantity column

        Returns:
            Cleaned dataframe sorted by date
        """
        self.date_column = date_col
        self.sales_column = sales_col

        df = df.copy()

        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Drop rows where date could not be parsed
        df = df.dropna(subset=[date_col])

        # Convert sales to numeric
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

        # Remove duplicate dates (keep sum if aggregated, otherwise keep first)
        if df.duplicated(subset=[date_col]).any():
            df = df.groupby(date_col, as_index=False)[sales_col].sum()

        # Sort by date
        df = df.sort_values(by=date_col).reset_index(drop=True)

        # Handle missing values in sales column — forward fill then backfill then mean
        if df[sales_col].isnull().sum() > 0:
            df[sales_col] = df[sales_col].fillna(method='ffill').fillna(method='bfill')
            if df[sales_col].isnull().sum() > 0:
                df[sales_col] = df[sales_col].fillna(df[sales_col].mean())

        # Remove negative sales values
        df = df[df[sales_col] >= 0].copy()

        return df

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def create_time_features(self, df, date_col='Date'):
        """Create time-based features: day, week, month, quarter, year."""
        df = df.copy()
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Week'] = df[date_col].dt.isocalendar().week.astype(int)
        df['Day'] = df[date_col].dt.day
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['Quarter'] = df[date_col].dt.quarter
        df['IsWeekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        return df

    def create_lag_features(self, df, sales_col='Sales', lags=(1, 7, 14, 30)):
        """Create lag features from previous sales values."""
        df = df.copy()
        for lag in lags:
            df[f'Sales_Lag_{lag}'] = df[sales_col].shift(lag)
        # Drop NaN rows created by lag features
        df = df.dropna().reset_index(drop=True)
        return df

    def create_rolling_features(self, df, sales_col='Sales', windows=(7, 14, 30)):
        """Create rolling mean and std features."""
        df = df.copy()
        for w in windows:
            df[f'Rolling_Mean_{w}'] = df[sales_col].rolling(window=w, min_periods=1).mean()
            df[f'Rolling_Std_{w}'] = df[sales_col].rolling(window=w, min_periods=1).std().fillna(0)
        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_data(self, X, fit=True):
        """Scale numerical data using MinMaxScaler."""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)

    def inverse_scale(self, X_scaled):
        """Inverse transform scaled data back to original scale."""
        return self.scaler.inverse_transform(X_scaled)

    # ------------------------------------------------------------------
    # LSTM Data Preparation
    # ------------------------------------------------------------------

    def prepare_lstm_data(self, df, sales_col='Sales', seq_length=30):
        """
        Prepare data for LSTM: scale demand, then create sliding-window sequences.

        The scaler is fit on the ENTIRE series here (before split). The caller
        must use inverse_scale() to recover original-scale predictions.

        Returns:
            X (n_samples, seq_length, 1), y (n_samples,)
        """
        data = df[sales_col].values.reshape(-1, 1)
        data_scaled = self.scale_data(data, fit=True)

        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i: i + seq_length])
            y.append(data_scaled[i + seq_length, 0])

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Hybrid Model Data Preparation (no leakage)
    # ------------------------------------------------------------------

    def prepare_hybrid_data(self, df, sales_col='Sales', test_size=0.2):
        """
        Split the raw demand series chronologically for Hybrid ARIMA+XGBoost evaluation.

        No shuffling. Returns raw (unscaled) arrays.

        Returns:
            train_series (np.array), test_series (np.array), split_idx (int)
        """
        values = df[sales_col].values.astype(float)
        split_idx = int(len(values) * (1 - test_size))
        self.train_size = split_idx
        train_series = values[:split_idx]
        test_series = values[split_idx:]
        return train_series, test_series, split_idx

    # ------------------------------------------------------------------
    # Train/Test Split (for LSTM sequences)
    # ------------------------------------------------------------------

    def train_test_split_data(self, X, y, test_size=0.2):
        """
        Chronological split of LSTM sequences — NO shuffling.
        Returns X_train, X_test, y_train, y_test.
        """
        split = int(len(X) * (1 - test_size))
        self.train_size = split
        return X[:split], X[split:], y[:split], y[split:]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_date_range(self, df, date_col='Date'):
        """Get (min_date, max_date) from dataframe."""
        return df[date_col].min(), df[date_col].max()

    def resample_data(self, df, date_col='Date', sales_col='Sales', freq='D'):
        """Resample time-series data to specified frequency (sum)."""
        df = df.copy().set_index(date_col)
        df_resampled = df[sales_col].resample(freq).sum()
        return df_resampled.reset_index()


if __name__ == "__main__":
    print("Data preprocessing module loaded successfully")
