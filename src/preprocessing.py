import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Minimum rows required for LSTM with default seq_length=30 and 80/20 split
_MIN_ROWS_LSTM = 100


class DataPreprocessor:
    """Handles data preprocessing and feature engineering for time-series forecasting.

    Supports both single-series (legacy) and panel data (store_id, sku_id combinations).
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.date_column = None
        self.sales_column = None
        self.train_size = None  # set after train/test split
        self.store_column = None
        self.sku_column = None
        self._panel_scalers = {}  # per-group scalers for panel data
        self._is_panel = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_data(self, filepath):
        """Load CSV data from a file path."""
        return pd.read_csv(filepath)

    def load_train_test_data(self, train_path, test_path, sample_submission_path=None):
        """Load train.csv, test.csv, and optional sample_submission.csv.

        Args:
            train_path: Path to train.csv (with units_sold target)
            test_path: Path to test.csv (no units_sold)
            sample_submission_path: Optional path to sample_submission.csv

        Returns:
            Tuple of (train_df, test_df, sample_submission_df or None)
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        sample_submission_df = None
        if sample_submission_path:
            sample_submission_df = pd.read_csv(sample_submission_path)
        return train_df, test_df, sample_submission_df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_data(self, df, date_col, sales_col, store_col=None, sku_col=None):
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

        if store_col and store_col not in df.columns:
            errors.append(f"Store column '{store_col}' not found in dataset.")

        if sku_col and sku_col not in df.columns:
            errors.append(f"SKU column '{sku_col}' not found in dataset.")

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

        # Panel data specific checks
        if store_col and sku_col:
            n_groups = df.groupby([store_col, sku_col]).ngroups
            min_group_size = df.groupby([store_col, sku_col]).size().min()
            if min_group_size < seq_length + 10:
                warnings.append(
                    f"Some (store_id, sku_id) groups have fewer than {seq_length + 10} rows. "
                    f"Minimum group size: {min_group_size}. LSTM may not work well for these groups."
                )
            warnings.append(f"Detected {n_groups} unique (store_id, sku_id) combinations.")

        return True, warnings

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_data(self, df, date_col='Date', sales_col='Sales', store_col=None, sku_col=None):
        """
        Clean dataset: handle missing values, duplicates, format dates.

        For panel data, sorts by store/sku then date.

        Args:
            df: Input dataframe
            date_col: Name of date column
            sales_col: Name of sales/quantity column
            store_col: Optional store identifier column
            sku_col: Optional SKU identifier column

        Returns:
            Cleaned dataframe sorted by (store, sku, date) or just date
        """
        self.date_column = date_col
        self.sales_column = sales_col
        self.store_column = store_col
        self.sku_column = sku_col
        self._is_panel = store_col is not None and sku_col is not None

        df = df.copy()

        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Drop rows where date could not be parsed
        df = df.dropna(subset=[date_col])

        # Convert sales to numeric
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')

        # For panel data: sort by store, sku, then date
        # For single series: sort by date
        if self._is_panel:
            sort_cols = [store_col, sku_col, date_col]
            df = df.sort_values(by=sort_cols).reset_index(drop=True)

            # Remove duplicate (store, sku, date) combinations - keep sum
            dup_mask = df.duplicated(subset=[store_col, sku_col, date_col], keep=False)
            if dup_mask.any():
                df = df.groupby([store_col, sku_col, date_col], as_index=False)[sales_col].sum()
                df = df.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            # Single series - remove duplicate dates (keep sum)
            if df.duplicated(subset=[date_col]).any():
                df = df.groupby(date_col, as_index=False)[sales_col].sum()
            df = df.sort_values(by=date_col).reset_index(drop=True)

        # Handle missing values in sales column — forward fill then backfill then mean
        # For panel data, do this per group
        if self._is_panel:
            df[sales_col] = df.groupby([store_col, sku_col])[sales_col].transform(
                lambda x: x.ffill().bfill()
            )
            if df[sales_col].isnull().sum() > 0:
                df[sales_col] = df[sales_col].fillna(df[sales_col].mean())
        else:
            if df[sales_col].isnull().sum() > 0:
                df[sales_col] = df[sales_col].ffill().bfill()
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

    def create_lag_features(self, df, sales_col='Sales', lags=(1, 7, 14, 30),
                            store_col=None, sku_col=None):
        """Create lag features from previous sales values.

        For panel data, creates lags within each (store, sku) group.
        """
        df = df.copy()
        is_panel = store_col is not None and sku_col is not None

        if is_panel:
            for lag in lags:
                df[f'Sales_Lag_{lag}'] = df.groupby([store_col, sku_col])[sales_col].shift(lag)
        else:
            for lag in lags:
                df[f'Sales_Lag_{lag}'] = df[sales_col].shift(lag)

        # Drop NaN rows created by lag features
        df = df.dropna().reset_index(drop=True)
        return df

    def create_rolling_features(self, df, sales_col='Sales', windows=(7, 14, 30),
                                 store_col=None, sku_col=None):
        """Create rolling mean and std features.

        For panel data, computes rolling stats within each (store, sku) group.
        """
        df = df.copy()
        is_panel = store_col is not None and sku_col is not None

        if is_panel:
            for w in windows:
                df[f'Rolling_Mean_{w}'] = df.groupby([store_col, sku_col])[sales_col].transform(
                    lambda x: x.rolling(window=w, min_periods=1).mean()
                )
                df[f'Rolling_Std_{w}'] = df.groupby([store_col, sku_col])[sales_col].transform(
                    lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
                )
        else:
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

    def _create_sequences(self, data_scaled, seq_length):
        """Create sliding-window sequences from scaled data."""
        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i: i + seq_length])
            y.append(data_scaled[i + seq_length, 0])
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Panel Data Utilities
    # ------------------------------------------------------------------

    def _get_panel_groups(self, df):
        """Get sorted list of (store_id, sku_id) groups."""
        if not self._is_panel:
            return [('default', 'default')]
        return sorted(df.groupby([self.store_column, self.sku_column]).groups.keys())

    def _split_panel_data_chronological(self, df, test_size=0.2):
        """Split panel data chronologically within each group.

        Returns:
            train_df, test_df, split_info_dict
        """
        if not self._is_panel:
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            return train_df, test_df, {'default': split_idx}

        train_parts = []
        test_parts = []
        split_info = {}

        for (store_id, sku_id), group in df.groupby([self.store_column, self.sku_column]):
            group = group.sort_values(self.date_column).reset_index(drop=True)
            split_idx = int(len(group) * (1 - test_size))
            # Ensure minimum training size for sequences
            min_train = max(split_idx, 50)
            split_idx = min(min_train, len(group) - 1)
            split_info[(store_id, sku_id)] = split_idx

            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        return train_df, test_df, split_info

    # ------------------------------------------------------------------
    # LSTM Data Preparation - Single Series (Legacy/Non-Panel)
    # ------------------------------------------------------------------

    def prepare_lstm_data(self, df, sales_col='Sales', seq_length=30, test_size=0.2):
        """
        Prepare data for LSTM: split chronologically, then scale, then create sequences.
        NO DATA LEAKAGE - scaler fit ONLY on training data.

        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        data = df[sales_col].values.reshape(-1, 1)

        # Chronological split FIRST (before scaling)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]

        # Fit scaler ONLY on training data
        scaler = MinMaxScaler()
        scaler.fit(train_data)

        # Scale the complete dataset using only the training data scaler
        data_scaled = scaler.transform(data)

        # Create sequences from each partition
        X, y = self._create_sequences(data_scaled, seq_length)
        # The first test prediction starts after training portion.
        # We keep the previous `seq_length` values as historical context.
        sequence_split = split_idx - seq_length

        X_train = X[:sequence_split]
        y_train = y[:sequence_split]

        X_test = X[sequence_split:]
        y_test = y[sequence_split:]

        # Store scaler for inverse transform
        self.scaler = scaler
        self.train_size = split_idx

        return X_train, X_test, y_train, y_test, scaler

    def prepare_lstm_data_legacy(self, df, sales_col='Sales', seq_length=30):
        """
        Legacy method: scale full series then create sequences (WITH LEAKAGE).
        Kept for backward compatibility. Use prepare_lstm_data() for new code.
        """
        data = df[sales_col].values.reshape(-1, 1)
        data_scaled = self.scale_data(data, fit=True)

        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i: i + seq_length])
            y.append(data_scaled[i + seq_length, 0])

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # LSTM Data Preparation - Panel Data (store_id, sku_id)
    # ------------------------------------------------------------------

    def prepare_lstm_data_panel(self, df, sales_col='Units_Sold', seq_length=30, test_size=0.2,
                                 store_col='store_id', sku_col='sku_id'):
        """
        Prepare LSTM data for panel data (multiple store/SKU series).

        For each (store_id, sku_id) group:
        1. Sort by date
        2. Chronological split (80/20 by default)
        3. Fit scaler on training portion ONLY
        4. Create sequences with historical context

        Returns:
            X_train, X_test, y_train, y_test, scalers_dict, group_info
        """
        self._is_panel = True
        self.store_column = store_col
        self.sku_column = sku_col
        self.sales_column = sales_col

        # Ensure sorted
        df = df.sort_values([store_col, sku_col, self.date_column]).reset_index(drop=True)

        all_X_train = []
        all_y_train = []
        all_X_test = []
        all_y_test = []
        group_info = {}
        self._panel_scalers = {}

        for (store_id, sku_id), group in df.groupby([store_col, sku_col]):
            group = group.sort_values(self.date_column).reset_index(drop=True)
            data = group[sales_col].values.reshape(-1, 1)

            if len(data) < seq_length + 10:
                # Skip groups with insufficient data
                group_info[(store_id, sku_id)] = {
                    'skipped': True,
                    'reason': f'Insufficient data: {len(data)} rows',
                    'n_rows': len(data)
                }
                continue

            # Chronological split
            split_idx = int(len(data) * (1 - test_size))
            # Ensure minimum for sequences
            split_idx = max(split_idx, seq_length + 5)
            split_idx = min(split_idx, len(data) - 1)

            train_data = data[:split_idx]
            test_data = data[split_idx:]

            # Fit scaler ONLY on training data for this group
            scaler = MinMaxScaler()
            scaler.fit(train_data)
            self._panel_scalers[(store_id, sku_id)] = scaler

            # Scale full group data using training scaler
            data_scaled = scaler.transform(data)

            # Create sequences
            X, y = self._create_sequences(data_scaled, seq_length)
            sequence_split = split_idx - seq_length

            X_train = X[:sequence_split]
            y_train = y[:sequence_split]
            X_test = X[sequence_split:]
            y_test = y[sequence_split:]

            if len(X_train) > 0:
                all_X_train.append(X_train)
                all_y_train.append(y_train)
            if len(X_test) > 0:
                all_X_test.append(X_test)
                all_y_test.append(y_test)

            group_info[(store_id, sku_id)] = {
                'skipped': False,
                'n_rows': len(data),
                'train_size': split_idx,
                'n_train_seq': len(X_train),
                'n_test_seq': len(X_test),
                'sequence_split': sequence_split
            }

        # Combine all groups
        if all_X_train:
            X_train = np.concatenate(all_X_train, axis=0)
            y_train = np.concatenate(all_y_train, axis=0)
        else:
            X_train = np.array([]).reshape(0, seq_length, 1)
            y_train = np.array([])

        if all_X_test:
            X_test = np.concatenate(all_X_test, axis=0)
            y_test = np.concatenate(all_y_test, axis=0)
        else:
            X_test = np.array([]).reshape(0, seq_length, 1)
            y_test = np.array([])

        # Store first group's scaler as default for backward compat
        if self._panel_scalers:
            self.scaler = list(self._panel_scalers.values())[0]

        return X_train, X_test, y_train, y_test, self._panel_scalers, group_info

    def prepare_lstm_test_sequences(self, train_df, test_df, sales_col='Units_Sold',
                                     seq_length=30, store_col='store_id', sku_col='sku_id'):
        """
        Prepare LSTM sequences for test.csv using historical context from train.csv.

        For each (store_id, sku_id) in test_df:
        1. Get last `seq_length` observations from train_df
        2. Combine with test_df observations (which have no target)
        3. Scale using the training-fitted scaler
        4. Create sequences for prediction

        Args:
            train_df: Training dataframe with target column
            test_df: Test dataframe (no target column)
            sales_col: Target column name (in train_df)
            seq_length: Sequence length
            store_col: Store identifier column
            sku_col: SKU identifier column

        Returns:
            X_test_sequences: Array of shape (n_test_samples, seq_length, 1)
            test_record_ids: Array of record_IDs corresponding to each prediction
            group_info: Dict with per-group metadata
        """
        # Ensure we have scalers from training
        if not self._panel_scalers:
            raise ValueError("No panel scalers found. Train on panel data first using prepare_lstm_data_panel().")

        train_df = train_df.sort_values([store_col, sku_col, self.date_column]).reset_index(drop=True)
        test_df = test_df.sort_values([store_col, sku_col, self.date_column]).reset_index(drop=True)

        all_X_test = []
        all_record_ids = []
        group_info = {}

        for (store_id, sku_id), test_group in test_df.groupby([store_col, sku_col]):
            test_group = test_group.sort_values(self.date_column).reset_index(drop=True)

            # Get training data for this group
            train_group = train_df[
                (train_df[store_col] == store_id) & (train_df[sku_col] == sku_id)
            ].sort_values(self.date_column).reset_index(drop=True)

            if len(train_group) == 0:
                group_info[(store_id, sku_id)] = {
                    'skipped': True,
                    'reason': 'No training data for this group'
                }
                continue

            # Get scaler for this group
            scaler = self._panel_scalers.get((store_id, sku_id))
            if scaler is None:
                # Fit new scaler on training data (should not happen if trained properly)
                train_values = train_group[sales_col].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaler.fit(train_values)
                self._panel_scalers[(store_id, sku_id)] = scaler

            # Need last `seq_length` values from training as context
            train_values = train_group[sales_col].values
            if len(train_values) < seq_length:
                # Pad with mean if insufficient history
                pad_length = seq_length - len(train_values)
                mean_val = train_values.mean() if len(train_values) > 0 else 0
                context = np.concatenate([np.full(pad_length, mean_val), train_values])
            else:
                context = train_values[-seq_length:]

            # Test values (no target, so we only have context + features)
            # For forecasting, we create sequences starting from the context
            # and rolling forward through the test period
            test_features = test_group.drop(columns=[store_col, sku_col, self.date_column, 'record_ID'],
                                             errors='ignore')
            # For pure LSTM forecasting, we only need the historical target values
            # The test observations will be predicted autoregressively

            # Create initial sequence from context
            context_scaled = scaler.transform(context.reshape(-1, 1)).flatten()

            # We'll generate one sequence per test row for autoregressive forecasting
            # Each test row needs a sequence of `seq_length` historical values
            n_test = len(test_group)
            test_sequences = []
            record_ids = test_group['record_ID'].values

            # Start with the context sequence
            current_seq = context_scaled.copy()

            for i in range(n_test):
                test_sequences.append(current_seq.copy())
                # For the next sequence, we would need the predicted value
                # But since we're preparing input for model.predict(), we just need
                # the sequences. The actual autoregressive prediction happens in the model.
                # However, for batch prediction, we need all sequences upfront.
                # The model's forecast_future handles autoregressive generation.
                # So we just return the last context sequence for forecast_future.
                # But for evaluating on test (if targets were available), we'd need
                # sequences with actual values. Since test has no targets, we return
                # the context for forecast_future.

                # Actually, for proper test sequence preparation where we want to
                # predict each test step, we should create sequences that include
                # the context + any available test features. But LSTM only uses
                # the target history. So we return the context sequence and let
                # forecast_future handle the autoregressive generation.

            # Return single context sequence per group for forecast_future
            all_X_test.append(context_scaled.reshape(1, seq_length, 1))
            all_record_ids.append(record_ids)
            group_info[(store_id, sku_id)] = {
                'skipped': False,
                'n_test': n_test,
                'train_context_length': len(context)
            }

        if all_X_test:
            X_test = np.concatenate(all_X_test, axis=0)
            record_ids = np.concatenate(all_record_ids)
        else:
            X_test = np.array([]).reshape(0, seq_length, 1)
            record_ids = np.array([])

        return X_test, record_ids, group_info

    # ------------------------------------------------------------------
    # Hybrid Model Data Preparation - Single Series
    # ------------------------------------------------------------------

    def prepare_hybrid_data(self, df, sales_col='Sales', test_size=0.2):
        """
        Split the raw demand series chronologically for Hybrid ARIMA+XGBoost evaluation.

        No shuffling. Returns raw (unscaled) arrays.
        """
        values = df[sales_col].values.astype(float)
        split_idx = int(len(values) * (1 - test_size))
        self.train_size = split_idx
        train_series = values[:split_idx]
        test_series = values[split_idx:]
        return train_series, test_series, split_idx

    # ------------------------------------------------------------------
    # Hybrid Model Data Preparation - Panel Data
    # ------------------------------------------------------------------

    def prepare_hybrid_data_panel(self, df, sales_col='Units_Sold', test_size=0.2,
                                   store_col='store_id', sku_col='sku_id'):
        """
        Prepare Hybrid model data for panel data.

        Returns dict mapping (store_id, sku_id) -> (train_series, test_series, split_idx)
        """
        self._is_panel = True
        self.store_column = store_col
        self.sku_column = sku_col
        self.sales_column = sales_col

        df = df.sort_values([store_col, sku_col, self.date_column]).reset_index(drop=True)

        panel_data = {}
        for (store_id, sku_id), group in df.groupby([store_col, sku_col]):
            group = group.sort_values(self.date_column).reset_index(drop=True)
            values = group[sales_col].values.astype(float)

            if len(values) < 20:
                panel_data[(store_id, sku_id)] = {
                    'skipped': True,
                    'reason': f'Insufficient data: {len(values)} rows',
                    'train_series': None,
                    'test_series': None,
                    'split_idx': 0
                }
                continue

            split_idx = int(len(values) * (1 - test_size))
            split_idx = max(split_idx, 10)
            split_idx = min(split_idx, len(values) - 1)

            train_series = values[:split_idx]
            test_series = values[split_idx:]

            panel_data[(store_id, sku_id)] = {
                'skipped': False,
                'train_series': train_series,
                'test_series': test_series,
                'split_idx': split_idx,
                'n_train': len(train_series),
                'n_test': len(test_series)
            }

        return panel_data

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