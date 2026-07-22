import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta


class DataPreprocessor:
    """Handles data preprocessing and feature engineering for time-series forecasting."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.date_column = None
        self.sales_column = None
    
    def load_data(self, filepath):
        """Load CSV data."""
        return pd.read_csv(filepath)
    
    def clean_data(self, df, date_col='Date', sales_col='Sales'):
        """
        Clean dataset: handle missing values, duplicates, and format dates.
        
        Args:
            df: Input dataframe
            date_col: Name of date column
            sales_col: Name of sales/quantity column
        
        Returns:
            Cleaned dataframe
        """
        self.date_column = date_col
        self.sales_column = sales_col
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[date_col]).copy()
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(by=date_col).reset_index(drop=True)
        
        # Handle missing values in sales column
        if df[sales_col].isnull().sum() > 0:
            df[sales_col] = df[sales_col].fillna(df[sales_col].mean())
        
        # Remove negative sales values
        df = df[df[sales_col] >= 0].copy()
        
        return df
    
    def create_time_features(self, df, date_col='Date'):
        """Create time-based features (day, week, month, quarter)."""
        df = df.copy()
        df['Year'] = df[date_col].dt.year
        df['Month'] = df[date_col].dt.month
        df['Week'] = df[date_col].dt.isocalendar().week
        df['Day'] = df[date_col].dt.day
        df['DayOfWeek'] = df[date_col].dt.dayofweek
        df['Quarter'] = df[date_col].dt.quarter
        return df
    
    def create_lag_features(self, df, sales_col='Sales', lags=[1, 7, 30]):
        """Create lag features from previous sales values."""
        df = df.copy()
        for lag in lags:
            df[f'Sales_Lag_{lag}'] = df[sales_col].shift(lag)
        
        # Drop NaN rows created by lag features
        df = df.dropna().reset_index(drop=True)
        return df
    
    def scale_data(self, X, fit=True):
        """Scale numerical data using MinMaxScaler."""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def inverse_scale(self, X_scaled):
        """Inverse transform scaled data back to original scale."""
        return self.scaler.inverse_transform(X_scaled)
    
    def prepare_lstm_data(self, df, sales_col='Sales', seq_length=30):
        """
        Prepare data for LSTM: create sequences of data.
        
        Args:
            df: Preprocessed dataframe
            sales_col: Name of sales column
            seq_length: Length of sequences to create
        
        Returns:
            X, y: Sequences and corresponding target values
        """
        data = df[sales_col].values.reshape(-1, 1)
        data_scaled = self.scale_data(data, fit=True)
        
        X, y = [], []
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[i:i+seq_length])
            y.append(data_scaled[i+seq_length, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_traditional_ml_data(self, df, sales_col='Sales', target_col='Sales'):
        """Prepare data for ARIMA/XGBoost models."""
        # Select features (exclude date and original sales if duplicated)
        feature_cols = [col for col in df.columns if col not in [self.date_column, target_col, 'index']]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y
    
    def train_test_split_data(self, X, y, test_size=0.2):
        """Split data into train (80%) and test (20%) sets."""
        return train_test_split(X, y, test_size=test_size, shuffle=False, 
                               random_state=self.random_state)
    
    def get_date_range(self, df, date_col='Date'):
        """Get date range from dataframe."""
        return df[date_col].min(), df[date_col].max()
    
    def resample_data(self, df, date_col='Date', sales_col='Sales', freq='D'):
        """Resample time-series data to specified frequency."""
        df = df.copy()
        df = df.set_index(date_col)
        df_resampled = df[sales_col].resample(freq).sum()
        return df_resampled.reset_index()


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load sample data
    # df = preprocessor.load_data('data/sample_data/sales.csv')
    # df = preprocessor.clean_data(df)
    # df = preprocessor.create_time_features(df)
    # df = preprocessor.create_lag_features(df)
    # X_lstm, y_lstm = preprocessor.prepare_lstm_data(df)
    # X_ml, y_ml = preprocessor.prepare_traditional_ml_data(df)
    # X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(X_ml, y_ml)
    print("Data preprocessing module loaded successfully")
