import pandas as pd
import numpy as np
from datetime import datetime


def get_file_info(df):
    """Get basic information about a dataframe."""
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }


def format_metrics(metrics_dict, decimals=2):
    """Format metrics dictionary for display."""
    formatted = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            formatted[key] = round(value, decimals)
        else:
            formatted[key] = value
    return formatted


def save_forecast_results(df, filename):
    """Save forecast results to CSV."""
    df.to_csv(filename, index=False)
    return f"Results saved to {filename}"


def validate_csv_columns(df, required_cols):
    """Validate that CSV has required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "All required columns present"


def create_date_range(start_date, end_date, freq='D'):
    """Create date range for forecasting."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def aggregate_data(df, date_col, sales_col, freq='D'):
    """Aggregate data to specified frequency."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp = df_temp.set_index(date_col)
    return df_temp[sales_col].resample(freq).sum().reset_index()


def calculate_growth_rate(values):
    """Calculate growth rate from values."""
    if len(values) < 2:
        return 0
    return ((values[-1] - values[0]) / values[0]) * 100


def round_to_nearest(value, base=5):
    """Round value to nearest base."""
    return round(value / base) * base


if __name__ == "__main__":
    print("Utils module loaded successfully")
