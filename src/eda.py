import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


class ExploratoryAnalysis:
    """Performs exploratory data analysis on time-series sales data."""
    
    def __init__(self):
        sns.set_style("whitegrid")
        self.stats_summary = {}
    
    def generate_summary_statistics(self, df, sales_col='Sales'):
        """Generate statistical summary of sales data."""
        stats_dict = {
            'Count': df[sales_col].count(),
            'Mean': df[sales_col].mean(),
            'Std Dev': df[sales_col].std(),
            'Min': df[sales_col].min(),
            '25%': df[sales_col].quantile(0.25),
            'Median': df[sales_col].median(),
            '75%': df[sales_col].quantile(0.75),
            'Max': df[sales_col].max(),
            'Skewness': df[sales_col].skew(),
            'Kurtosis': df[sales_col].kurtosis()
        }
        self.stats_summary = stats_dict
        return pd.Series(stats_dict)
    
    def detect_outliers(self, df, sales_col='Sales', method='iqr', threshold=1.5):
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input dataframe
            sales_col: Sales column name
            method: 'iqr' or 'zscore'
            threshold: Multiplier for IQR (default 1.5) or z-score limit (default 1.5)
        
        Returns:
            Dataframe with outlier flag
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[sales_col].quantile(0.25)
            Q3 = df[sales_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df['Outlier'] = (df[sales_col] < lower_bound) | (df[sales_col] > upper_bound)
        
        elif method == 'zscore':
            df['Z_Score'] = np.abs(stats.zscore(df[sales_col]))
            df['Outlier'] = df['Z_Score'] > threshold
            df = df.drop('Z_Score', axis=1)
        
        return df
    
    def plot_time_series(self, df, date_col='Date', sales_col='Sales', title='Sales Over Time'):
        """Plot time series with Plotly."""
        fig = px.line(df, x=date_col, y=sales_col, title=title,
                     labels={sales_col: 'Sales', date_col: 'Date'})
        fig.update_layout(hovermode='x unified', template='plotly_white')
        return fig
    
    def plot_seasonal_pattern(self, df, date_col='Date', sales_col='Sales', period='Month'):
        """Plot seasonal patterns (daily, weekly, monthly)."""
        df_temp = df.copy()
        df_temp[period] = df_temp[date_col].dt.month if period == 'Month' else \
                          df_temp[date_col].dt.dayofweek if period == 'Day' else \
                          df_temp[date_col].dt.isocalendar().week
        
        avg_by_period = df_temp.groupby(period)[sales_col].mean()
        
        fig = px.bar(x=avg_by_period.index, y=avg_by_period.values,
                    title=f'Average Sales by {period}',
                    labels={'x': period, 'y': 'Average Sales'})
        fig.update_layout(template='plotly_white')
        return fig
    
    def plot_distribution(self, df, sales_col='Sales', title='Sales Distribution'):
        """Plot distribution of sales values."""
        fig = px.histogram(df, x=sales_col, nbins=50, title=title,
                          labels={sales_col: 'Sales'})
        fig.add_vline(x=df[sales_col].mean(), line_dash="dash", 
                     annotation_text="Mean", annotation_position="top right")
        fig.update_layout(template='plotly_white')
        return fig
    
    def plot_correlation_heatmap(self, df, numeric_cols=None):
        """Plot correlation heatmap of numerical columns."""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title='Correlation Heatmap', template='plotly_white')
        return fig
    
    def plot_acf_pacf(self, df, sales_col='Sales', lags=40):
        """Plot ACF and PACF for stationarity analysis."""
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        plot_acf(df[sales_col], lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(df[sales_col], lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    
    def analyze_trend(self, df, date_col='Date', sales_col='Sales', window=30):
        """Analyze trend using moving average."""
        df = df.copy()
        df['MA_Short'] = df[sales_col].rolling(window=window).mean()
        df['MA_Long'] = df[sales_col].rolling(window=window*3).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[date_col], y=df[sales_col],
                                mode='lines', name='Actual Sales',
                                line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=df[date_col], y=df['MA_Short'],
                                mode='lines', name=f'{window}-day MA',
                                line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=df[date_col], y=df['MA_Long'],
                                mode='lines', name=f'{window*3}-day MA',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(title='Trend Analysis with Moving Averages',
                         hovermode='x unified', template='plotly_white')
        return fig
    
    def compare_products(self, df, date_col='Date', sales_col='Sales', product_col='Product'):
        """Compare sales across different products."""
        if product_col not in df.columns:
            return None
        
        fig = px.line(df, x=date_col, y=sales_col, color=product_col,
                     title='Sales Comparison by Product',
                     labels={sales_col: 'Sales', date_col: 'Date'})
        fig.update_layout(hovermode='x unified', template='plotly_white')
        return fig
    
    def generate_eda_report(self, df, date_col='Date', sales_col='Sales'):
        """Generate comprehensive EDA report."""
        report = {
            'summary_stats': self.generate_summary_statistics(df, sales_col),
            'outliers': self.detect_outliers(df, sales_col),
            'ts_plot': self.plot_time_series(df, date_col, sales_col),
            'distribution': self.plot_distribution(df, sales_col),
            'monthly_seasonal': self.plot_seasonal_pattern(df, date_col, sales_col, 'Month'),
            'trend_analysis': self.analyze_trend(df, date_col, sales_col)
        }
        return report


if __name__ == "__main__":
    print("EDA module loaded successfully")
