import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


class ExploratoryAnalysis:
    """Performs exploratory data analysis on time-series sales data."""

    def __init__(self):
        self.stats_summary = {}

    # ------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------

    def generate_summary_statistics(self, df, sales_col='Sales'):
        """
        Generate statistical summary of sales data.

        Returns:
            Plain dict (compatible with both dict-style and attribute access)
        """
        s = df[sales_col].dropna()
        result = {
            'Count': int(s.count()),
            'Mean': float(s.mean()),
            'Std Dev': float(s.std()),
            'Min': float(s.min()),
            'Q1 (25%)': float(s.quantile(0.25)),
            'Median': float(s.median()),
            'Q3 (75%)': float(s.quantile(0.75)),
            'Max': float(s.max()),
            'Skewness': float(s.skew()),
            'Kurtosis': float(s.kurtosis()),
        }
        self.stats_summary = result
        return result

    # ------------------------------------------------------------------
    # Outlier Detection
    # ------------------------------------------------------------------

    def detect_outliers(self, df, sales_col='Sales', method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method.

        Returns:
            DataFrame with added 'Outlier' boolean column
        """
        df = df.copy()

        if method == 'iqr':
            Q1 = df[sales_col].quantile(0.25)
            Q3 = df[sales_col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df['Outlier'] = (df[sales_col] < lower) | (df[sales_col] > upper)
        elif method == 'zscore':
            z = np.abs(stats.zscore(df[sales_col].fillna(0)))
            df['Outlier'] = z > threshold
        else:
            df['Outlier'] = False

        return df

    # ------------------------------------------------------------------
    # Core Plots
    # ------------------------------------------------------------------

    def plot_time_series(self, df, date_col='Date', sales_col='Sales',
                         title='Demand Over Time'):
        """Plot demand trend as a line chart."""
        fig = px.line(
            df, x=date_col, y=sales_col,
            title=title,
            labels={sales_col: 'Demand', date_col: 'Date'}
        )
        fig.update_traces(line_color='#1f77b4', line_width=1.5)
        fig.update_layout(hovermode='x unified', template='plotly_white',
                          margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def plot_distribution(self, df, sales_col='Sales', title='Demand Distribution'):
        """Histogram + KDE of demand values."""
        fig = px.histogram(
            df, x=sales_col, nbins=40, title=title,
            labels={sales_col: 'Demand'},
            color_discrete_sequence=['#1f77b4']
        )
        mean_val = df[sales_col].mean()
        fig.add_vline(x=mean_val, line_dash='dash', line_color='red',
                      annotation_text=f'Mean: {mean_val:.1f}',
                      annotation_position='top right')
        fig.update_layout(template='plotly_white',
                          margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def plot_seasonal_pattern(self, df, date_col='Date', sales_col='Sales',
                              period='Month'):
        """
        Bar chart of average demand by period.
        period: 'Month', 'Day' (day-of-week), or 'Week'
        """
        df_temp = df.copy()
        if period == 'Month':
            df_temp['_period'] = df_temp[date_col].dt.month
            x_labels = {i: m for i, m in enumerate(
                ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec'], 1)}
            x_title = 'Month'
        elif period == 'Day':
            df_temp['_period'] = df_temp[date_col].dt.dayofweek
            x_labels = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
            x_title = 'Day of Week'
        else:
            df_temp['_period'] = df_temp[date_col].dt.isocalendar().week.astype(int)
            x_labels = {}
            x_title = 'Week'

        avg = df_temp.groupby('_period')[sales_col].mean().reset_index()
        avg.columns = [x_title, 'Average Demand']

        if x_labels:
            avg[x_title] = avg[x_title].map(x_labels).fillna(avg[x_title].astype(str))

        fig = px.bar(
            avg, x=x_title, y='Average Demand',
            title=f'Average Demand by {x_title}',
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template='plotly_white',
                          margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def plot_monthly_boxplot(self, df, date_col='Date', sales_col='Sales'):
        """Box plot of demand spread by month."""
        df_temp = df.copy()
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        df_temp['Month'] = df_temp[date_col].dt.month.map(
            {i: m for i, m in enumerate(month_names, 1)}
        )
        fig = px.box(
            df_temp, x='Month', y=sales_col,
            title='Monthly Demand Distribution',
            labels={sales_col: 'Demand'},
            category_orders={'Month': month_names},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(template='plotly_white',
                          margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def plot_correlation_heatmap(self, df, numeric_cols=None):
        """Correlation heatmap of numerical columns."""
        if numeric_cols is None:
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 2:
            return None

        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=9)
        ))
        fig.update_layout(
            title='Feature Correlation Heatmap',
            template='plotly_white',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    def analyze_trend(self, df, date_col='Date', sales_col='Sales', window=30):
        """Overlay actual demand with short and long moving averages."""
        df = df.copy()
        long_w = max(window * 3, window + 1)
        df['MA_Short'] = df[sales_col].rolling(window=window, min_periods=1).mean()
        df['MA_Long'] = df[sales_col].rolling(window=long_w, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[sales_col],
            mode='lines', name='Actual',
            line=dict(color='#aec7e8', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['MA_Short'],
            mode='lines', name=f'{window}-day MA',
            line=dict(color='#1f77b4', dash='dash', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df['MA_Long'],
            mode='lines', name=f'{long_w}-day MA',
            line=dict(color='#d62728', dash='dot', width=2)
        ))
        fig.update_layout(
            title='Trend Analysis — Moving Averages',
            xaxis_title='Date', yaxis_title='Demand',
            hovermode='x unified', template='plotly_white',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    def plot_acf_pacf(self, df, sales_col='Sales', lags=40):
        """Return matplotlib figure with ACF and PACF plots."""
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        plot_acf(df[sales_col].dropna(), lags=min(lags, len(df) // 2 - 1),
                 ax=axes[0], color='#1f77b4')
        axes[0].set_title('Autocorrelation Function (ACF)')
        plot_pacf(df[sales_col].dropna(), lags=min(lags, len(df) // 2 - 1),
                  ax=axes[1], color='#1f77b4')
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        plt.tight_layout()
        return fig

    def generate_eda_report(self, df, date_col='Date', sales_col='Sales'):
        """Generate dict of all EDA outputs for programmatic use."""
        return {
            'summary_stats': self.generate_summary_statistics(df, sales_col),
            'outliers': self.detect_outliers(df, sales_col),
            'ts_plot': self.plot_time_series(df, date_col, sales_col),
            'distribution': self.plot_distribution(df, sales_col),
            'monthly_seasonal': self.plot_seasonal_pattern(df, date_col, sales_col, 'Month'),
            'dayofweek_seasonal': self.plot_seasonal_pattern(df, date_col, sales_col, 'Day'),
            'trend_analysis': self.analyze_trend(df, date_col, sales_col),
        }


if __name__ == "__main__":
    print("EDA module loaded successfully")
