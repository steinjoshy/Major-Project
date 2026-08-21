"""Tests for IngestionService."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import io

ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.services.ingestion_service import IngestionService, IngestionError


class TestIngestionService:
    """Test IngestionService functionality."""
    
    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Sales': np.random.poisson(50, 100),
            'Price': np.random.uniform(10, 100, 100),
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def service(self):
        """Create IngestionService instance."""
        return IngestionService(use_duckdb=False, min_rows_lstm=50)
    
    def test_load_from_csv(self, service, sample_csv):
        """Test loading a single CSV file."""
        df, meta = service.load_from_csv(sample_csv, date_col='Date', sales_col='Sales')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'Date' in df.columns
        assert 'Sales' in df.columns
        assert meta['n_rows'] == 100
        assert meta['date_col'] == 'Date'
        assert meta['sales_col'] == 'Sales'
    
    def test_load_from_csv_auto_detect(self, service, sample_csv):
        """Test auto column detection."""
        df, meta = service.load_from_csv(sample_csv)
        
        assert meta['date_col'] == 'Date'
        assert meta['sales_col'] == 'Sales'
    
    def test_load_nonexistent_file(self, service):
        """Test loading nonexistent file raises error."""
        with pytest.raises(IngestionError):
            service.load_from_csv('/nonexistent/path.csv')
    
    def test_validate_data_success(self, service):
        """Test data validation with valid data."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Sales': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        
        is_valid, messages = service.validate_data(df, 'Date', 'Sales')
        assert is_valid is True
        assert isinstance(messages, list)
    
    def test_validate_data_missing_date_col(self, service):
        """Test validation fails with missing date column."""
        df = pd.DataFrame({'Sales': [10, 20, 30]})
        
        is_valid, messages = service.validate_data(df, 'Date', 'Sales')
        assert is_valid is False
        assert any('Date column' in m for m in messages)
    
    def test_validate_data_missing_sales_col(self, service):
        """Test validation fails with missing sales column."""
        df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02']})
        
        is_valid, messages = service.validate_data(df, 'Date', 'Sales')
        assert is_valid is False
        assert any('Sales' in m for m in messages)
    
    def test_clean_data(self, service):
        """Test data cleaning."""
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03'],
            'Sales': [10, np.nan, 20, -5],
        })
        
        df_clean = service.clean_data(df, 'Date', 'Sales')
        
        # Duplicates removed (summed): 2023-01-01 = 10+20=30
        # Missing values filled: 2023-01-02 NaN filled with ffill (30)
        # Negative values removed: 2023-01-03 -5 removed
        assert len(df_clean) == 2
        assert (df_clean['Sales'] >= 0).all()
        assert df_clean['Sales'].isna().sum() == 0
    
    def test_prepare_lstm_data(self, service):
        """Test LSTM data preparation."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': np.random.poisson(50, 100),
        })
        
        X_tr, X_te, y_tr, y_te, scaler = service.prepare_lstm_data(
            df, 'Sales', seq_length=10, test_size=0.2
        )
        
        assert X_tr.ndim == 3  # (n_samples, seq_length, 1)
        assert X_te.ndim == 3
        assert X_tr.shape[1] == 10  # seq_length
        assert X_te.shape[1] == 10
        assert y_tr.ndim == 1
        assert y_te.ndim == 1
        assert scaler is not None
    
    def test_prepare_hybrid_data(self, service):
        """Test Hybrid data preparation."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': np.random.poisson(50, 100),
        })
        
        train, test, split = service.prepare_hybrid_data(df, 'Sales', test_size=0.2)
        
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)
        assert len(train) + len(test) == 100
        assert split == len(train)
    
    def test_create_time_features(self, service):
        """Test time feature creation."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Sales': range(10),
        })
        
        df_feat = service.create_time_features(df, 'Date')
        
        expected = ['Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend']
        for col in expected:
            assert col in df_feat.columns
    
    def test_create_lag_features(self, service):
        """Test lag feature creation."""
        df = pd.DataFrame({'Sales': range(50)})
        
        df_lag = service.create_lag_features(df, 'Sales', lags=(1, 7, 14))
        
        assert 'Sales_Lag_1' in df_lag.columns
        assert 'Sales_Lag_7' in df_lag.columns
        assert 'Sales_Lag_14' in df_lag.columns
        assert df_lag['Sales_Lag_1'].isna().sum() == 0
    
    def test_create_rolling_features(self, service):
        """Test rolling feature creation."""
        df = pd.DataFrame({'Sales': [10, 20, 30, 40, 50] * 4})
        
        df_roll = service.create_rolling_features(df, 'Sales', windows=(3, 5))
        
        assert 'Rolling_Mean_3' in df_roll.columns
        assert 'Rolling_Std_3' in df_roll.columns
        assert df_roll['Rolling_Mean_3'].isna().sum() == 0
    
    def test_load_multiple_csvs(self, service):
        """Test loading multiple CSV files."""
        # Create two temp files
        df1 = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=50),
            'Sales': np.random.poisson(50, 50),
        })
        df2 = pd.DataFrame({
            'Date': pd.date_range('2023-02-20', periods=50),
            'Sales': np.random.poisson(50, 50),
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_1.csv', delete=False) as f1:
            df1.to_csv(f1.name, index=False)
            path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2.csv', delete=False) as f2:
            df2.to_csv(f2.name, index=False)
            path2 = f2.name
        
        try:
            df, meta = service.load_from_multiple_csvs([path1, path2], date_col='Date', sales_col='Sales')
            assert len(df) == 100
            assert meta['n_rows'] == 100
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)
    
    def test_load_from_folder(self, service):
        """Test loading from folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df1 = pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=30),
                'Sales': np.random.poisson(50, 30),
            })
            df2 = pd.DataFrame({
                'Date': pd.date_range('2023-02-01', periods=30),
                'Sales': np.random.poisson(50, 30),
            })
            
            df1.to_csv(Path(tmpdir) / 'file1.csv', index=False)
            df2.to_csv(Path(tmpdir) / 'file2.csv', index=False)
            
            df, meta = service.load_from_folder(tmpdir, date_col='Date', sales_col='Sales')
            assert len(df) == 60
    
    def test_insufficient_rows_raises(self, service):
        """Test that insufficient rows raises error."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Sales': range(10),
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        
        try:
            with pytest.raises(IngestionError, match="Insufficient data"):
                service.load_from_csv(path, date_col='Date', sales_col='Sales')
        finally:
            Path(path).unlink(missing_ok=True)
    
    def test_file_object_loading(self, service):
        """Test loading from file objects."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=50),
            'Sales': np.random.poisson(50, 50),
        })
        
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        # Wrap in a file-like object
        file_obj = io.BytesIO(buffer.getvalue().encode())
        file_obj.name = 'test.csv'
        
        loaded_df, meta = service.load_from_file_objects([file_obj], date_col='Date', sales_col='Sales')
        assert len(loaded_df) == 50


class TestIngestionServiceM5Format:
    """Test M5 wide format detection and melting."""
    
    @pytest.fixture
    def m5_csv(self):
        """Create M5 wide format CSV."""
        # M5 format: id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ...
        n_days = 200
        df = pd.DataFrame({
            'id': ['FOODS_1_001_CA_1'] * 10,
            'item_id': ['FOODS_1_001'] * 10,
            'dept_id': ['FOODS_1'] * 10,
            'cat_id': ['FOODS'] * 10,
            'store_id': ['CA_1'] * 10,
            'state_id': ['CA'] * 10,
        })
        
        # Add day columns
        for d in range(1, n_days + 1):
            df[f'd_{d}'] = np.random.poisson(5, 10)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)
    
    def test_m5_detection(self, service, m5_csv):
        """Test M5 wide format auto-detection."""
        # Read columns to verify detection
        all_cols = service._read_columns(m5_csv)
        is_wide = service._detect_wide_format(all_cols)
        assert is_wide is True
    
    def test_m5_loading(self, service, m5_csv):
        """Test M5 format loading and melting."""
        # For M5 format, we need to specify columns explicitly since there are no standard date/sales columns
        df, meta = service.load_from_csv(m5_csv, date_col='date', sales_col='sales')
        
        # Should be melted to daily aggregates
        assert 'date' in df.columns or 'Date' in df.columns
        assert 'sales' in df.columns or 'Sales' in df.columns


class TestIngestionServiceEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, service):
        """Test validation with empty dataframe."""
        df = pd.DataFrame()
        is_valid, messages = service.validate_data(df, 'Date', 'Sales')
        assert is_valid is False
        assert any('empty' in m.lower() for m in messages)
    
    def test_constant_sales_warning(self, service):
        """Test warning for constant sales."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': [50] * 100,
        })
        
        is_valid, messages = service.validate_data(df, 'Date', 'Sales')
        assert is_valid is True
        assert any('constant' in m.lower() or 'zero variance' in m.lower() for m in messages)
    
    def test_high_missing_rate_warning(self, service):
        """Test warning for high missing rate."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': [np.nan if i % 2 == 0 else 10 for i in range(100)],
        })
        
        is_valid, messages = service.validate_data(df, 'Date', 'Sales')
        assert is_valid is True
        assert any('missing' in m.lower() for m in messages)
    
    def test_last_processed_info(self):
        """Test getting last processed info."""
        # Use a fresh service instance to avoid fixture state pollution
        from src.services.ingestion_service import IngestionService
        service = IngestionService(use_duckdb=False, min_rows_lstm=50)
        
        assert service.get_last_processed_info() is None
        
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Sales': np.random.poisson(50, 100),
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        
        try:
            service.load_from_csv(path, date_col='Date', sales_col='Sales')
            info = service.get_last_processed_info()
            
            assert info is not None
            assert info['n_rows'] == 100
            assert info['date_col'] == 'Date'
            assert info['sales_col'] == 'Sales'
        finally:
            Path(path).unlink(missing_ok=True)