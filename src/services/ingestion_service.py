"""
Ingestion Service for AI Demand Forecasting.

Handles data loading, validation, cleaning, and feature engineering.
Pure Python service with no Streamlit dependencies.
"""
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from src.preprocessing import DataPreprocessor

# Column aliases for auto-detection
KNOWN_DATE_ALIASES = [
    'date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP',
    'order_date', 'OrderDate', 'order_Date', 'week_start_date',
    'ds', 'time', 'Time', 'period', 'Period',
]
KNOWN_DEMAND_ALIASES = [
    'sales', 'Sales', 'SALES',
    'unit_sales', 'units_sold', 'UnitSales', 'Units',
    'quantity', 'Quantity', 'QUANTITY', 'qty', 'Qty',
    'Sales_Quantity', 'sales_quantity',
    'demand', 'Demand', 'DEMAND',
    'Weekly_Sales', 'weekly_sales',
    'cnt', 'count', 'Count', 'sold',
    'revenue', 'Revenue',
]
M5_MARKER_COLS = {'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'}

DUCK_THRESHOLD = 80 * 1024 * 1024  # 80 MB


class IngestionError(Exception):
    """Raised when data ingestion fails."""
    pass


class IngestionService:
    """
    Service for ingesting and preprocessing time-series demand data.
    
    Encapsulates all data loading, validation, cleaning, and feature engineering
    logic. No Streamlit dependencies.
    """

    def __init__(self, use_duckdb: bool = True, min_rows_lstm: int = 100):
        """
        Initialize the ingestion service.
        
        Args:
            use_duckdb: Whether to use DuckDB for large files (>80MB)
            min_rows_lstm: Minimum rows required for LSTM training
        """
        self.use_duckdb = use_duckdb and DUCKDB_AVAILABLE
        self.min_rows_lstm = min_rows_lstm
        self.preprocessor = DataPreprocessor()
        self._last_processed_info: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_csv(
        self,
        filepath: str | Path,
        date_col: str | None = None,
        sales_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load and preprocess a single CSV file.
        
        Args:
            filepath: Path to CSV file
            date_col: Date column name (auto-detected if None)
            sales_col: Sales/demand column name (auto-detected if None)
            
        Returns:
            Tuple of (cleaned_dataframe, metadata_dict)
        """
        return self._load_and_process(
            sources=[filepath],
            date_col=date_col,
            sales_col=sales_col,
        )

    def load_from_multiple_csvs(
        self,
        filepaths: list[str | Path],
        date_col: str | None = None,
        sales_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load and merge multiple CSV files.
        
        Args:
            filepaths: List of CSV file paths
            date_col: Date column name (auto-detected if None)
            sales_col: Sales/demand column name (auto-detected if None)
            
        Returns:
            Tuple of (cleaned_dataframe, metadata_dict)
        """
        return self._load_and_process(
            sources=filepaths,
            date_col=date_col,
            sales_col=sales_col,
        )

    def load_from_folder(
        self,
        folder_path: str | Path,
        date_col: str | None = None,
        sales_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load all CSV files from a folder and merge them.
        
        Args:
            folder_path: Path to folder containing CSV files
            date_col: Date column name (auto-detected if None)
            sales_col: Sales/demand column name (auto-detected if None)
            
        Returns:
            Tuple of (cleaned_dataframe, metadata_dict)
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise IngestionError(f"Folder not found: {folder_path}")

        csv_files = sorted(folder.glob('*.csv'))
        if not csv_files:
            raise IngestionError(f"No CSV files found in: {folder_path}")

        return self._load_and_process(
            sources=[str(f) for f in csv_files],
            date_col=date_col,
            sales_col=sales_col,
        )

    def load_from_file_objects(
        self,
        file_objects: list[Any],
        date_col: str | None = None,
        sales_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load from file-like objects (e.g., Streamlit uploads).
        
        Args:
            file_objects: List of file-like objects with read/seek methods
            date_col: Date column name (auto-detected if None)
            sales_col: Sales/demand column name (auto-detected if None)
            
        Returns:
            Tuple of (cleaned_dataframe, metadata_dict)
        """
        return self._load_and_process(
            sources=file_objects,
            date_col=date_col,
            sales_col=sales_col,
        )

    def validate_data(
        self,
        df: pd.DataFrame,
        date_col: str,
        sales_col: str,
    ) -> tuple[bool, list[str]]:
        """
        Validate a dataframe before processing.
        
        Args:
            df: Dataframe to validate
            date_col: Name of date column
            sales_col: Name of sales column
            
        Returns:
            Tuple of (is_valid, messages)
        """
        return self.preprocessor.validate_data(df, date_col, sales_col)

    def clean_data(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date',
        sales_col: str = 'Sales',
    ) -> pd.DataFrame:
        """
        Clean a dataframe: parse dates, coerce numeric, deduplicate, fill missing.
        
        Args:
            df: Input dataframe
            date_col: Date column name
            sales_col: Sales column name
            
        Returns:
            Cleaned dataframe
        """
        return self.preprocessor.clean_data(df, date_col, sales_col)

    def prepare_lstm_data(
        self,
        df: pd.DataFrame,
        sales_col: str = 'Sales',
        seq_length: int = 30,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Prepare LSTM training data with no leakage.
        
        Args:
            df: Cleaned dataframe
            sales_col: Sales column name
            seq_length: Sequence length for LSTM
            test_size: Test split ratio
            
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        return self.preprocessor.prepare_lstm_data(
            df, sales_col, seq_length, test_size
        )

    def prepare_hybrid_data(
        self,
        df: pd.DataFrame,
        sales_col: str = 'Sales',
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Prepare Hybrid model data (raw series split).
        
        Args:
            df: Cleaned dataframe
            sales_col: Sales column name
            test_size: Test split ratio
            
        Returns:
            train_series, test_series, split_idx
        """
        return self.preprocessor.prepare_hybrid_data(df, sales_col, test_size)

    def create_time_features(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date',
    ) -> pd.DataFrame:
        """Create time-based features."""
        return self.preprocessor.create_time_features(df, date_col)

    def create_lag_features(
        self,
        df: pd.DataFrame,
        sales_col: str = 'Sales',
        lags: tuple[int, ...] = (1, 7, 14, 30),
    ) -> pd.DataFrame:
        """Create lag features."""
        return self.preprocessor.create_lag_features(df, sales_col, lags)

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        sales_col: str = 'Sales',
        windows: tuple[int, ...] = (7, 14, 30),
    ) -> pd.DataFrame:
        """Create rolling mean/std features."""
        return self.preprocessor.create_rolling_features(df, sales_col, windows)

    def get_last_processed_info(self) -> dict | None:
        """Get info about the last processed dataset."""
        return self._last_processed_info

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _load_and_process(
        self,
        sources: list[Any],
        date_col: str | None = None,
        sales_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Load and process data from various sources.
        
        Args:
            sources: List of file paths or file objects
            date_col: Date column name (auto-detected if None)
            sales_col: Sales column name (auto-detected if None)
            
        Returns:
            Tuple of (cleaned_dataframe, metadata_dict)
        """
        if not sources:
            raise IngestionError("No data sources provided")

        # Detect columns from first source
        primary_src = sources[0]
        all_cols = self._read_columns(primary_src)
        is_wide = self._detect_wide_format(all_cols)

        # Auto-detect columns
        if date_col is None or sales_col is None:
            auto_dc, auto_sc = self._auto_detect_cols(all_cols)
            date_col = date_col or auto_dc
            sales_col = sales_col or auto_sc

        if date_col is None or sales_col is None:
            raise IngestionError(
                f"Could not auto-detect date/demand columns. "
                f"Available: {all_cols}. Please specify explicitly."
            )

        # Load and merge data
        merged_df = self._load_and_merge(sources, date_col, sales_col, is_wide)

        if merged_df.empty:
            raise IngestionError("No data loaded after merging")

        # Validate
        is_valid, messages = self.validate_data(merged_df, date_col, sales_col)
        if not is_valid:
            errors = [m for m in messages if not m.startswith('Warning')]
            raise IngestionError(f"Validation failed: {'; '.join(errors)}")

        # Clean
        df_clean = self.clean_data(merged_df, date_col, sales_col)

        # Check minimum rows
        if len(df_clean) < self.min_rows_lstm:
            raise IngestionError(
                f"Insufficient data: {len(df_clean)} rows. "
                f"Minimum required: {self.min_rows_lstm}"
            )

        # Store metadata
        self._last_processed_info = {
            'n_rows': len(df_clean),
            'date_range': (
                df_clean[date_col].min().isoformat(),
                df_clean[date_col].max().isoformat(),
            ),
            'date_col': date_col,
            'sales_col': sales_col,
            'sources': [self._src_name(s) for s in sources],
            'warnings': [m for m in messages if m.startswith('Warning')],
        }

        return df_clean, self._last_processed_info

    def _read_columns(self, src: Any) -> list[str]:
        """Read column names from a file or file object."""
        try:
            if self._is_file_obj(src):
                src.seek(0)
                cols = list(pd.read_csv(src, nrows=0).columns)
                src.seek(0)
                return cols
            return list(pd.read_csv(src, nrows=0).columns)
        except FileNotFoundError:
            raise IngestionError(f"File not found: {src}")
        except Exception as e:
            raise IngestionError(f"Failed to read columns: {e}")

    def _is_file_obj(self, src: Any) -> bool:
        """Check if source is a file-like object."""
        return hasattr(src, 'read') and hasattr(src, 'seek')

    def _src_name(self, src: Any) -> str:
        """Get source name for metadata."""
        if self._is_file_obj(src):
            return getattr(src, 'name', 'uploaded_file')
        return os.path.basename(str(src))

    def _src_size(self, src: Any) -> int:
        """Get source size in bytes."""
        if self._is_file_obj(src):
            if hasattr(src, 'size') and src.size is not None:
                return int(src.size)
            try:
                return os.path.getsize(str(src))
            except OSError:
                return 0
        return os.path.getsize(str(src))

    def _detect_wide_format(self, columns: list[str]) -> bool:
        """Detect M5 wide format (d_* columns)."""
        col_set = {c.lower() for c in columns}
        day_cols = [c for c in columns if c.startswith('d_') and c[2:].isdigit()]
        return len(day_cols) > 100 and bool(col_set & {c.lower() for c in M5_MARKER_COLS})

    def _auto_detect_cols(self, columns: list[str]) -> tuple[str | None, str | None]:
        """Auto-detect date and sales columns from aliases."""
        lower_map = {c.lower(): c for c in columns}
        date_col = next((lower_map[a.lower()] for a in KNOWN_DATE_ALIASES if a.lower() in lower_map), None)
        demand_col = next((lower_map[a.lower()] for a in KNOWN_DEMAND_ALIASES if a.lower() in lower_map), None)
        return date_col, demand_col

    def _load_and_merge(
        self,
        sources: list[Any],
        date_col: str,
        sales_col: str,
        is_wide: bool,
    ) -> pd.DataFrame:
        """Load and merge data from multiple sources."""
        dfs = []

        for src in sources:
            try:
                if is_wide:
                    df = self._load_wide_format(src, date_col, sales_col)
                elif len(sources) > 1:
                    df = self._load_single_file(src, date_col, sales_col)
                else:
                    df = self._load_single_file_optimized(src, date_col, sales_col)

                if not df.empty and date_col in df.columns and sales_col in df.columns:
                    dfs.append(df[[date_col, sales_col]])
            except Exception:
                # Skip files that fail to load
                continue

        if not dfs:
            return pd.DataFrame(columns=[date_col, sales_col])

        merged = pd.concat(dfs, ignore_index=True)
        return merged

    def _load_wide_format(self, src: Any, date_col: str, sales_col: str) -> pd.DataFrame:
        """Load M5 wide format and melt to long."""
        if self._is_file_obj(src):
            src.seek(0)
            df_wide = pd.read_csv(src, low_memory=False)
            src.seek(0)
        else:
            df_wide = pd.read_csv(src, low_memory=False)

        day_cols = [c for c in df_wide.columns if c.startswith('d_') and c[2:].isdigit()]
        if not day_cols:
            return pd.DataFrame()

        melted = df_wide[day_cols].sum(axis=0)
        melted.index = range(len(melted))
        return pd.DataFrame({date_col: melted.index, sales_col: melted.values})

    def _load_single_file(self, src: Any, date_col: str, sales_col: str) -> pd.DataFrame:
        """Load a single CSV file."""
        if self._is_file_obj(src):
            src.seek(0)
            df = pd.read_csv(src, low_memory=False)
            src.seek(0)
        else:
            df = pd.read_csv(src, low_memory=False)
        return df

    def _load_single_file_optimized(self, src: Any, date_col: str, sales_col: str) -> pd.DataFrame:
        """Load single file with DuckDB or chunked fallback for large files."""
        size = self._src_size(src)

        if self.use_duckdb and size > DUCK_THRESHOLD:
            return self._load_via_duckdb(src, date_col, sales_col)
        elif size > DUCK_THRESHOLD:
            return self._load_chunked(src, date_col, sales_col)
        else:
            return self._load_single_file(src, date_col, sales_col)

    def _load_via_duckdb(self, src: Any, date_col: str, sales_col: str) -> pd.DataFrame:
        """Load large file via DuckDB."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            if self._is_file_obj(src):
                src.seek(0)
                tmp.write(src.read())
                src.seek(0)
            else:
                with open(str(src), 'rb') as f:
                    tmp.write(f.read())
            tmp_path = tmp.name

        try:
            con = duckdb.connect()
            df = con.execute(
                f"SELECT {date_col}, SUM(CAST({sales_col} AS DOUBLE)) AS {sales_col} "
                f"FROM read_csv_auto('{tmp_path}') "
                f"WHERE {sales_col} IS NOT NULL AND CAST({sales_col} AS DOUBLE) >= 0 "
                f"GROUP BY {date_col} ORDER BY {date_col}"
            ).df()
            con.close()
            return df
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _load_chunked(self, src: Any, date_col: str, sales_col: str) -> pd.DataFrame:
        """Load large file in chunks."""
        if self._is_file_obj(src):
            src.seek(0)
            file_bytes = src.read()
            src.seek(0)
        else:
            with open(str(src), 'rb') as f:
                file_bytes = f.read()

        return self._aggregate_chunked(file_bytes, self._src_name(src), date_col, sales_col)

    def _aggregate_chunked(
        self,
        file_bytes: bytes,
        filename: str,
        date_col: str,
        sales_col: str,
        chunksize: int = 250_000,
    ) -> pd.DataFrame:
        """Aggregate chunked CSV reads."""
        buf = io.BytesIO(file_bytes)
        agg = None
        for chunk in pd.read_csv(buf, usecols=[date_col, sales_col],
                                  chunksize=chunksize, low_memory=False):
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
            chunk[sales_col] = pd.to_numeric(chunk[sales_col], errors='coerce')
            chunk = chunk.dropna().query(f'{sales_col} >= 0')
            if chunk.empty:
                continue
            grp = chunk.groupby(date_col)[sales_col].sum()
            agg = grp if agg is None else agg.add(grp, fill_value=0)

        if agg is None:
            return pd.DataFrame(columns=[date_col, sales_col])
        return agg.reset_index().sort_values(date_col).reset_index(drop=True)
