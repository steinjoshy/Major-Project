"""
Model Comparison Service for AI Demand Forecasting.

Handles model evaluation, comparison, and best model selection.
Pure Python service with no Streamlit dependencies.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    n_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Model': self.model_name,
            'MAE': round(self.mae, 4),
            'RMSE': round(self.rmse, 4),
            'MAPE (%)': round(self.mape, 2),
            'N Samples': self.n_samples,
        }


@dataclass
class ComparisonResult:
    """Container for model comparison results."""
    metrics_table: pd.DataFrame
    best_model: str
    best_metrics: ModelMetrics
    improvement_pct: float
    metric_used: str = 'RMSE'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'comparison_table': self.metrics_table.to_dict('records'),
            'best_model': self.best_model,
            'best_metrics': self.best_metrics.to_dict(),
            'improvement_pct': self.improvement_pct,
            'metric_used': self.metric_used,
        }


class ModelComparisonService:
    """
    Service for evaluating and comparing forecasting models.
    
    Provides RMSE, MAE, MAPE metrics and model comparison.
    No Streamlit dependencies.
    """
    
    def __init__(self):
        """Initialize the comparison service."""
        self._results: Dict[str, ModelMetrics] = {}
        self._last_comparison: Optional[ComparisonResult] = None
    
    # ------------------------------------------------------------------
    # Metric Calculations
    # ------------------------------------------------------------------
    
    @staticmethod
    def rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
        """Root Mean Squared Error."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
        """Mean Absolute Error."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
        """
        Mean Absolute Percentage Error (%).
        Avoids division by zero using epsilon.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        eps = 1e-8
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    
    def evaluate_model(
        self,
        y_true: Union[np.ndarray, List[float]],
        y_pred: Union[np.ndarray, List[float]],
        model_name: str = "Model",
    ) -> ModelMetrics:
        """
        Evaluate a single model.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            ModelMetrics object
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        metrics = ModelMetrics(
            model_name=model_name,
            mae=self.mae(y_true, y_pred),
            rmse=self.rmse(y_true, y_pred),
            mape=self.mape(y_true, y_pred),
            n_samples=len(y_true),
        )
        
        self._results[model_name] = metrics
        return metrics
    
    def compare_models(
        self,
        y_true: Union[np.ndarray, List[float]],
        predictions_dict: Dict[str, Union[np.ndarray, List[float]]],
        metric: str = 'RMSE',
    ) -> ComparisonResult:
        """
        Compare multiple models and return sorted results.
        
        Args:
            y_true: Actual test values
            predictions_dict: {model_name: predictions_array}
            metric: Metric to sort by ('RMSE', 'MAE', 'MAPE')
            
        Returns:
            ComparisonResult with sorted DataFrame and best model info
        """
        y_true = np.asarray(y_true, dtype=float)
        
        rows = []
        for name, y_pred in predictions_dict.items():
            y_pred = np.asarray(y_pred, dtype=float)
            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Length mismatch for {name}: "
                    f"y_true={len(y_true)}, y_pred={len(y_pred)}"
                )
            metrics = self.evaluate_model(y_true, y_pred, name)
            rows.append(metrics)
        
        if not rows:
            raise ValueError("No models to compare")
        
        # Build DataFrame
        df = pd.DataFrame([m.to_dict() for m in rows])
        
        # Sort by metric
        valid_metrics = {'RMSE', 'MAE', 'MAPE (%)'}
        sort_metric = metric if metric in valid_metrics else 'RMSE'
        df = df.sort_values(sort_metric).reset_index(drop=True)
        
        # Best model
        best_idx = df[sort_metric].idxmin()
        best_row = df.loc[best_idx]
        best_model = best_row['Model']
        best_metrics = self._results[best_model]
        
        # Improvement over worst
        worst_val = df[sort_metric].max()
        best_val = df[sort_metric].min()
        improvement = ((worst_val - best_val) / worst_val * 100) if worst_val > 0 else 0.0
        
        result = ComparisonResult(
            metrics_table=df,
            best_model=best_model,
            best_metrics=best_metrics,
            improvement_pct=round(improvement, 1),
            metric_used=sort_metric,
        )
        
        self._last_comparison = result
        return result
    
    def get_best_model(
        self,
        metric: str = 'RMSE',
    ) -> Dict[str, Any]:
        """
        Get best model info from last comparison.
        
        Args:
            metric: Metric to use
            
        Returns:
            Dict with best_model, metrics, improvement_pct
        """
        if self._last_comparison is None:
            raise ValueError("No comparison performed yet. Call compare_models() first.")
        
        return {
            'best_model': self._last_comparison.best_model,
            'metrics': self._last_comparison.best_metrics.to_dict(),
            'improvement_pct': self._last_comparison.improvement_pct,
        }
    
    def generate_summary(
        self,
        y_true: Union[np.ndarray, List[float]],
        predictions_dict: Dict[str, Union[np.ndarray, List[float]]],
        metric: str = 'RMSE',
    ) -> Dict[str, Any]:
        """
        Full pipeline: compare + select best model.
        
        Args:
            y_true: Actual values
            predictions_dict: {model_name: predictions}
            metric: Metric for comparison
            
        Returns:
            Dict with comparison_table and best_model_info
        """
        comparison = self.compare_models(y_true, predictions_dict, metric)
        return {
            'comparison_table': comparison.metrics_table,
            'best_model_info': self.get_best_model(metric),
        }
    
    def export_results(
        self,
        filepath: str,
        results_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Export comparison results to CSV.
        
        Args:
            filepath: Output file path
            results_df: DataFrame to export (uses last comparison if None)
        """
        if results_df is None:
            if self._last_comparison is None:
                raise ValueError("No results to export")
            results_df = self._last_comparison.metrics_table
        results_df.to_csv(filepath, index=False)
    
    def get_results(self) -> Dict[str, ModelMetrics]:
        """Get all evaluated model results."""
        return self._results.copy()
    
    def get_last_comparison(self) -> Optional[ComparisonResult]:
        """Get last comparison result."""
        return self._last_comparison
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()
        self._last_comparison = None