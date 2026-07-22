import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelComparison:
    """Evaluate and compare forecasting models using standard metrics."""
    
    def __init__(self):
        self.results = {}
    
    @staticmethod
    def rmse(y_true, y_pred):
        """
        Calculate Root Mean Squared Error.
        
        Penalizes larger errors more heavily.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        Average absolute difference between actual and predicted.
        """
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error.
        
        Percentage error relative to actual values.
        Avoids division by zero by adding small epsilon.
        """
        epsilon = 1e-10
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate single model with all metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'Model': model_name,
            'RMSE': self.rmse(y_true, y_pred),
            'MAE': self.mae(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred)
        }
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self, y_true, predictions_dict):
        """
        Compare multiple models.
        
        Args:
            y_true: Actual values
            predictions_dict: Dictionary with model_name -> predictions mapping
        
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, y_pred in predictions_dict.items():
            metrics = self.evaluate_model(y_true, y_pred, model_name)
            comparison_results.append(metrics)
        
        results_df = pd.DataFrame(comparison_results)
        results_df = results_df.sort_values('RMSE').reset_index(drop=True)
        
        return results_df
    
    def get_best_model(self, results_df, metric='RMSE'):
        """
        Identify best performing model.
        
        Args:
            results_df: DataFrame with comparison results
            metric: Metric to use for comparison ('RMSE', 'MAE', 'MAPE')
        
        Returns:
            Best model name and metrics
        """
        if metric not in ['RMSE', 'MAE', 'MAPE']:
            raise ValueError(f"Unknown metric: {metric}")
        
        best_idx = results_df[metric].idxmin()
        best_row = results_df.iloc[best_idx]
        
        return {
            'best_model': best_row['Model'],
            'metrics': best_row.to_dict(),
            'rank': best_idx + 1
        }
    
    def generate_comparison_summary(self, y_true, predictions_dict, 
                                   lstm_pred, hybrid_pred):
        """
        Generate comprehensive comparison summary.
        
        Args:
            y_true: Actual test values
            predictions_dict: Dict of model predictions
            lstm_pred: LSTM predictions
            hybrid_pred: Hybrid model predictions
        
        Returns:
            Dictionary with detailed comparison
        """
        # Compare models
        comparison_df = self.compare_models(y_true, predictions_dict)
        
        # Get best model
        best_lstm = self.get_best_model(comparison_df, 'RMSE')
        
        # Generate visual comparison
        summary = {
            'comparison_table': comparison_df,
            'best_model_info': best_lstm,
            'all_metrics': self.results,
            'improvement_over_baseline': self._calculate_improvement(comparison_df)
        }
        
        return summary
    
    @staticmethod
    def _calculate_improvement(comparison_df):
        """Calculate percentage improvement of best model over worst."""
        rmse_values = comparison_df['RMSE'].values
        best_rmse = rmse_values.min()
        worst_rmse = rmse_values.max()
        
        if worst_rmse > 0:
            improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
        else:
            improvement = 0
        
        return round(improvement, 2)
    
    def export_results(self, filepath, results_df):
        """Export comparison results to CSV."""
        results_df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
    
    def get_metrics_dataframe(self):
        """Get all evaluation results as DataFrame."""
        return pd.DataFrame(self.results).T


if __name__ == "__main__":
    print("Model Comparison module loaded successfully")
