import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelComparison:
    """Evaluate and compare forecasting models using standard metrics."""

    def __init__(self):
        self.results = {}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error — penalises large errors."""
        return float(np.sqrt(mean_squared_error(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float)
        )))

    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error — average absolute deviation."""
        return float(mean_absolute_error(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float)
        ))

    @staticmethod
    def mape(y_true, y_pred):
        """
        Mean Absolute Percentage Error (%).
        Avoids division by zero using a small epsilon floor.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        eps = 1e-8
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate a single model with MAE, RMSE, MAPE.

        Returns:
            Dict with model name and metric values
        """
        metrics = {
            'Model': model_name,
            'MAE': self.mae(y_true, y_pred),
            'RMSE': self.rmse(y_true, y_pred),
            'MAPE (%)': self.mape(y_true, y_pred),
        }
        self.results[model_name] = metrics
        return metrics

    def compare_models(self, y_true, predictions_dict):
        """
        Compare multiple models and return a sorted DataFrame.

        Args:
            y_true: Actual test values (array-like)
            predictions_dict: {model_name: predictions_array}

        Returns:
            pd.DataFrame sorted by RMSE ascending (best model first)
        """
        rows = []
        for name, y_pred in predictions_dict.items():
            rows.append(self.evaluate_model(y_true, y_pred, name))

        df = pd.DataFrame(rows)
        df = df.sort_values('RMSE').reset_index(drop=True)
        # Round for display
        for col in ['MAE', 'RMSE', 'MAPE (%)']:
            df[col] = df[col].round(4)
        return df

    def get_best_model(self, results_df, metric='RMSE'):
        """
        Return info about the best-performing model.

        Args:
            results_df: DataFrame from compare_models()
            metric: 'RMSE', 'MAE', or 'MAPE (%)'

        Returns:
            Dict: {best_model, metrics, improvement_pct}
        """
        valid_metrics = {'RMSE', 'MAE', 'MAPE (%)'}
        if metric not in valid_metrics:
            metric = 'RMSE'

        best_idx = results_df[metric].idxmin()
        best_row = results_df.loc[best_idx]

        # Improvement over worst model
        worst_val = results_df[metric].max()
        best_val = results_df[metric].min()
        improvement = ((worst_val - best_val) / worst_val * 100) if worst_val > 0 else 0.0

        return {
            'best_model': best_row['Model'],
            'metrics': best_row.to_dict(),
            'improvement_pct': round(improvement, 1),
        }

    def generate_comparison_summary(self, y_true, predictions_dict):
        """Full pipeline: compare + select best model."""
        df = self.compare_models(y_true, predictions_dict)
        best = self.get_best_model(df, 'RMSE')
        return {
            'comparison_table': df,
            'best_model_info': best,
        }

    def export_results(self, filepath, results_df):
        """Export comparison results to CSV."""
        results_df.to_csv(filepath, index=False)


if __name__ == "__main__":
    print("Model Comparison module loaded successfully")
