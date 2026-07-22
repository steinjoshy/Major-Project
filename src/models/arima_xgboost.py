import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


class HybridArimaXGBoost:
    """Hybrid model combining ARIMA for linear patterns and XGBoost for residual correction."""
    
    def __init__(self, arima_order=(1, 1, 1), xgb_params=None):
        """
        Initialize Hybrid ARIMA + XGBoost model.
        
        Args:
            arima_order: Tuple (p, d, q) for ARIMA
            xgb_params: Dictionary of XGBoost parameters
        """
        self.arima_order = arima_order
        self.arima_model = None
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=0,
            **(xgb_params or {})
        )
        self.scaler = StandardScaler()
        self.arima_fitted = False
        self.xgb_trained = False
    
    def fit_arima(self, data):
        """
        Fit ARIMA model on time-series data.
        
        Args:
            data: 1D array of historical demand values
        """
        try:
            self.arima_model = ARIMA(data, order=self.arima_order)
            self.arima_model = self.arima_model.fit()
            self.arima_fitted = True
            print(f"ARIMA{self.arima_order} fitted successfully")
        except Exception as e:
            print(f"ARIMA fitting error: {e}. Using differencing approach.")
            self.arima_order = (0, 1, 0)
            self.arima_model = ARIMA(data, order=self.arima_order).fit()
            self.arima_fitted = True
    
    def create_features_for_xgb(self, data, lags=30):
        """
        Create lag features for XGBoost from time-series data.
        
        Args:
            data: 1D array of values
            lags: Number of lag features to create
        
        Returns:
            Feature matrix, corresponding targets
        """
        X, y = [], []
        for i in range(lags, len(data)):
            X.append(data[i-lags:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit_xgb_on_residuals(self, data):
        """
        Train XGBoost on ARIMA residuals.
        
        Args:
            data: Historical demand data
        """
        if not self.arima_fitted:
            self.fit_arima(data)
        
        # Get ARIMA residuals
        arima_residuals = self.arima_model.resid.values
        
        # Create features from residuals
        X_resid, y_resid = self.create_features_for_xgb(arima_residuals, lags=10)
        
        if len(X_resid) > 0:
            # Scale features
            X_resid_scaled = self.scaler.fit_transform(X_resid)
            
            # Train XGBoost on residuals
            self.xgb_model.fit(X_resid_scaled, y_resid)
            self.xgb_trained = True
            print("XGBoost trained on ARIMA residuals successfully")
        else:
            print("Insufficient data for XGBoost training")
    
    def predict(self, steps=1):
        """
        Generate predictions using hybrid model.
        
        Args:
            steps: Number of steps to forecast
        
        Returns:
            Predicted values
        """
        if not self.arima_fitted:
            raise ValueError("ARIMA model not fitted. Call fit_arima() first.")
        
        # Get ARIMA predictions
        arima_forecast = self.arima_model.get_forecast(steps=steps).predicted_mean.values
        
        # Get XGBoost correction (if trained)
        xgb_correction = np.zeros(steps)
        if self.xgb_trained:
            try:
                # Use recent residuals as features for correction
                recent_residuals = self.arima_model.resid.values[-10:]
                
                if len(recent_residuals) == 10:
                    recent_residuals_scaled = self.scaler.transform(
                        recent_residuals.reshape(1, -1)
                    )
                    xgb_correction[0] = self.xgb_model.predict(recent_residuals_scaled)[0]
            except:
                pass
        
        # Combine ARIMA forecast with XGBoost correction
        hybrid_forecast = arima_forecast + xgb_correction
        return hybrid_forecast
    
    def fit_predict(self, data, steps=1):
        """Fit model and generate predictions."""
        self.fit_arima(data)
        self.fit_xgb_on_residuals(data)
        return self.predict(steps)
    
    def get_arima_params(self):
        """Get fitted ARIMA parameters."""
        if self.arima_model is None:
            return None
        return {
            'order': self.arima_order,
            'aic': self.arima_model.aic if hasattr(self.arima_model, 'aic') else None,
            'bic': self.arima_model.bic if hasattr(self.arima_model, 'bic') else None
        }


if __name__ == "__main__":
    print("Hybrid ARIMA + XGBoost module loaded successfully")
