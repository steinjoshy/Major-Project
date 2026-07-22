import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class LSTMForecaster:
    """LSTM-based time-series forecasting model."""
    
    def __init__(self, seq_length=30, lstm_units=64, epochs=50, batch_size=32):
        """
        Initialize LSTM forecaster.
        
        Args:
            seq_length: Length of input sequences
            lstm_units: Number of LSTM units
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.scaler = None
    
    def build_model(self, input_shape):
        """Build LSTM neural network architecture."""
        model = Sequential([
            LSTM(self.lstm_units, activation='relu', input_shape=input_shape, 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', 
                     metrics=['mae'])
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences (samples, seq_length, features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
        
        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        return self.history
    
    def predict(self, X_test):
        """Generate predictions on test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X_test, verbose=0)
    
    def forecast_future(self, data, steps=30, scaler=None):
        """
        Forecast future values beyond the training data.
        
        Args:
            data: Last sequence of data (scaled)
            steps: Number of steps to forecast
            scaler: MinMaxScaler for inverse transformation
        
        Returns:
            Future predictions (original scale if scaler provided)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        future_predictions = []
        current_sequence = data.copy()
        
        for _ in range(steps):
            # Predict next value
            next_value = self.model.predict(current_sequence.reshape(1, 
                                            self.seq_length, 1), verbose=0)[0, 0]
            future_predictions.append(next_value)
            
            # Update sequence by removing first element and adding new prediction
            current_sequence = np.append(current_sequence[1:], next_value)
        
        future_predictions = np.array(future_predictions)
        
        # Inverse scale if scaler provided
        if scaler is not None:
            future_predictions = scaler.inverse_transform(
                future_predictions.reshape(-1, 1)
            ).flatten()
        
        return future_predictions
    
    def get_training_loss(self):
        """Get training history."""
        if self.history is None:
            return None
        return self.history.history
    
    def save_model(self, filepath):
        """Save trained model to file."""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load trained model from file."""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)


if __name__ == "__main__":
    print("LSTM Forecaster module loaded successfully")
