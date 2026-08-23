import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class LSTMForecaster:
    """
    LSTM-based time-series forecasting model using TensorFlow/Keras.

    Architecture:
        LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)

    Data must be MinMax-scaled before passing to train()/predict().
    Use the same scaler from DataPreprocessor to inverse-transform predictions.
    """

    def __init__(self, seq_length=30, lstm_units=64, epochs=50, batch_size=32):
        """
        Args:
            seq_length:  Length of each input sequence (look-back window)
            lstm_units:  Number of units in the first LSTM layer
            epochs:      Maximum training epochs (EarlyStopping may stop sooner)
            batch_size:  Mini-batch size during training
        """
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None

    # ------------------------------------------------------------------
    # Model Architecture
    # ------------------------------------------------------------------

    def build_model(self, input_shape):
        """Build and compile the LSTM neural network."""
        model = Sequential([
            LSTM(self.lstm_units, activation='tanh', input_shape=input_shape,
                 return_sequences=True),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.model = model
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        """ Train the LSTM model Automatically ensures input data has the required shape: (samples, sequence_length, features)"""

        # Convert to NumPy arrays
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        # --------------------------------------------------
        # FIX: Ensure LSTM input is 3-dimensional
        # --------------------------------------------------
        if X_train.ndim == 2:
           X_train = X_train.reshape(
               X_train.shape[0],
               X_train.shape[1],
               1
            )

        if X_train.ndim != 3:
           raise ValueError(
               f"LSTM training data must be 3D "
               f"(samples, sequence_length, features). "
               f"Received: {X_train.shape}"
            )

        # Validation data
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)

            if X_val.ndim == 2:
                X_val = X_val.reshape(
                    X_val.shape[0],
                    X_val.shape[1],
                    1
                )

        if X_val.ndim != 3:
            raise ValueError(
                f"LSTM validation data must be 3D. "
                f"Received: {X_val.shape}"
            )

    if y_val is not None:
        y_val = np.asarray(y_val, dtype=np.float32)

    # --------------------------------------------------
    # Build model using corrected shape
    # --------------------------------------------------
    if self.model is None:
        self.build_model(
            (X_train.shape[1], X_train.shape[2])
        )

    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=8,
            restore_best_weights=True
        )
    ]

    fit_kwargs = dict(
        epochs=self.epochs,
        batch_size=self.batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    if X_val is not None and y_val is not None:
        fit_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "callbacks": [
                EarlyStopping(
                    monitor="loss",
                    patience=8,
                    restore_best_weights=True
                )
            ],
            "verbose": verbose
        }

    self.history = self.model.fit(
        X_train,
        y_train,
        **fit_kwargs
    )

    return self.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X_test):
        """Generate predictions on scaled test sequences. Returns scaled output."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X_test, verbose=0)

    # ------------------------------------------------------------------
    # Future Forecast
    # ------------------------------------------------------------------

    def forecast_future(self, last_scaled_sequence, steps=30, scaler=None):
        """
        Auto-regressively forecast `steps` future values.

        Args:
            last_scaled_sequence: 1-D array of length seq_length, already MinMax-scaled.
                                  Pass `scaler.transform(last_raw_values.reshape(-1,1)).flatten()`
                                  from the caller — do NOT pass raw (unscaled) values.
            steps:  Number of future periods to forecast.
            scaler: MinMaxScaler used during preprocessing. If provided, predictions
                    are inverse-transformed back to original scale.

        Returns:
            np.array of length `steps` (original scale if scaler provided, else scaled)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if len(last_scaled_sequence) != self.seq_length:
            raise ValueError(
                f"last_scaled_sequence must have length {self.seq_length}, "
                f"got {len(last_scaled_sequence)}."
            )

        future_scaled = []
        current_seq = last_scaled_sequence.copy()

        for _ in range(steps):
            next_val = self.model.predict(
                current_seq.reshape(1, self.seq_length, 1), verbose=0
            )[0, 0]
            future_scaled.append(next_val)
            current_seq = np.append(current_seq[1:], next_val)

        future_scaled = np.array(future_scaled)

        if scaler is not None:
            return scaler.inverse_transform(
                future_scaled.reshape(-1, 1)
            ).flatten()

        return future_scaled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_training_history(self):
        """Return dict with 'loss' (and optionally 'val_loss') lists."""
        if self.history is None:
            return {}
        return self.history.history

    def save_model(self, filepath):
        """Save the trained Keras model to a file."""
        if self.model is not None:
            self.model.save(filepath)

    def load_model(self, filepath):
        """Load a previously saved Keras model."""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)


if __name__ == "__main__":
    print("LSTM Forecaster module loaded successfully")
