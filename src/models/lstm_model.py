import logging
import time

import numpy as np
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)


class _ProgressLogger(Callback):
    """Keras callback that logs each epoch and forwards progress to a callback.

    Kept deliberately small so training can report progress to the UI without
    changing the model architecture or the training loop semantics.
    """

    def __init__(self, total_epochs, log_every=5, progress_callback=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.log_every = log_every
        self.progress_callback = progress_callback
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        done = epoch + 1
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if (self.log_every and done % self.log_every == 0) or done == self.total_epochs:
            logger.info(
                "LSTM epoch %d/%d loss=%s val_loss=%s (%.1fs)",
                done, self.total_epochs,
                f"{loss:.5f}" if loss is not None else "n/a",
                f"{val_loss:.5f}" if val_loss is not None else "n/a",
                time.perf_counter() - self.started,
            )
        if self.progress_callback is not None:
            try:
                self.progress_callback(done, self.total_epochs)
            except Exception:
                logger.debug("LSTM progress_callback failed", exc_info=True)


class LSTMForecaster:
    """
    LSTM-based time-series forecasting model using TensorFlow/Keras.

    Architecture:
        LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2)
        → Dense(16) → Dense(1)
    """

    def __init__(self, seq_length=30, lstm_units=64, epochs=50, batch_size=32):
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def build_model(self, input_shape):
        """Build and compile the LSTM neural network."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(
                self.lstm_units,
                activation="tanh",
                return_sequences=True,
            ),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, activation="tanh"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=0,
              progress_callback=None, log_every=5):
        """Train the LSTM model."""

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]

        if X_train.ndim != 3:
            raise ValueError(
                f"X_train must be 3D (samples, sequence_length, features). "
                f"Received {X_train.shape}"
            )

        if X_train.shape[1:] != (self.seq_length, 1):
            raise ValueError(
                f"Expected X_train shape (*, {self.seq_length}, 1), "
                f"received {X_train.shape}"
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train/y_train size mismatch: "
                f"{len(X_train)} vs {len(y_train)}"
            )

        validation_data = None

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

            if X_val.ndim == 2:
                X_val = X_val[:, :, np.newaxis]

            if X_val.ndim != 3:
                raise ValueError(
                    f"X_val must be 3D. Received {X_val.shape}"
                )

            if X_val.shape[1:] != (self.seq_length, 1):
                raise ValueError(
                    f"Expected X_val shape (*, {self.seq_length}, 1), "
                    f"received {X_val.shape}"
                )

            if len(X_val) != len(y_val):
                raise ValueError(
                    f"X_val/y_val size mismatch: "
                    f"{len(X_val)} vs {len(y_val)}"
                )

            validation_data = (X_val, y_val)

        if self.model is None:
            self.build_model((self.seq_length, 1))

        monitor = "val_loss" if validation_data is not None else "loss"

        callbacks = [
            _ProgressLogger(
                total_epochs=self.epochs,
                log_every=log_every,
                progress_callback=progress_callback,
            ),
            EarlyStopping(
                monitor=monitor,
                patience=8,
                restore_best_weights=True,
            )
        ]

        fit_kwargs = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "callbacks": callbacks,
            "verbose": verbose,
        }

        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data

        logger.info(
            "Training LSTM: samples=%d, seq_length=%d, max_epochs=%d, batch_size=%d",
            len(X_train), self.seq_length, self.epochs, self.batch_size,
        )
        started = time.perf_counter()
        self.history = self.model.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )
        logger.info(
            "LSTM training complete: %d epochs in %.1fs",
            len(self.history.history.get("loss", [])),
            time.perf_counter() - started,
        )

        return self.history

    def predict(self, X_test):
        """Generate predictions on scaled test sequences."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_test = np.asarray(X_test, dtype=np.float32)

        if X_test.ndim == 2:
            X_test = X_test[:, :, np.newaxis]

        if X_test.ndim != 3:
            raise ValueError(
                f"X_test must be 3D. Received {X_test.shape}"
            )

        if X_test.shape[1:] != (self.seq_length, 1):
            raise ValueError(
                f"Expected X_test shape (*, {self.seq_length}, 1), "
                f"received {X_test.shape}"
            )

        return self.model.predict(X_test, verbose=0)

    def forecast_future(self, last_scaled_sequence, steps=30, scaler=None):
        """
        Auto-regressively forecast future values.

        last_scaled_sequence must contain exactly seq_length values.
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        current_seq = np.asarray(
            last_scaled_sequence,
            dtype=np.float32,
        ).reshape(-1)

        if len(current_seq) != self.seq_length:
            raise ValueError(
                f"last_scaled_sequence must have length {self.seq_length}, "
                f"got {len(current_seq)}"
            )

        future_scaled = []

        for _ in range(steps):
            model_input = current_seq.reshape(
                1,
                self.seq_length,
                1,
            )

            next_value = self.model.predict(
                model_input,
                verbose=0,
            )[0, 0]

            future_scaled.append(next_value)
            current_seq = np.append(current_seq[1:], next_value)

        future_scaled = np.asarray(
            future_scaled,
            dtype=np.float32,
        )

        if scaler is not None:
            return scaler.inverse_transform(
                future_scaled.reshape(-1, 1)
            ).flatten()

        return future_scaled

    def forecast_future_multiple(self, last_scaled_sequences, steps=30, scalers=None):
        """
        Auto-regressively forecast future values for multiple sequences (panel data).

        Args:
            last_scaled_sequences: Array of shape (n_groups, seq_length) or list of sequences
            steps: Number of future steps to forecast per sequence
            scalers: List of scalers (one per group) or single scaler for all

        Returns:
            Array of shape (n_groups, steps) with forecasts in original scale
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Handle both array and list inputs
        if isinstance(last_scaled_sequences, list):
            sequences = [np.asarray(s, dtype=np.float32).reshape(-1) for s in last_scaled_sequences]
        else:
            sequences = [np.asarray(s, dtype=np.float32).reshape(-1) for s in last_scaled_sequences]

        n_groups = len(sequences)

        if scalers is None:
            scalers = [None] * n_groups
        elif not isinstance(scalers, list):
            scalers = [scalers] * n_groups

        all_forecasts = []

        for i, current_seq in enumerate(sequences):
            if len(current_seq) != self.seq_length:
                raise ValueError(
                    f"Sequence {i} must have length {self.seq_length}, "
                    f"got {len(current_seq)}"
                )

            future_scaled = []

            for _ in range(steps):
                model_input = current_seq.reshape(
                    1,
                    self.seq_length,
                    1,
                )

                next_value = self.model.predict(
                    model_input,
                    verbose=0,
                )[0, 0]

                future_scaled.append(next_value)
                current_seq = np.append(current_seq[1:], next_value)

            future_scaled = np.asarray(future_scaled, dtype=np.float32)

            if scalers[i] is not None:
                future_original = scalers[i].inverse_transform(
                    future_scaled.reshape(-1, 1)
                ).flatten()
            else:
                future_original = future_scaled

            all_forecasts.append(future_original)

        return np.array(all_forecasts)

    def predict_multiple(self, X_test):
        """
        Generate predictions on multiple test sequences (panel data).

        Args:
            X_test: Array of shape (n_samples, seq_length, 1) or list of arrays

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if isinstance(X_test, list):
            X_test = np.concatenate(X_test, axis=0)

        X_test = np.asarray(X_test, dtype=np.float32)

        if X_test.ndim == 2:
            X_test = X_test[:, :, np.newaxis]

        if X_test.ndim != 3:
            raise ValueError(
                f"X_test must be 3D. Received {X_test.shape}"
            )

        if X_test.shape[1:] != (self.seq_length, 1):
            raise ValueError(
                f"Expected X_test shape (*, {self.seq_length}, 1), "
                f"received {X_test.shape}"
            )

        return self.model.predict(X_test, verbose=0)

    def get_training_history(self):
        """Return training history."""
        if self.history is None:
            return {}

        return self.history.history

    def save_model(self, filepath):
        """Save the trained Keras model."""
        if self.model is not None:
            self.model.save(filepath)

    def load_model(self, filepath):
        """Load a previously saved Keras model."""
        from tensorflow.keras.models import load_model

        self.model = load_model(filepath)


if __name__ == "__main__":
    print("LSTM Forecaster module loaded successfully")