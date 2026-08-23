"""
Model Registry for AI Demand Forecasting.

Provides model lifecycle management: registration, retrieval, metadata.
Pure Python service with no Streamlit dependencies.
"""
import builtins
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    model_type: str  # 'lstm', 'hybrid', 'arima', 'xgboost', etc.
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    training_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    feature_config: dict[str, Any] = field(default_factory=dict)
    file_path: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Registry for managing trained models.
    
    Provides registration, retrieval, and metadata management.
    Supports in-memory storage with optional file-based persistence.
    No Streamlit dependencies.
    """

    def __init__(self, storage_dir: str | Path | None = None):
        """
        Initialize the model registry.
        
        Args:
            storage_dir: Optional directory for persisting metadata
        """
        self._models: dict[str, ModelMetadata] = {}
        self._model_objects: dict[str, Any] = {}

        if storage_dir is not None:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_metadata()
        else:
            self.storage_dir = None

    # ------------------------------------------------------------------
    # Core Registry Operations
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model_type: str,
        model_object: Any = None,
        version: str = "1.0.0",
        training_config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        feature_config: dict[str, Any] | None = None,
        file_path: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> ModelMetadata:
        """
        Register a model in the registry.
        
        Args:
            name: Unique model name
            model_type: Type of model ('lstm', 'hybrid', 'arima', etc.)
            model_object: The actual model object (stored in memory)
            version: Model version
            training_config: Training configuration dict
            metrics: Evaluation metrics dict
            feature_config: Feature engineering config
            file_path: Path to saved model file
            description: Human-readable description
            tags: List of tags
            overwrite: Whether to overwrite existing model
            
        Returns:
            ModelMetadata for the registered model
            
        Raises:
            ValueError: If name exists and overwrite=False
        """
        if name in self._models and not overwrite:
            raise ValueError(f"Model '{name}' already exists. Use overwrite=True to replace.")

        metadata = ModelMetadata(
            name=name,
            model_type=model_type,
            version=version,
            training_config=training_config or {},
            metrics=metrics or {},
            feature_config=feature_config or {},
            file_path=file_path,
            description=description,
            tags=tags or [],
        )

        self._models[name] = metadata
        if model_object is not None:
            self._model_objects[name] = model_object

        if self.storage_dir:
            self._save_metadata()

        return metadata

    def get(self, name: str) -> ModelMetadata | None:
        """
        Get model metadata by name.
        
        Args:
            name: Model name
            
        Returns:
            ModelMetadata or None if not found
        """
        return self._models.get(name)

    def get_object(self, name: str) -> Any | None:
        """
        Get model object by name (in-memory).
        
        Args:
            name: Model name
            
        Returns:
            Model object or None
        """
        return self._model_objects.get(name)

    def exists(self, name: str) -> bool:
        """Check if model exists in registry."""
        return name in self._models

    def remove(self, name: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            name: Model name
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._models:
            return False

        del self._models[name]
        self._model_objects.pop(name, None)

        if self.storage_dir:
            self._save_metadata()

        return True

    def list(self, model_type: str | None = None) -> list[ModelMetadata]:
        """
        List all registered models.
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            List of ModelMetadata
        """
        models = list(self._models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def list_names(self, model_type: str | None = None) -> builtins.list[str]:
        """List model names."""
        return [m.name for m in self.list(model_type)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        """Save metadata to storage directory."""
        if not self.storage_dir:
            return

        metadata_file = self.storage_dir / "registry_metadata.json"
        data = {
            name: meta.to_dict()
            for name, meta in self._models.items()
        }

        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_metadata(self) -> None:
        """Load metadata from storage directory."""
        if not self.storage_dir:
            return

        metadata_file = self.storage_dir / "registry_metadata.json"
        if not metadata_file.exists():
            return

        with open(metadata_file) as f:
            data = json.load(f)

        for name, meta_dict in data.items():
            self._models[name] = ModelMetadata.from_dict(meta_dict)

    def save_model_object(
        self,
        name: str,
        filepath: str,
        serializer: str = 'joblib',
    ) -> bool:
        """
        Save model object to disk.
        
        Args:
            name: Model name
            filepath: Destination file path
            serializer: 'joblib' or 'keras'
            
        Returns:
            True if saved successfully
        """
        if name not in self._model_objects:
            return False

        obj = self._model_objects[name]
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        try:
            if serializer == 'joblib':
                joblib.dump(obj, filepath)
            elif serializer == 'keras':
                if hasattr(obj, 'save_model'):
                    obj.save_model(filepath)
                else:
                    # Assume it's a Keras model directly
                    obj.save(filepath)
            else:
                raise ValueError(f"Unknown serializer: {serializer}")

            # Update metadata
            if name in self._models:
                self._models[name].file_path = filepath
                if self.storage_dir:
                    self._save_metadata()

            return True
        except Exception as e:
            print(f"Failed to save model: {e}")
            return False

    def load_model_object(
        self,
        name: str,
        filepath: str,
        serializer: str = 'joblib',
        **kwargs,
    ) -> bool:
        """
        Load model object from disk.
        
        Args:
            name: Model name
            filepath: Source file path
            serializer: 'joblib' or 'keras'
            **kwargs: Additional args for loader (e.g., seq_length for LSTM)
            
        Returns:
            True if loaded successfully
        """
        try:
            if serializer == 'joblib':
                obj = joblib.load(filepath)
            elif serializer == 'keras':
                from tensorflow.keras.models import load_model
                # Handle LSTMForecaster wrapper
                if name in self._models:
                    meta = self._models[name]
                    if meta.model_type == 'lstm':
                        from src.models.lstm_model import LSTMForecaster
                        seq_length = kwargs.get('seq_length', 30)
                        forecaster = LSTMForecaster(seq_length=seq_length)
                        forecaster.load_model(filepath)
                        obj = forecaster
                    else:
                        obj = load_model(filepath)
                else:
                    obj = load_model(filepath)
            else:
                raise ValueError(f"Unknown serializer: {serializer}")

            self._model_objects[name] = obj
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def update_metrics(self, name: str, metrics: dict[str, float]) -> bool:
        """Update model metrics."""
        if name not in self._models:
            return False
        self._models[name].metrics.update(metrics)
        if self.storage_dir:
            self._save_metadata()
        return True

    def update_config(self, name: str, config: dict[str, Any]) -> bool:
        """Update training configuration."""
        if name not in self._models:
            return False
        self._models[name].training_config.update(config)
        if self.storage_dir:
            self._save_metadata()
        return True

    def add_tags(self, name: str, tags: builtins.list[str]) -> bool:
        """Add tags to model."""
        if name not in self._models:
            return False
        self._models[name].tags.extend(tags)
        if self.storage_dir:
            self._save_metadata()
        return True

    def get_by_tag(self, tag: str) -> builtins.list[ModelMetadata]:
        """Get models with a specific tag."""
        return [m for m in self._models.values() if tag in m.tags]

    def get_latest(self, model_type: str | None = None) -> ModelMetadata | None:
        """Get most recently registered model."""
        models = self.list(model_type)
        return models[0] if models else None

    def clear(self) -> None:
        """Clear all models from registry."""
        self._models.clear()
        self._model_objects.clear()
        if self.storage_dir:
            self._save_metadata()

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models

    def __iter__(self):
        return iter(self._models.values())


# Convenience function for creating a default registry
def create_registry(storage_dir: str | None = None) -> ModelRegistry:
    """Create a ModelRegistry with default settings."""
    return ModelRegistry(storage_dir)
