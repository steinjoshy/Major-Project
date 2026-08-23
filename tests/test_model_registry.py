"""Tests for ModelRegistry."""
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.services.model_registry import (
    ModelMetadata,
    ModelRegistry,
    create_registry,
)


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return ModelRegistry()

    @pytest.fixture
    def registry_with_storage(self, tmp_path):
        """Create a registry with file storage."""
        return ModelRegistry(tmp_path / "registry")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_register_model(self, registry):
        """Test registering a model."""
        metadata = registry.register(
            name="test_model",
            model_type="lstm",
            model_object="dummy_object",
            version="1.0.0",
            training_config={"epochs": 50, "batch_size": 32},
            metrics={"RMSE": 10.5, "MAE": 8.2},
            feature_config={"seq_length": 30},
            description="Test LSTM model",
            tags=["experimental", "v1"],
        )

        assert isinstance(metadata, ModelMetadata)
        assert metadata.name == "test_model"
        assert metadata.model_type == "lstm"
        assert metadata.version == "1.0.0"
        assert metadata.training_config["epochs"] == 50
        assert metadata.metrics["RMSE"] == 10.5
        assert metadata.feature_config["seq_length"] == 30
        assert metadata.description == "Test LSTM model"
        assert "experimental" in metadata.tags
        assert registry.exists("test_model")

    def test_register_overwrite(self, registry):
        """Test overwriting an existing model."""
        registry.register("model1", "lstm", "obj1")
        registry.register("model1", "hybrid", "obj2", overwrite=True)

        meta = registry.get("model1")
        assert meta.model_type == "hybrid"
        assert registry.get_object("model1") == "obj2"

    def test_register_no_overwrite_raises(self, registry):
        """Test registering duplicate without overwrite raises error."""
        registry.register("model1", "lstm", "obj1")

        with pytest.raises(ValueError, match="already exists"):
            registry.register("model1", "hybrid", "obj2")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def test_get_metadata(self, registry):
        """Test getting model metadata."""
        registry.register("model1", "lstm", "obj1", metrics={"RMSE": 10.0})

        meta = registry.get("model1")

        assert meta is not None
        assert meta.name == "model1"
        assert meta.metrics["RMSE"] == 10.0

    def test_get_nonexistent(self, registry):
        """Test getting nonexistent model returns None."""
        assert registry.get("nonexistent") is None

    def test_get_object(self, registry):
        """Test getting model object."""
        obj = {"layers": [64, 32], "weights": "dummy"}
        registry.register("model1", "lstm", obj)

        retrieved = registry.get_object("model1")
        assert retrieved is obj

    def test_get_nonexistent_object(self, registry):
        """Test getting nonexistent object returns None."""
        assert registry.get_object("nonexistent") is None

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    def test_exists(self, registry):
        """Test checking model existence."""
        assert not registry.exists("model1")
        registry.register("model1", "lstm")
        assert registry.exists("model1")

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def test_remove(self, registry):
        """Test removing a model."""
        registry.register("model1", "lstm", "obj1")
        assert registry.exists("model1")

        result = registry.remove("model1")
        assert result is True
        assert not registry.exists("model1")
        assert registry.get_object("model1") is None

    def test_remove_nonexistent(self, registry):
        """Test removing nonexistent model returns False."""
        result = registry.remove("nonexistent")
        assert result is False

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def test_list_all(self, registry):
        """Test listing all models."""
        registry.register("model1", "lstm")
        registry.register("model2", "hybrid")
        registry.register("model3", "arima")

        models = registry.list()

        assert len(models) == 3
        names = [m.name for m in models]
        assert set(names) == {"model1", "model2", "model3"}

    def test_list_by_type(self, registry):
        """Test listing models by type."""
        registry.register("lstm1", "lstm")
        registry.register("lstm2", "lstm")
        registry.register("hybrid1", "hybrid")

        lstm_models = registry.list("lstm")
        hybrid_models = registry.list("hybrid")

        assert len(lstm_models) == 2
        assert len(hybrid_models) == 1

    def test_list_names(self, registry):
        """Test listing model names."""
        registry.register("model1", "lstm")
        registry.register("model2", "hybrid")

        names = registry.list_names()
        assert set(names) == {"model1", "model2"}

        lstm_names = registry.list_names("lstm")
        assert lstm_names == ["model1"]

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def test_register_with_tags(self, registry):
        """Test registering with tags."""
        registry.register("model1", "lstm", tags=["experimental", "v1"])

        meta = registry.get("model1")
        assert "experimental" in meta.tags
        assert "v1" in meta.tags

    def test_add_tags(self, registry):
        """Test adding tags to existing model."""
        registry.register("model1", "lstm")
        registry.add_tags("model1", ["production", "v2"])

        meta = registry.get("model1")
        assert "production" in meta.tags
        assert "v2" in meta.tags

    def test_get_by_tag(self, registry):
        """Test getting models by tag."""
        registry.register("model1", "lstm", tags=["experimental"])
        registry.register("model2", "hybrid", tags=["experimental"])
        registry.register("model3", "arima", tags=["production"])

        experimental = registry.get_by_tag("experimental")
        assert len(experimental) == 2
        names = [m.name for m in experimental]
        assert set(names) == {"model1", "model2"}

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def test_update_metrics(self, registry):
        """Test updating model metrics."""
        registry.register("model1", "lstm", metrics={"RMSE": 10.0})

        registry.update_metrics("model1", {"MAE": 8.5, "MAPE": 5.2})

        meta = registry.get("model1")
        assert meta.metrics["RMSE"] == 10.0
        assert meta.metrics["MAE"] == 8.5
        assert meta.metrics["MAPE"] == 5.2

    def test_update_config(self, registry):
        """Test updating training config."""
        registry.register("model1", "lstm", training_config={"epochs": 50})

        registry.update_config("model1", {"batch_size": 64, "lr": 0.001})

        meta = registry.get("model1")
        assert meta.training_config["epochs"] == 50
        assert meta.training_config["batch_size"] == 64
        assert meta.training_config["lr"] == 0.001

    # ------------------------------------------------------------------
    # Special Queries
    # ------------------------------------------------------------------

    def test_get_latest(self, registry):
        """Test getting latest registered model."""
        import time
        registry.register("model1", "lstm")
        time.sleep(0.01)
        registry.register("model2", "hybrid")
        time.sleep(0.01)
        registry.register("model3", "arima")

        latest = registry.get_latest()
        assert latest.name == "model3"

        latest_lstm = registry.get_latest("lstm")
        assert latest_lstm.name == "model1"

    def test_get_latest_none(self, registry):
        """Test getting latest from empty registry."""
        assert registry.get_latest() is None
        assert registry.get_latest("lstm") is None

    # ------------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------------

    def test_clear(self, registry):
        """Test clearing all models."""
        registry.register("model1", "lstm")
        registry.register("model2", "hybrid")

        assert len(registry) == 2

        registry.clear()

        assert len(registry) == 0
        assert not registry.exists("model1")

    # ------------------------------------------------------------------
    # Persistence (File Storage)
    # ------------------------------------------------------------------

    def test_persistence_save_load(self, tmp_path):
        """Test saving and loading registry from disk."""
        storage_dir = tmp_path / "registry"

        # Create registry with storage
        registry1 = ModelRegistry(storage_dir)
        registry1.register("model1", "lstm", metrics={"RMSE": 10.0}, tags=["test"])
        registry1.register("model2", "hybrid", metrics={"RMSE": 8.5})

        # Create new registry with same storage
        registry2 = ModelRegistry(storage_dir)

        assert registry2.exists("model1")
        assert registry2.exists("model2")

        meta1 = registry2.get("model1")
        assert meta1.metrics["RMSE"] == 10.0
        assert "test" in meta1.tags

        meta2 = registry2.get("model2")
        assert meta2.metrics["RMSE"] == 8.5

    def test_persistence_metadata_file(self, tmp_path):
        """Test metadata file is created."""
        storage_dir = tmp_path / "registry"
        registry = ModelRegistry(storage_dir)
        registry.register("model1", "lstm")

        meta_file = storage_dir / "registry_metadata.json"
        assert meta_file.exists()

    # ------------------------------------------------------------------
    # Model Object Persistence
    # ------------------------------------------------------------------

    def test_save_model_object_joblib(self, registry, tmp_path):
        """Test saving model object with joblib."""
        import joblib

        obj = {"model": "dummy", "params": {"lr": 0.001}}
        registry.register("model1", "lstm", model_object=obj)

        filepath = tmp_path / "model1.joblib"
        result = registry.save_model_object("model1", str(filepath), serializer="joblib")

        assert result is True
        assert filepath.exists()

        # Verify it can be loaded
        loaded = joblib.load(filepath)
        assert loaded == {"model": "dummy", "params": {"lr": 0.001}}

    def test_load_model_object_joblib(self, registry, tmp_path):
        """Test loading model object with joblib."""
        import joblib

        obj = {"model": "dummy", "params": {"lr": 0.001}}
        filepath = tmp_path / "model1.joblib"
        joblib.dump(obj, filepath)

        result = registry.load_model_object("model1", str(filepath), serializer="joblib")

        assert result is True
        assert registry.get_object("model1") == obj

    def test_load_model_object_keras(self, registry, tmp_path):
        """Test loading Keras model object."""
        # This test would need a real Keras model - skip for now
        pytest.skip("Requires actual Keras model")

    # ------------------------------------------------------------------
    # Convenience Function
    # ------------------------------------------------------------------

    def test_create_registry(self):
        """Test create_registry convenience function."""
        registry = create_registry()

        assert isinstance(registry, ModelRegistry)
        assert registry.storage_dir is None

    def test_create_registry_with_storage(self, tmp_path):
        """Test create_registry with storage directory."""
        registry = create_registry(str(tmp_path / "registry"))

        assert isinstance(registry, ModelRegistry)
        assert registry.storage_dir is not None

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------

    def test_len(self, registry):
        """Test len(registry)."""
        assert len(registry) == 0

        registry.register("model1", "lstm")
        assert len(registry) == 1

        registry.register("model2", "hybrid")
        assert len(registry) == 2

    def test_contains(self, registry):
        """Test 'in' operator."""
        registry.register("model1", "lstm")

        assert "model1" in registry
        assert "model2" not in registry

    def test_iter(self, registry):
        """Test iteration over registry."""
        registry.register("model1", "lstm")
        registry.register("model2", "hybrid")

        models = list(registry)
        assert len(models) == 2
        assert all(isinstance(m, ModelMetadata) for m in models)

    # ------------------------------------------------------------------
    # ModelMetadata
    # ------------------------------------------------------------------

    def test_metadata_to_dict(self):
        """Test ModelMetadata serialization."""
        meta = ModelMetadata(
            name="test",
            model_type="lstm",
            version="2.0.0",
            training_config={"epochs": 100},
            metrics={"RMSE": 5.0},
            feature_config={"seq_length": 30},
            tags=["tag1", "tag2"],
        )

        d = meta.to_dict()

        assert d["name"] == "test"
        assert d["model_type"] == "lstm"
        assert d["version"] == "2.0.0"
        assert d["training_config"]["epochs"] == 100
        assert d["metrics"]["RMSE"] == 5.0
        assert d["feature_config"]["seq_length"] == 30
        assert d["tags"] == ["tag1", "tag2"]
        assert "created_at" in d

    def test_metadata_from_dict(self):
        """Test ModelMetadata deserialization."""
        data = {
            "name": "test",
            "model_type": "lstm",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00",
            "training_config": {"epochs": 50},
            "metrics": {"RMSE": 10.0},
            "feature_config": {},
            "file_path": None,
            "description": "",
            "tags": ["tag1"],
        }

        meta = ModelMetadata.from_dict(data)

        assert meta.name == "test"
        assert meta.model_type == "lstm"
        assert meta.version == "1.0.0"
        assert meta.training_config["epochs"] == 50
        assert meta.metrics["RMSE"] == 10.0
        assert meta.tags == ["tag1"]

    # ------------------------------------------------------------------
    # Edge Cases
    # ------------------------------------------------------------------

    def test_empty_registry(self, registry):
        """Test operations on empty registry."""
        assert len(registry) == 0
        assert registry.list() == []
        assert registry.get_latest() is None
        assert registry.get("nonexistent") is None

    def test_register_without_object(self, registry):
        """Test registering without model object."""
        registry.register("model1", "lstm")

        assert registry.exists("model1")
        assert registry.get_object("model1") is None

    def test_register_with_file_path(self, registry):
        """Test registering with file path."""
        registry.register("model1", "lstm", file_path="/path/to/model.keras")

        meta = registry.get("model1")
        assert meta.file_path == "/path/to/model.keras"

    def test_metadata_version(self, registry):
        """Test model version handling."""
        registry.register("model1", "lstm", version="1.0.0")
        registry.register("model1", "lstm", version="2.0.0", overwrite=True)

        meta = registry.get("model1")
        assert meta.version == "2.0.0"
