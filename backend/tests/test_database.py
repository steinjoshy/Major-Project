"""Tests for database models and operations (synchronous)."""
import pytest
from datetime import datetime
from uuid import uuid4
import tempfile
import os
import gc

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, joinedload
from sqlalchemy.pool import StaticPool

from backend.app.db.models import (
    Base,
    User,
    Dataset,
    ForecastingJob,
    Forecast,
    InventoryAnalysis,
    ModelRegistry,
    ModelMetrics,
    AuditLog,
    JobStatus,
    ModelType,
)


@pytest.fixture(scope="function")
def temp_db_path():
    """Create a unique temporary database file for each test."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_path = temp_file.name
    temp_file.close()  # Close the file handle, we'll let SQLAlchemy manage it
    yield temp_path
    # Cleanup - force garbage collection to release file handles
    import gc
    gc.collect()
    try:
        os.unlink(temp_db_path)
    except:
        pass


@pytest.fixture(scope="function")
def engine(temp_db_path):
    """Create a fresh SQLite engine for each test."""
    db_url = f"sqlite:///{temp_db_path}"
    engine = create_engine(
        db_url,
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup - properly dispose engine before deleting file
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    # Force garbage collection to release file handles
    import gc
    gc.collect()


@pytest.fixture(scope="function")
def db_session(engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password="hashed_password",
        full_name="Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_dataset(db_session, test_user):
    """Create a test dataset."""
    from datetime import datetime
    dataset = Dataset(
        name="Test Dataset",
        description="Test dataset for testing",
        filename="test.csv",
        file_path="/tmp/test.csv",
        file_size=1000,
        mime_type="text/csv",
        date_column="Date",
        sales_column="Sales",
        row_count=100,
        date_range_start=datetime(2023, 1, 1),
        date_range_end=datetime(2023, 12, 31),
        owner_id=test_user.id,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


@pytest.fixture
def test_job(db_session, test_dataset, test_user):
    """Create a test forecasting job."""
    job = ForecastingJob(
        name="Test Job",
        description="Test forecasting job",
        job_type="train_all",
        status=JobStatus.QUEUED,
        config={"seq_length": 30, "test_size": 0.2},
        seq_length=30,
        lstm_epochs=10,
        lstm_batch_size=16,
        arima_order=[1, 1, 1],
        forecast_steps=30,
        test_size=0.2,
        owner_id=test_user.id,
        dataset_id=test_dataset.id,
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job


class TestUserModel:
    """Tests for User model."""
    
    def test_create_user(self, db_session):
        """Test creating a user."""
        user = User(
            email="newuser@example.com",
            hashed_password="hashed_pwd",
            full_name="New User",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.full_name == "New User"
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_relationships(self, db_session, test_user, test_dataset):
        """Test user relationships."""
        # Refresh to load relationships
        db_session.refresh(test_user, ["datasets"])
        assert len(test_user.datasets) == 1
        assert test_user.datasets[0].id == test_dataset.id


class TestDatasetModel:
    """Tests for Dataset model."""
    
    def test_create_dataset(self, db_session, test_user):
        """Test creating a dataset."""
        from datetime import datetime
        dataset = Dataset(
            name="Test Dataset",
            description="Test dataset for testing",
            filename="test.csv",
            file_path="/tmp/test.csv",
            file_size=1000,
            mime_type="text/csv",
            date_column="Date",
            sales_column="Sales",
            row_count=100,
            date_range_start=datetime(2023, 1, 1),
            date_range_end=datetime(2023, 12, 31),
            owner_id=test_user.id,
        )
        db_session.add(dataset)
        db_session.commit()
        db_session.refresh(dataset)
        
        assert dataset.id is not None
        assert dataset.name == "Test Dataset"
        assert dataset.owner_id == test_user.id
        assert dataset.row_count == 100
        assert dataset.date_range_start is not None
    
    def test_dataset_relationships(self, db_session, test_dataset, test_user):
        """Test dataset relationships."""
        # Check owner relationship
        from sqlalchemy import select
        stmt = select(Dataset).where(Dataset.id == test_dataset.id).options(
            joinedload(Dataset.owner)
        )
        dataset = db_session.execute(select(Dataset).where(Dataset.id == test_dataset.id).options(
            joinedload(Dataset.owner)
        )).scalars().first()
        
        assert dataset is not None
        assert dataset.owner is not None
        assert dataset.owner.id == test_user.id


class TestForecastingJobModel:
    """Tests for ForecastingJob model."""
    
    def test_create_job(self, db_session, test_dataset, test_user):
        """Test creating a forecasting job."""
        job = ForecastingJob(
            name="Test Job",
            job_type="train",
            status=JobStatus.QUEUED,
            config={"seq_length": 30},
            seq_length=30,
            lstm_epochs=50,
            lstm_batch_size=32,
            arima_order=[1, 1, 1],
            forecast_steps=30,
            test_size=0.2,
            owner_id=test_user.id,
            dataset_id=test_dataset.id,
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        
        assert job.id is not None
        assert job.name == "Test Job"
        assert job.status == JobStatus.QUEUED
        assert job.seq_length == 30
        assert job.owner_id == test_user.id
        assert job.dataset_id is not None
    
    def test_job_relationships(self, db_session, test_job, test_dataset, test_user):
        """Test job relationships."""
        # Check owner relationship
        job = test_job
        assert job.owner_id == test_user.id
        assert job.dataset_id == test_dataset.id
    
    def test_job_status_transition(self, db_session, test_job):
        """Test job status transitions."""
        assert test_job.status == JobStatus.QUEUED
        test_job.status = JobStatus.RUNNING
        test_job.started_at = datetime.utcnow()
        db_session.add(test_job)
        db_session.commit()
        db_session.refresh(test_job)
        assert test_job.status == JobStatus.RUNNING
        assert test_job.started_at is not None


class TestForecastModel:
    """Tests for Forecast model."""
    
    def test_create_forecast(self, db_session, test_job):
        """Test creating a forecast."""
        forecast = Forecast(
            job_id=test_job.id,
            model_type="lstm",
            forecast_steps=30,
            forecast_data=[1.0, 2.0, 3.0],
            forecast_dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            forecast_metadata={"model_version": "1.0"},
        )
        db_session.add(forecast)
        db_session.commit()
        db_session.refresh(forecast)
        
        assert forecast.id is not None
        assert forecast.job_id == test_job.id
        assert forecast.model_type == "lstm"
        assert len(forecast.forecast_data) == 3


class TestInventoryAnalysisModel:
    """Tests for InventoryAnalysis model."""
    
    def test_create_inventory_analysis(self, db_session, test_job):
        """Test creating inventory analysis."""
        forecast_data = [10.0, 20.0, 15.0]
        analysis = InventoryAnalysis(
            job_id=test_job.id,
            service_level=0.95,
            lead_time=7,
            current_stock=1000.0,
            forecast_data=[100.0, 200.0, 150.0],
        )
        db_session.add(analysis)
        db_session.commit()
        db_session.refresh(analysis)
        
        assert analysis.id is not None
        assert analysis.job_id == test_job.id
        assert analysis.safety_stock > 0
        assert analysis.reorder_point > 0
        assert analysis.safety_stock_basis == 'forecast variability'


class TestModelRegistry:
    """Tests for ModelRegistry model."""
    
    def test_register_model(self, db_session):
        """Test registering a model."""
        model = ModelRegistry(
            name="test_lstm",
            model_type="lstm",
            version="1.0.0",
            description="Test LSTM model",
            training_config={"epochs": 50, "batch_size": 32},
            metrics={"rmse": 10.5, "mae": 8.2},
            feature_config={"seq_length": 30},
            tags=["experimental", "v1"],
        )
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)
        
        assert model.id is not None
        assert model.name == "test_lstm"
        assert model.model_type == "lstm"
        assert model.version == "1.0.0"
        assert "experimental" in model.tags
    
    def test_model_unique_name(self, db_session):
        """Test unique constraint on model name."""
        model1 = ModelRegistry(name="model1", model_type="lstm")
        db_session.add(model1)
        db_session.commit()
        
        model2 = ModelRegistry(name="model1", model_type="hybrid")
        db_session.add(model2)
        
        with pytest.raises(Exception):
            db_session.commit()


class TestModelMetrics:
    """Tests for ModelMetrics model."""
    
    def test_create_metrics(self, db_session):
        """Test creating model metrics."""
        model = ModelRegistry(name="test_model", model_type="lstm")
        db_session.add(model)
        db_session.commit()
        db_session.refresh(model)
        
        metrics = ModelMetrics(
            model_id=model.id,
            model_version="1.0",
            mae=5.0,
            rmse=7.5,
            mape=2.5,
            r2=0.95,
            n_samples=100,
            test_period_start=datetime(2023, 1, 1),
            test_period_end=datetime(2023, 12, 31),
            forecast_horizon=30,
        )
        db_session.add(metrics)
        db_session.commit()
        db_session.refresh(metrics)
        
        assert metrics.id is not None
        assert metrics.model_id == model.id
        assert metrics.mae == 5.0
        assert metrics.rmse == 7.5


class TestAuditLog:
    """Tests for AuditLog model."""
    
    def test_create_audit_log(self, db_session, test_user):
        """Test creating an audit log."""
        from uuid import uuid4
        log = AuditLog(
            user_id=test_user.id,
            action="create_dataset",
            resource_type="dataset",
            resource_id=uuid4(),
            old_values={"name": "old_name"},
            new_values={"name": "new_name"},
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0",
        )
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)
        
        assert log.id is not None
        assert log.action == "create_dataset"
        assert log.resource_type == "dataset"
        assert log.user_id == test_user.id
        assert log.old_values == {"name": "old_name"}
        assert log.new_values == {"name": "new_name"}


class TestDatabaseOperations:
    """Tests for database operations."""
    
    def test_cascade_delete(self, db_session, test_user, test_dataset):
        """Test cascade delete behavior."""
        # Create a job linked to the dataset
        job = ForecastingJob(
            name="Test Job",
            job_type="train",
            status=JobStatus.QUEUED,
            owner_id=test_user.id,
            dataset_id=test_dataset.id,
        )
        db_session.add(job)
        db_session.commit()
        job_id = job.id
        
        # Delete the dataset - should cascade delete the job
        db_session.delete(test_dataset)
        db_session.commit()
        
        # Check if job was deleted
        from sqlalchemy import select
        result = db_session.execute(select(ForecastingJob).where(ForecastingJob.id == job.id))
        job_result = result.scalar_one_or_none()
        assert job_result is None  # Job should be deleted
    
    def test_unique_constraints(self, db_session, test_user):
        """Test unique constraints."""
        # Create first user
        user1 = User(email="unique@test.com", hashed_password="pwd1")
        db_session.add(user1)
        db_session.commit()
        
        # Try to create another user with same email
        user2 = User(email="unique@test.com", hashed_password="pwd2")
        db_session.add(user2)
        
        with pytest.raises(Exception):
            db_session.commit()


class TestDatabaseSession:
    """Tests for database session management."""
    
    def test_session_rollback(self, db_session):
        """Test session rollback on error."""
        user = User(email="test@test.com", hashed_password="pwd")
        db_session.add(user)
        db_session.commit()
        
        # Try to add duplicate - should fail
        user2 = User(email="test@test.com", hashed_password="pwd2")
        db_session.add(user2)
        
        with pytest.raises(Exception):
            db_session.commit()
        
        # Verify rollback worked - first user should still exist
        from sqlalchemy import select
        result = db_session.execute(select(User).where(User.email == "test@test.com"))
        users = result.scalars().all()
        assert len(users) == 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])