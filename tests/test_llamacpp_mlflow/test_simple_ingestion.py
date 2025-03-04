from dotenv import load_dotenv
import os
from pathlib import Path
import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import time

mlflow_docker_path = Path(__file__).parents[2] / "scripts/dev/mlflow"
load_dotenv(mlflow_docker_path / "config.env")
mlflow_port = os.getenv("MLFLOW_PORT")
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_ACCESS_KEY")

from mlflow.tracking import MlflowClient
import pytest

@pytest.fixture
def tracking_uri():
    mlflow_port = os.getenv("MLFLOW_PORT")
    tracking_uri = f"http://localhost:{mlflow_port}/"
    return tracking_uri

def test_connect_to_mlflow(tracking_uri: str):
     # Configure MLflow client
    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri
    )

    try:
        # Test connection by listing experiments
        experiments = client.search_experiments()
        print("Successfully connected to MLflow server")
        print("Available experiments:", experiments)
        return True
    except Exception as e:
        print(f"Failed to connect to MLflow server: {e}")
        return False

def test_create_and_log_experiment(tracking_uri: str):
    """Test creating a new experiment and logging metrics"""
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create a new experiment with unique name using timestamp
    experiment_name = f"test_experiment_{int(time.time())}"
    
    # Delete experiment if it exists
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            client.delete_experiment(experiment.experiment_id)
    except Exception:
        pass  # Experiment doesn't exist, which is fine
    
    # Create new experiment
    experiment_id = mlflow.create_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id):
        # Log some metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.1)
        
        # Log parameters
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)
        
        # Create and log a small model with more features
        X, y = make_classification(
            n_samples=100,
            n_features=10,  # Increased total features
            n_informative=5,  # 5 informative features
            n_redundant=3,    # 3 redundant features
            n_repeated=1,     # 1 repeated feature
            random_state=42
        )
        model = LogisticRegression(random_state=42).fit(X, y)
        mlflow.sklearn.log_model(model, "model")
        
        # Log an artifact
        np.save("test_array.npy", np.array([1, 2, 3]))
        mlflow.log_artifact("test_array.npy")
        os.remove("test_array.npy")  # Clean up
    
    # Verify experiment exists
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    assert any(exp.name == experiment_name for exp in experiments), "Experiment not found"
    
    # Verify run data
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    assert len(runs) > 0, "No runs found in experiment"
    
    # Clean up
    mlflow.delete_experiment(experiment_id)

def test_model_registration(tracking_uri: str):
    """Test registering a model in the model registry"""
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create a simple model with more features
    X, y = make_classification(
        n_samples=100,
        n_features=10,  # Increased total features
        n_informative=5,  # 5 informative features
        n_redundant=3,    # 3 redundant features
        n_repeated=1,     # 1 repeated feature
        random_state=42
    )
    model = LogisticRegression(random_state=42).fit(X, y)
    
    # Log and register the model
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mv = mlflow.register_model(model_uri, "test_model")
        
        # Add model version description
        client = MlflowClient(tracking_uri=tracking_uri)
        client.update_model_version(
            name="test_model",
            version=mv.version,
            description="Test model version"
        )
    
    # Verify model exists in registry
    client = MlflowClient(tracking_uri=tracking_uri)
    model_versions = client.search_model_versions(f"name='test_model'")
    assert len(model_versions) > 0, "Model not found in registry"
    
    # Clean up
    client.delete_model_version("test_model", mv.version)
    client.delete_registered_model("test_model")


