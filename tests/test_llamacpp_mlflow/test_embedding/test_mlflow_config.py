import os
from dotenv import load_dotenv
import pytest
from pathlib import Path
import json
import tempfile
import shutil
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient
from loguru import logger

mlflow_docker_path = Path(__file__).parents[3] / "scripts/dev/mlflow"
load_dotenv(mlflow_docker_path / "config.env")
mlflow_port = os.getenv("MLFLOW_PORT")
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_ACCESS_KEY")

@pytest.fixture
def tracking_uri():
    mlflow_port = os.getenv("MLFLOW_PORT")
    tracking_uri = f"http://localhost:{mlflow_port}/"
    print(tracking_uri)
    return tracking_uri

@pytest.fixture
def temp_config_dir():
    """Fixture to create and clean up the models directory."""
    models_path = Path(__file__).parent / "mlflow_model_config"
    models_path.mkdir(exist_ok=True)
    yield models_path
    # Clean up after tests
    # if models_path.exists():
    #     shutil.rmtree(models_path)

@pytest.fixture
def mlflow_client(tracking_uri):
    """Fixture to create MLflow client."""
    yield MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

def test_connect_to_mlflow(mlflow_client):
     # Configure MLflow client
    client = mlflow_client

    try:
        # Test connection by listing experiments
        experiments = client.search_experiments()
        print("Successfully connected to MLflow server")
        print("Available experiments:", experiments)
       
    except Exception as e:
        print(f"Failed to connect to MLflow server: {e}")

# @pytest.fixture
# def model_config_manager(mlflow_client, temp_config_dir):
#     """Fixture to create MLflowModelConfigManager instance."""
#     config_path = temp_config_dir / "model_config.json"
#     print(f"Config path: {config_path}")
#     print(f"MLflow client: {mlflow_client}")
#     experiments = mlflow_client.search_experiments()
#     print(f"Experiments: {experiments}")
#     yield MLflowModelConfigManager(mlflow_client=mlflow_client, config_path=config_path)

# def test_mlflow_config_from_env():
#     # Setup test environment variables
#     os.environ["MLFLOW_TRACKING_URI"] = "http://test-uri"
#     os.environ["MLFLOW_TRACKING_USERNAME"] = "test-user"
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = "test-pass"
    
#     config = MLFlowConfig.from_env()
    
#     assert config.tracking_uri == "http://test-uri"
#     assert config.registry_uri == "http://test-uri"  # Should default to tracking_uri
#     assert config.username == "test-user"
#     assert config.password == "test-pass"

# def test_mlflow_config_with_registry():
#     os.environ["MLFLOW_TRACKING_URI"] = "http://test-uri"
#     os.environ["MLFLOW_REGISTRY_URI"] = "http://registry-uri"
#     os.environ["MLFLOW_TRACKING_USERNAME"] = "test-user"
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = "test-pass"
    
#     config = MLFlowConfig.from_env()
    
#     assert config.tracking_uri == "http://test-uri"
#     assert config.registry_uri == "http://registry-uri"

# def test_mlflow_config_missing_env():
#     # Clear required environment variables
#     if "MLFLOW_TRACKING_URI" in os.environ:
#         del os.environ["MLFLOW_TRACKING_URI"]
    
#     with pytest.raises(AssertionError):
#         MLFlowConfig.from_env()

# @pytest.fixture
# def temp_config_dir():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         yield Path(tmpdir)

def test_connect_to_mlflow(mlflow_client):
     # Configure MLflow client
    client = mlflow_client

    try:
        # Test connection by listing experiments
        experiments = client.search_experiments()
        print("Successfully connected to MLflow server")
        print("Available experiments:", experiments)
        
    except Exception as e:
        print(f"Failed to connect to MLflow server: {e}")
        
    
# def test_model(model_config_manager):
#     client = model_config_manager.mlflow_client
#     experiments = client.search_experiments()
#     logger.info(f"Experiments: {experiments}")

# def test_add_model(model_config_manager):
#     """Test adding a model to the configuration."""
#     model_dir = model_config_manager.add_model("qa_model", alias="champion" , artifact_path="model_path")
    
#     assert model_dir.exists()
#     assert model_dir.name == "qa_model"
    
#     # # Verify config was updated
#     # config = model_config_manager.model_config.load_model_config("qa_model")
#     # assert config is not None
#     # assert config.model_name == "qa_model"
#     # assert config.alias == "champion"

# def test_check_for_update(model_config_manager):
#     """Test checking for model updates."""
#     # First add the model
#     model_config_manager.add_model("qa_model", alias="champion")
    
#     # Check for updates
#     needs_update = model_config_manager.check_for_update("qa_model")
#     assert isinstance(needs_update, bool)

# def test_check_for_updates(model_config_manager):
#     """Test checking updates for all models."""
#     # Add multiple models
#     model_config_manager.add_model("qa_model", alias="champion")
#     model_config_manager.add_model("rag_model", alias="champion")
    
#     updates = model_config_manager.check_for_updates()
#     assert isinstance(updates, dict)
#     assert "qa_model" in updates
#     assert "rag_model" in updates

# def test_get_model_status(model_config_manager):
#     """Test getting model status."""
#     # Add a model first
#     model_config_manager.add_model("qa_model", alias="champion")
    
#     # Get status for specific model
#     status = model_config_manager.get_model_status("qa_model")
#     assert isinstance(status, dict)
#     assert status["model_name"] == "qa_model"
#     assert "needs_update" in status
    
#     # Get status for all models
#     all_status = model_config_manager.get_model_status()
#     assert isinstance(all_status, dict)
#     assert "qa_model" in all_status

# def test_delete_model(model_config_manager):
#     """Test deleting a model."""
#     # Add a model first
#     model_config_manager.add_model("test_model", alias="champion")
    
#     # Delete the model
#     model_config_manager.delete_model("test_model")
    
#     # Verify model was deleted
#     config = model_config_manager.model_config.load_config()
#     assert "test_model" not in config

# def test_update_model_alias(model_config_manager):
#     """Test updating model alias."""
#     # Add a model first
#     model_config_manager.add_model("qa_model", alias="champion")
    
#     # Update alias
#     new_alias = "production"
#     model_config_manager.update_model_alias("qa_model", new_alias)
    
#     # Verify alias was updated
#     config = model_config_manager.model_config.load_model_config("qa_model")
#     assert config.alias == new_alias

# def test_list_models(model_config_manager):
#     """Test listing all models."""
#     # Add multiple models
#     model_config_manager.add_model("qa_model", alias="champion")
#     model_config_manager.add_model("test_model", alias="champion")
    
#     models = model_config_manager.list_models()
#     assert isinstance(models, list)
#     assert "qa_model" in models
#     assert "test_model" in models
