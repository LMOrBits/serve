from pathlib import Path
import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
from llama_cpp import Llama
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient
import pytest
import shutil
from mlflow.artifacts import download_artifacts

mlflow_docker_path = Path(__file__).parents[2] / "scripts/dev/mlflow"
load_dotenv(mlflow_docker_path / "config.env")
mlflow_port = os.getenv("MLFLOW_PORT")

class LlamaGGUFWrapper(PythonModel):
    def load_context(self, context):
        """Load the GGUF model when the model is loaded."""
        model_path = context.artifacts["model_path"]
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Adjust context window if needed
            n_threads=4   # Adjust based on your hardware
        )
    
    def predict(self, context, model_input):
        """Make predictions using the GGUF model."""
        if isinstance(model_input, pd.DataFrame):
            if "prompt" not in model_input.columns:
                raise ValueError("Input DataFrame must contain a 'prompt' column")
            prompts = model_input["prompt"].tolist()
        else:
            prompts = model_input
            
        results = []
        for prompt in prompts:
            output = self.model(
                prompt,
                max_tokens=128,  # Adjust as needed
                temperature=0.7,
                stop=["</s>"],   # Adjust stop tokens as needed
            )
            results.append(output["choices"][0]["text"])
            
        return results

@pytest.fixture
def tracking_uri():
    mlflow_port = os.getenv("MLFLOW_PORT")
    tracking_uri = f"http://localhost:{mlflow_port}/"
    return tracking_uri

@pytest.fixture
def model_name():
    return "rag_model"

@pytest.fixture
def models_dir():
    """Fixture to create and clean up the models directory."""
    models_path = Path(__file__).parent / "models"
    models_path.mkdir(exist_ok=True)
    yield models_path
    # Clean up after tests
    # if models_path.exists():
    #     shutil.rmtree(models_path)

def test_gguf_model_ingestion(tracking_uri: str, model_name: str):
    """Test ingesting and using a GGUF model with MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    
    # Path to your GGUF model
    model_save_path = f"{Path(__file__).parent}/saved_model"
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the custom model
        mlflow.pyfunc.log_model(
            artifact_path="model_path",
            python_model=LlamaGGUFWrapper(),
            artifacts={
                "model_path": f"{model_save_path}/model.gguf"
            },
            pip_requirements=[
                "mlflow==2.4.0",
                "llama-cpp-python",
                "pandas"
            ]
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_details = mlflow.register_model(
            model_uri=model_uri, name=model_name
        )
        client = MlflowClient(tracking_uri=tracking_uri)
        client.update_model_version(
            name=model_name,
            version=model_details.version,
            description="Test model version"
        )
        # Verify model exists in registry
        model_versions = client.search_model_versions(f"name='{model_name}'")
        assert len(model_versions) > 0, "Model not found in registry"   

        # # Load the model back
        # model_uri = f"runs:/{run.info.run_id}/model"
        # loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # # Test the model with a sample input
        # test_input = pd.DataFrame({
        #     "prompt": ["Tell me a short joke about programming."]
        # })
        
        # result = loaded_model.predict(test_input)
        # assert isinstance(result, list)
        # assert len(result) > 0
        # print("Model prediction:", result[0])

def test_download_and_store_model(tracking_uri: str, model_name: str, models_dir: Path):
    """Test downloading a model from MLflow and storing it locally."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # First ensure we have a model to download
    model_versions = client.search_model_versions(f"name='{model_name}'")
    assert len(model_versions) > 0, "No model versions found to download"
    
    # Get the latest version
    latest_version = max(int(mv.version) for mv in model_versions)
    model_version = client.get_model_version(model_name, str(latest_version))
    
    # Create a unique directory for this model version
    model_save_dir = models_dir / f"{model_name}_v{latest_version}"
    model_save_dir.mkdir(exist_ok=True)
    
    # Download the model
    download_artifacts(
        run_id=model_version.run_id,
        artifact_path="model_path",
        dst_path=str(model_save_dir)
    )
    
    # Verify the model files were downloaded
    assert model_save_dir.exists(), "Model directory was not created"
    assert any(model_save_dir.iterdir()), "No files were downloaded"
    
    # # Try loading the model to verify it's valid
    # try:
    #     _ = Llama(
    #         model_path=str(model_save_dir / "model.gguf"),
    #         n_ctx=2048,
    #         n_threads=4
    #     )
    #     model_loads = True
    # except Exception as e:
    #     model_loads = False
    #     print(f"Error loading model: {e}")
    
    # assert model_loads, "Downloaded model could not be loaded"
    
    # return model_save_dir
