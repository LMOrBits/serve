# export GOOGLE_APPLICATION_CREDENTIALS="/Users/parsa/Desk/projects/university/slmops-project/infra/provision/cloud/terraform/environments/dev/keys/super-admin-key.json"

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

load_dotenv(Path(__file__).parent / ".env")
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
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    return tracking_uri

@pytest.fixture
def model_name():
    return "rag_model"

@pytest.fixture
def models_dir():
    """Fixture to create and clean up the models directory."""
    models_path = Path(__file__).parent / "model/m1/model_path/artifacts"
    models_path.mkdir(exist_ok=True)
    yield models_path
    # Clean up after tests
    # if models_path.exists():
    #     shutil.rmtree(models_path)

def test_gguf_model_ingestion(tracking_uri: str, model_name: str, models_dir: Path):
    """Test ingesting and using a GGUF model with MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"
    # For example, use a 50 MB threshold: if file size >= 50 MB, use multipart upload.
    os.environ["MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE"] = str(50 * 1024 * 1024)
    # Set chunk size to 10 MB (adjust based on your network/storage performance).
    os.environ["MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE"] = str(10 * 1024 * 1024)
    
    # Path to your GGUF model
    model_save_path = f"{models_dir}"
    
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


def test_download_and_store_model(tracking_uri: str, model_name: str, models_dir: Path):
    """Test downloading a model from MLflow and storing it locally."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/parsa/Desk/projects/university/slmops-project/slmops-thesis/lmorbits/infrastructure/cloud/terraform/environments/dev/keys/storage-user-key.json"
    
    # First ensure we have a model to download
    model_versions = client.search_model_versions(f"name='{model_name}'")
    assert len(model_versions) > 0, "No model versions found to download"
    
    from serve.servers.llamacpp.serve import LlamaCppServer
    here = Path(__file__).parent
    model_check = here / "model_check"
    model_check.mkdir(exist_ok=True)
    server = LlamaCppServer(model_check, client, gcp=True)

    server.update_model(model_name , "champion")

def test_gcp_client():
    from google.cloud import storage
    from google.oauth2 import service_account
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/parsa/Desk/projects/university/slmops-project/slmops-thesis/lmorbits/infrastructure/cloud/terraform/environments/dev/keys/storage-user-key.json"
    
    credentials = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    client = storage.Client()
    bucket = client.bucket("slmops-dev-ml-artifacts")
    blobs = bucket.list_blobs()
    print("================")
    print(blobs)
    print("================")
    # Set the path to the service account key file
    