import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
from google.cloud import storage
import serve.mlflow.config as config
import os
from tqdm import tqdm
from serve.mlflow.model_config import ModelConfig
from serve.mlflow.check import model_needs_update
from loguru import logger


class TqdmWriter:
    def __init__(self, file_obj, total):
        self.file_obj = file_obj
        self.pbar = tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading the model"
        )

    def write(self, data):
        self.file_obj.write(data)
        self.pbar.update(len(data))

    def flush(self):
        self.file_obj.flush()

    def close(self):
        self.pbar.close()


def download_model_artifact(
    model_name="llama-cpp-qa",
    alias="champion",
    artifact_path="model_path",
    gcs_bucket="mlflow-artifacts-bucket",
):
    """
    Download model artifact directly from Google Cloud Storage using run ID.

    Args:
        model_name (str): Name of the model in the registry
        alias (str): Alias of the model version (e.g., 'champion', 'challenger')
        artifact_path (str): Name of the artifact to download
        gcs_bucket (str): Name of the GCS bucket

    Returns:
        str: Local path to the downloaded artifact
    """
    client = MlflowClient()
    storage_client = storage.Client()

    # Get the model version by alias
    model_version = client.get_model_version_by_alias(model_name, alias)
    run_id = model_version.run_id
    logger.info(f"Run ID: {run_id}")

    # Get the run to access artifact URI
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    logger.info(f"Artifact URI: {artifact_uri}")
    # Extract the relative path from the artifact URI
    # artifact_uri format: gs://bucket/mlflow-artifacts/run_id/artifacts/
    gcs_path = artifact_uri.replace("mlflow-artifacts:", f"gs://{gcs_bucket}")
    logger.info(f"GCS Path: {gcs_path}")

    # Get bucket and blob
    # Parse the GCS path to get the correct bucket and blob path
    gcs_path = gcs_path.replace("gs://", "")  # Remove gs:// prefix
    path_parts = gcs_path.split("/")
    bucket = storage_client.bucket(gcs_bucket)  # Use the provided bucket name
    blob_path = (
        "/".join(path_parts[1:]) + "/" + artifact_path
    )  # Construct full blob path
    blob_path = blob_path + "/artifacts/model.gguf"
    blob = bucket.blob(blob_path)
    blob.reload()
    logger.info(blob_path, blob.size)

    # Create local directory if it doesn't exist
    local_dir = (
        Path(__file__).parents[3] / "models"
    )  # Remove model.gguf from directory path
    os.makedirs(local_dir, exist_ok=True)

    # Download to local path
    local_path = local_dir / "model.gguf"  # Directly specify the output filename
    with open(local_path, "wb") as file_obj:
        writer = TqdmWriter(file_obj, blob.size)
        blob.download_to_file(writer)
        writer.close()

    logger.info(f"Artifact downloaded to: {local_path}")

    # Update the model config after successful download
    model_config = ModelConfig()
    model_config.update_model_info(run_id, model_name, alias)

    return local_path


def download_model_artifact_from_gcs(
    model_name="qa_model",
    alias="champion",
    artifact_path="model_path",
    gcs_bucket="slmops-dev-ml-artifacts",
    force_download=False,
):
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "true"
    if force_download:
        download_model_artifact(model_name, alias, artifact_path, gcs_bucket)
    else:
        if model_needs_update():
            download_model_artifact(model_name, alias, artifact_path, gcs_bucket)
        else:
            logger.info("Model is up to date")
