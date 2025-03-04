from pathlib import Path
import os

from loguru import logger
from typing import Optional

from mlflow.tracking import MlflowClient

from serve.utils.mlflow.config import MLFlowConfig, ModelConfig
from serve.utils.gcs.download import download_from_gcs
from ss.experiment_tracker.mlflow_gcs_llamacpp.mlflow.check import model_needs_update

def download_model_artifact(
    model_name="llama-cpp-qa", 
    alias="champion", 
    artifact_path="model_path",
    gcs_bucket="mlflow-artifacts-bucket",
    local_dir: Optional[Path] = None,
    gcs_credentials: Optional[Path] = None
) -> Path:
    """
    Download model artifact directly from Google Cloud Storage using run ID.
    
    Args:
        model_name (str): Name of the model in the registry
        alias (str): Alias of the model version (e.g., 'champion', 'challenger')
        artifact_path (str): Name of the artifact to download
        gcs_bucket (str): Name of the GCS bucket
    
    Returns:
        Path: Local path to the downloaded model file
    """
    mlflow_config = MLFlowConfig.from_env()
    client = MlflowClient()
    
    # Get the model version by alias
    model_version = client.get_model_version_by_alias(model_name, alias)
    run_id = model_version.run_id
    logger.info(f"Run ID: {run_id}")
    
    # Get the run to access artifact URI
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    logger.info(f"Artifact URI: {artifact_uri}")
    
    # Extract the relative path from the artifact URI
    gcs_path = artifact_uri.replace("mlflow-artifacts:", f"gs://{gcs_bucket}")
    logger.info(f"GCS Path: {gcs_path}")
    
    # Parse the GCS path
    gcs_path = gcs_path.replace('gs://', '')
    path_parts = gcs_path.split('/')
    blob_path = '/'.join(path_parts[1:]) + '/' + artifact_path + "/artifacts"
    
    # Set up local paths
    local_dir = local_dir or Path(__file__).parents[3] / "models"
    
    # Download the model
    downloaded_files = download_from_gcs(
        gcs_bucket=gcs_bucket,
        source_path=blob_path,
        destination_path=local_dir,
        credentials=gcs_credentials
    )
    
    if not downloaded_files:
        raise FileNotFoundError(f"No files were downloaded from {blob_path}")
    
    # Update the model config after successful download
    model_config = ModelConfig()
    model_config.update_model_info(run_id, model_name, alias)

    return local_dir

def download_model_artifact_from_gcs(
    model_name="qa_model", 
    alias="champion", 
    artifact_path="model_path",
    gcs_bucket="slmops-dev-ml-artifacts",
    force_download=False,
    local_dir: Optional[Path] = None
):
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "true"
    if force_download:
        download_model_artifact(model_name, alias, artifact_path, gcs_bucket, local_dir)
    else:
        if model_needs_update():
            download_model_artifact(model_name, alias, artifact_path, gcs_bucket)
        else:
            logger.info("Model is up to date")
