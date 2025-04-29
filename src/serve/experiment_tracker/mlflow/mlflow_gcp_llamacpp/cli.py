from typing import Optional
import click
from serve._cli import TaskCLI
from pathlib import Path
from loguru import logger

from serve.experiment_tracker.mlflow_gcp_llamacpp.manager import ModelTracker


model_config_path = Path(__file__).parents[4] / "models" 
model_manager = None

@click.group()
def mlflow_gcp_llamacpp():
    """CLI tool for model management and inference."""
    pass


@mlflow_gcp_llamacpp.command()
@click.option('--model-config-path', default=model_config_path, help='Path to model config')
def init(model_config_path:str):
    global model_manager
    model_manager = ModelTracker(model_config_path=model_config_path)

@mlflow_gcp_llamacpp.command()
def add_model(
    model_name: str,
    alias: str = "champion",
    artifact_path: str = "model_path",
    gcs_bucket: str = "mlflow-artifacts-bucket",
    local_dir: Optional[Path] = None,
    gcs_credentials: Optional[Path] = None
):
    global model_manager
    model_manager.add_model(model_name, alias, artifact_path, gcs_bucket, local_dir, gcs_credentials)

@mlflow_gcp_llamacpp.command()
def status():
    global model_manager
    model_manager.get_model_status()