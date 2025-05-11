from pathlib import Path

from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts


def get_model_run_id(mlflow_client: MlflowClient, model_name: str, alias: str ):
    model_version = mlflow_client.get_model_version_by_alias(model_name, alias)
    if not model_version:
        raise ValueError(f"No model version found for {model_name} with alias {alias}")
    return model_version.run_id

def get_model(mlflow_client: MlflowClient, model_name: str, alias: str, desired_path: Path, artifact_path: str):
    model_version = mlflow_client.get_model_version_by_alias(model_name, alias)
    if not model_version:
        raise ValueError(f"No model version found for {model_name} with alias {alias}")

    model_save_dir = Path(desired_path) / f"{model_name}"
    model_save_dir.mkdir(exist_ok=True)
    
    # Download the model
    download_artifacts(
        run_id=model_version.run_id,
        artifact_path=artifact_path,
        dst_path=str(model_save_dir)
    )
    return model_save_dir , model_version.run_id