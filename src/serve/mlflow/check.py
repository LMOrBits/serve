import mlflow

from loguru import logger
from mlflow.tracking import MlflowClient

from serve.mlflow.config import config_init
from serve.mlflow.model_config import ModelConfig


def model_needs_update(model_name: str = "qa_model", alias: str = "champion") -> bool:
    """
    Check if the current model is up to date.

    Args:
        model_name (str): Name of the model in the registry
        alias (str): Alias of the model version

    Returns:
        bool: True if model is up to date, False if it needs updating
    """
    mlflow_config = config_init()
    mlflow.set_tracking_uri(mlflow_config["mlflow"]["URL"])
    client = MlflowClient()
    model_config = ModelConfig()
    logger.info(f"Checking if model {model_name} with alias {alias} is up to date")

    # Get the current model version by alias

    model_version = client.get_model_version_by_alias(model_name, alias)
    current_run_id = model_version.run_id
    logger.info(f"latest Model ID: {current_run_id}")
    # Load stored configuration
    config = model_config.load_config()
    stored_run_id = config["current_model"]["run_id"]
    logger.info(f"Stored Model ID: {stored_run_id}")
    if stored_run_id is None:
        return True

    # Compare run IDs
    return stored_run_id != current_run_id
