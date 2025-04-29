from typing import Optional, Union
from pathlib import Path

import click
from loguru import logger

from serve.experiment_tracker.mlflow.mlflow_llamacpp.manager import ModelManager


# Default paths and configurations
DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "models"
DEFAULT_GCS_BUCKET = "mlflow-artifacts-bucket"
DEFAULT_ARTIFACT_PATH = "model_path"

# Global model manager instance
model_manager: Optional[ModelManager] = None


def validate_model_manager() -> None:
    """Ensure model manager is initialized.
    
    Raises:
        click.UsageError: If model manager not initialized
    """
    if model_manager is None:
        raise click.UsageError(
            "Model manager not initialized. Run 'init' command first."
        )


@click.group()
def mlflow_llamacpp():
    """CLI tool for LLaMA.cpp model management with MLflow integration.
    
    This tool provides commands to:
    - Initialize the model manager
    - Add models from MLflow
    - Check model status
    - Delete models
    """
    pass


@mlflow_llamacpp.command()
@click.option(
    '--model-config-path',
    type=click.Path(path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    help='Path to model configuration directory'
)
@click.option(
    '--tracking-uri',
    type=str,
    help='MLflow tracking server URI'
)
@click.option(
    '--registry-uri',
    type=str,
    help='MLflow registry server URI'
)
@click.option(
    '--username',
    type=str,
    envvar='MLFLOW_TRACKING_USERNAME',
    help='MLflow username'
)
@click.option(
    '--password',
    type=str,
    envvar='MLFLOW_TRACKING_PASSWORD',
    help='MLflow password'
)
def init(
    model_config_path: Path,
    tracking_uri: Optional[str],
    registry_uri: Optional[str],
    username: Optional[str],
    password: Optional[str]
) -> None:
    """Initialize the model manager with MLflow configuration.
    
    Args:
        model_config_path: Directory for model configurations
        tracking_uri: MLflow tracking server URI
        registry_uri: MLflow registry server URI
        username: MLflow authentication username
        password: MLflow authentication password
    """
    try:
        global model_manager
        model_manager = ModelManager(
            config_path=model_config_path,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            username=username,
            password=password
        )
        logger.info(f"Initialized model manager with config path: {model_config_path}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {str(e)}")
        raise click.ClickException(str(e))


@mlflow_llamacpp.command()
@click.argument('model_name')
@click.option(
    '--alias',
    type=str,
    default="champion",
    help='Model version alias'
)
@click.option(
    '--artifact-path',
    type=str,
    default=DEFAULT_ARTIFACT_PATH,
    help='Path to model artifacts in MLflow'
)
@click.option(
    '--model-dir',
    type=click.Path(path_type=Path),
    help='Custom directory to save model'
)
def add_model(
    model_name: str,
    alias: str,
    artifact_path: str,
    model_dir: Optional[Path]
) -> None:
    """Add a model from MLflow to local storage.
    
    Args:
        model_name: Name of the model in MLflow registry
        alias: Model version alias
        artifact_path: Path to model artifacts in MLflow
        model_dir: Custom directory to save model
    """
    try:
        validate_model_manager()
        
        if not model_name:
            raise click.BadParameter("Model name cannot be empty")
            
        model_manager.add_model(
            model_name=model_name,
            alias=alias,
            artifact_path=artifact_path,
            model_dir=model_dir
        )
        logger.info(f"Successfully added model: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to add model: {str(e)}")
        raise click.ClickException(str(e))


@mlflow_llamacpp.command()
@click.argument('model_name', required=False)
def status(model_name: Optional[str] = None) -> None:
    """Check status of models.
    
    Args:
        model_name: Optional specific model to check.
                   If not provided, checks all models.
    """
    try:
        validate_model_manager()
        
        if model_name:
            status = model_manager.get_model_status(model_name)
            click.echo(f"Status for model {model_name}:")
            click.echo(status)
        else:
            statuses = model_manager.get_model_status()
            click.echo("Status for all models:")
            for name, status in statuses.items():
                click.echo(f"\n{name}:")
                click.echo(status)
                
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise click.ClickException(str(e))


@mlflow_llamacpp.command()
@click.argument('model_name')
def delete(model_name: str) -> None:
    """Delete a model and its configuration.
    
    Args:
        model_name: Name of the model to delete
    """
    try:
        validate_model_manager()
        
        if not model_name:
            raise click.BadParameter("Model name cannot be empty")
            
        if click.confirm(f"Are you sure you want to delete model {model_name}?"):
            model_manager.delete_model(model_name)
            logger.info(f"Successfully deleted model: {model_name}")
        else:
            click.echo("Deletion cancelled")
            
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise click.ClickException(str(e))


@mlflow_llamacpp.command()
@click.argument('model_name')
@click.option(
    '--server-port',
    type=int,
    default=8000,
    help='Port for server API'
)
@click.option(
    '--ui-port',
    type=int,
    default=8080,
    help='Port for web UI'
)
def serve(
    model_name: str,
    server_port: int,
    ui_port: int
) -> None:
    """Start serving a model through LLaMA.cpp server.
    
    Args:
        model_name: Name of the model to serve
        server_port: Port for server API
        ui_port: Port for web UI
    """
    try:
        validate_model_manager()
        
        if not model_name:
            raise click.BadParameter("Model name cannot be empty")
            
        model_manager.serve_model(
            model_name=model_name,
            server_port=server_port,
            ui_port=ui_port
        )
        logger.info(
            f"Started serving model {model_name} on "
            f"ports {server_port}/{ui_port}"
        )
        
    except Exception as e:
        logger.error(f"Failed to serve model: {str(e)}")
        raise click.ClickException(str(e))


@mlflow_llamacpp.command()
@click.argument('model_name')
def stop(model_name: str) -> None:
    """Stop a running model server.
    
    Args:
        model_name: Name of the model to stop
    """
    try:
        validate_model_manager()
        
        if not model_name:
            raise click.BadParameter("Model name cannot be empty")
            
        model_manager.stop_model(model_name)
        logger.info(f"Stopped model: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to stop model: {str(e)}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    mlflow_llamacpp()