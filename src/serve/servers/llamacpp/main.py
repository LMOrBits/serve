import click
from pathlib import Path
from typing import Optional
from loguru import logger

from serve._cli import TaskCLI


def setup_logger():
    """Configure logger settings for the application."""
    logger.add(
        "llamacpp_server.log",
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


@click.group()
def llama_cpp():
    """CLI tool for LLaMA.cpp model management and inference.
    
    This tool provides commands to manage LLaMA.cpp model instances,
    including serving models, checking status, and stopping servers.
    """
    setup_logger()
    logger.info("Starting LLaMA.cpp CLI tool")


@llama_cpp.command()
@click.option('--model-path', 
              type=click.Path(exists=True, path_type=Path),
              help='Path to model directory')
@click.option('--server-port',
              type=int,
              default=8000,
              help='Server port (1024-65535)',
              callback=lambda ctx, param, value: value if 1024 <= value <= 65535 else None)
@click.option('--ui-port',
              type=int,
              default=8080,
              help='UI port (1024-65535)',
              callback=lambda ctx, param, value: value if 1024 <= value <= 65535 else None)
@click.option('--model-name',
              type=str,
              default='model.gguf',
              help='Model filename')
@click.option('--model-id',
              type=str,
              default='model',
              help='Unique model identifier')
def serve(model_path: Optional[Path],
         server_port: int,
         ui_port: int,
         model_name: str,
         model_id: str) -> None:
    """Start the LLaMA.cpp server with specified configuration.
    
    Args:
        model_path: Directory containing the model file
        server_port: Port for the server API (1024-65535)
        ui_port: Port for the web UI (1024-65535)
        model_name: Name of the model file
        model_id: Unique identifier for this model instance
    """
    try:
        cli = TaskCLI(Path(__file__).parent)
        
        if model_path is None:
            model_path = Path(__file__).parents[3] / "models"
        
        if not model_path.exists():
            raise click.BadParameter(f"Model path does not exist: {model_path}")
            
        if server_port == ui_port:
            raise click.BadParameter("Server port and UI port must be different")
            
        logger.info(f"Starting LLaMA.cpp server with model {model_name} at {model_path}")
        
        cli.run("serve",
                MODEL_PATH=model_path.absolute(),
                SERVER_PORTS=server_port,
                UI_PORT=ui_port,
                MODEL_NAME=model_name,
                MODEL_ID=model_id)
                
        logger.info(f"Server started successfully with model ID: {model_id}")
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise click.ClickException(str(e))


@llama_cpp.command()
@click.option('--model-id',
              type=str,
              default='model',
              help='Model identifier to check status for')
@click.option('--all',
              is_flag=True,
              help='Show status for all models')
def status(model_id: str, all: bool) -> None:
    """Check the status of LLaMA.cpp server instances.
    
    Args:
        model_id: The unique identifier of the model to check
        all: If True, show status for all running models
    """
    try:
        cli = TaskCLI(Path(__file__).parent)
        logger.info(f"Checking status for model{'s' if all else f' {model_id}'}")
        cli.run("status", MODEL_ID=model_id, ALL=all)
        
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise click.ClickException(str(e))


@llama_cpp.command()
@click.option('--model-id',
              type=str,
              default='model',
              help='Model identifier to stop')
def stop(model_id: str) -> None:
    """Stop a running LLaMA.cpp server instance.
    
    Args:
        model_id: The unique identifier of the model instance to stop
    """
    try:
        cli = TaskCLI(Path(__file__).parent)
        logger.info(f"Stopping server for model ID: {model_id}")
        cli.run("stop", MODEL_ID=model_id)
        logger.info(f"Server stopped successfully")
        
    except Exception as e:
        logger.error(f"Failed to stop server: {str(e)}")
        raise click.ClickException(str(e))


@llama_cpp.command()
@click.option('--model-id',
              type=str,
              default='model',
              help='Model identifier to delete')
def delete(model_id: str) -> None:
    """Delete a LLaMA.cpp server instance.
    
    Args:
        model_id: The unique identifier of the model instance to delete
    """
    try:
        cli = TaskCLI(Path(__file__).parent)
        logger.info(f"Deleting server instance for model ID: {model_id}")
        cli.run("delete", MODEL_ID=model_id)
        logger.info(f"Server instance deleted successfully")
        
    except Exception as e:
        logger.error(f"Failed to delete server instance: {str(e)}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    llama_cpp()