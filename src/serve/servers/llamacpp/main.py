import click
from serve._cli import TaskCLI
from pathlib import Path
from loguru import logger




@click.group()
def llama_cpp():
    """CLI tool for model management and inference."""
    pass

@llama_cpp.command()
@click.option('--model-path', default=None, help='Path to model directory')
@click.option('--server-port', default='8000', help='Server port')
@click.option('--ui-port', default='8080', help='UI port')
@click.option('--model-name', default='model.gguf', help='Model filename')
@click.option('--model-id', default='model', help='Model ID')
def serve( model_path, server_port, ui_port, model_name, model_id):
    """Start the LlamaCpp server"""
    cli = TaskCLI(Path(__file__).parent)
    if model_path is None:
        model_path = Path(__file__).parents[3] / "models"
    else:
        model_path = Path(model_path).resolve()
    
    logger.info(f"Starting LlamaCpp server with model {model_name} at {model_path}")
    cli.run("serve", 
            MODEL_PATH=model_path.absolute(),
            SERVER_PORTS=server_port,
            UI_PORT=ui_port,
            MODEL_NAME=model_name,
            MODEL_ID=model_id)

@llama_cpp.command()
@click.option('--model-id', default='model', help='Model ID')
@click.option('--all', is_flag=True, help='Show all models')
def status(model_id, all):
    """Check the status of the LlamaCpp server"""
    cli = TaskCLI(Path(__file__).parent)
    cli.run("status", MODEL_ID=model_id, ALL=False)


@llama_cpp.command()
@click.option('--model-id', default='model', help='Model ID')
def stop(model_id):
    """Stop the LlamaCpp server"""
    cli = TaskCLI(Path(__file__).parent)
    cli.run("stop", MODEL_ID=model_id)

@llama_cpp.command()
@click.option('--model-id', default='model', help='Model ID')
def delete(model_id):
    """Delete the LlamaCpp server"""
    cli = TaskCLI(Path(__file__).parent)
    cli.run("delete", MODEL_ID=model_id)

llama_cpp.add_command(serve)