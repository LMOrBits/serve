from pathlib import Path
from typing import Optional, List, Union
from dataclasses import dataclass
from loguru import logger

from pydantic import BaseModel, field_validator, FieldValidationInfo
from serve._cli.task import TaskCLI


class LlamaCppServerConfig(BaseModel):
    """Configuration for LLaMA CPP Server instance.

    Attributes:
        server_port: Port number for the server API
        ui_port: Port number for the web UI
        model_name: Name of the model file
        model_id: Unique identifier for the model instance
        model_path: Path to the model file
    """

    port: Optional[int] = 8080
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[Union[Path, str]] = None

    @field_validator('port')
    @classmethod
    def validate_ports(cls, v):
        if v is not None and not (1024 <= v <= 65535):
            raise ValueError("Port number must be between 1024 and 65535")
        return v

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v):
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Model path does not exist: {v}")
        return v


class LlamaCppServer:
    """Manager class for LLaMA CPP server instances.
    
    This class handles multiple LLaMA model instances, managing their lifecycle
    including serving, stopping, and status monitoring.
    """
    
    def __init__(self):
        """Initialize the LLaMA CPP server manager."""
        self.cli = TaskCLI(Path(__file__).parent)
        self.configs: List[LlamaCppServerConfig] = []
        logger.info("Initialized LLaMA CPP server manager")

    def add_serve(self,
                 port: Optional[int] = 8080,
                 model_name: Optional[str] = None,
                 model_id: Optional[str] = None,
                 model_path: Optional[Union[Path, str]] = None) -> None:
        """Add and start a new model server instance.
        
        Args:
            server_port: Port for the server API
            ui_port: Port for the web UI
            model_name: Name of the model file
            model_id: Unique identifier for the model instance
            model_path: Path to the model file
            
        Raises:
            ValueError: If the configuration is invalid
        """
        try:
            config = LlamaCppServerConfig(
                port=port,
                model_name=model_name,
                model_id=model_id,
                model_path=model_path
            )
            
            if any(c.model_id == model_id for c in self.configs):
                raise ValueError(f"Model ID '{model_id}' already exists")
                
            self.serve(
                model_path=config.model_path,
                port=config.port,
                model_name=config.model_name,
                model_id=config.model_id
            )
            self.configs.append(config)
            logger.info(f"Added new server instance with model ID: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to add server instance: {str(e)}")
            raise

    def delete_serve(self, model_id: str) -> None:
        """Delete a server instance by its model ID.
        
        Args:
            model_id: The unique identifier of the model instance to delete
            
        Raises:
            ValueError: If the model ID doesn't exist
        """
        if not any(config.model_id == model_id for config in self.configs):
            raise ValueError(f"Model ID '{model_id}' not found")
            
        try:
            self.cli.run("delete", MODEL_ID=model_id)
            self.configs = [config for config in self.configs if config.model_id != model_id]
            logger.info(f"Deleted server instance with model ID: {model_id}")
        except Exception as e:
            logger.error(f"Failed to delete server instance: {str(e)}")
            raise

    def serve(self,
             model_path: Optional[Union[Path, str]] = None,
             port: Optional[int] = 8080,
             model_name: Optional[str] = None,
             model_id: Optional[str] = None) -> None:
        """Start serving a model instance.
        
        Args:
            model_path: Path to the model file
            server_port: Port for the server API
            ui_port: Port for the web UI
            model_name: Name of the model file
            model_id: Unique identifier for the model instance
            
        Raises:
            Exception: If the server fails to start
        """
        try:
            self.cli.run("serve",
                        MODEL_PATH=model_path,
                        PORT=port,
                        MODEL_NAME=model_name,
                        MODEL_ID=model_id)
            logger.info(f"Started server for model ID: {model_id}")
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

    def get_status(self, model_id: str) -> dict:
        """Get status of a specific model instance.
        
        Args:
            model_id: The unique identifier of the model instance
            
        Returns:
            dict: Status information for the model instance
            
        Raises:
            ValueError: If the model ID doesn't exist
        """
        try:
            if not any(config.model_id == model_id for config in self.configs):
                raise ValueError(f"Model ID '{model_id}' not found")
            
            self.cli.run("status", MODEL_ID=model_id, ALL=False)
            return True
        except Exception as e:
            logger.error(f"Failed to get status for model {model_id}: {str(e)}")
            return None

    def get_all_statuses(self) -> List[dict]:
        """Get status of all model instances.
        
        Returns:
            List[dict]: Status information for all model instances
        """
        statuses = []
        for config in self.configs:
            try:
                status = self.get_status(config.model_id)
                statuses.append(status)
            except Exception as e:
                logger.warning(f"Failed to get status for model {config.model_id}: {str(e)}")
        return statuses

    def stop(self, model_id: str) -> None:
        """Stop a model instance.
        
        Args:
            model_id: The unique identifier of the model instance to stop
            
        Raises:
            ValueError: If the model ID doesn't exist
        """
        if not any(config.model_id == model_id for config in self.configs):
            raise ValueError(f"Model ID '{model_id}' not found")
            
        try:
            self.cli.run("stop", MODEL_ID=model_id)
            logger.info(f"Stopped server instance with model ID: {model_id}")
        except Exception as e:
            logger.error(f"Failed to stop server instance: {str(e)}")
            raise

    def delete(self, model_id: str, all: bool = False) -> None:
        """Delete a model instance.
        
        Args:
            model_id: The unique identifier of the model instance to delete
            
        Raises:
            ValueError: If the model ID doesn't exist
        """
        if all:
            self.cli.run("delete", ALL=all)
            return
        elif not any(config.model_id == model_id for config in self.configs):
            raise ValueError(f"Model ID '{model_id}' not found")
            
        try:
            self.cli.run("delete", MODEL_ID=model_id, ALL=all)
            logger.info(f"Deleted server instance with model ID: {model_id}")
        except Exception as e:
            logger.error(f"Failed to delete server instance: {str(e)}")
            raise

    @staticmethod
    def delete_all() -> None:
        """Delete all model instances."""
        server = LlamaCppServer()
        server.delete("", all=True)
