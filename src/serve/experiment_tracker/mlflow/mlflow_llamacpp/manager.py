from pathlib import Path
from typing import Optional, Union
import os

from loguru import logger
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from serve.servers.llamacpp.serve import LlamaCppServer
from serve.utils.mlflow.config import MLflowModelConfigManager, MLFlowConfig


class ModelManagerError(Exception):
    """Custom exception for model management errors."""
    pass


class ModelManager:
    """Manager for LLaMA.cpp models with MLflow integration.
    
    This class handles the lifecycle of LLaMA.cpp models, including:
    - Adding models from MLflow
    - Serving models through LLaMA.cpp server
    - Managing model configurations and status
    - Stopping and deleting model instances
    """
    
    def __init__(
        self,
        config_path: Optional[Union[Path, str]] = None,
        mlflow_client: Optional[MlflowClient] = None,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize the model manager.
        
        Args:
            config_path: Path to configuration file
            mlflow_client: Pre-configured MLflow client
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow registry server URI
            username: MLflow authentication username
            password: MLflow authentication password
            
        Raises:
            ModelManagerError: If initialization fails
            ValueError: If required parameters are missing
        """
        try:
            # Initialize MLflow configuration
            if not mlflow_client and not tracking_uri:
                raise ValueError(
                    "Either mlflow_client or tracking_uri must be provided"
                )
                
            if tracking_uri:
                try:
                    mlflow_config = MLFlowConfig(
                        tracking_uri=tracking_uri,
                        registry_uri=registry_uri,
                        username=username or os.getenv("MLFLOW_TRACKING_USERNAME"),
                        password=password or os.getenv("MLFLOW_TRACKING_PASSWORD")
                    )
                    mlflow_client = MlflowClient(
                        tracking_uri=mlflow_config.tracking_uri,
                        registry_uri=mlflow_config.registry_uri
                    )
                except Exception as e:
                    raise ModelManagerError(f"Failed to initialize MLflow: {str(e)}")
            
            self.mlflow_client = mlflow_client
            self.config_path = Path(config_path) if config_path else None
            self.serve_manager = LlamaCppServer()
            
            # Initialize model configuration manager
            try:
                self.model_config_manager = MLflowModelConfigManager(
                    mlflow_client=mlflow_client,
                    config_path=self.config_path
                )
            except Exception as e:
                raise ModelManagerError(
                    f"Failed to initialize model configuration manager: {str(e)}"
                )
                
            logger.info("Initialized model manager successfully")
            
        except Exception as e:
            if not isinstance(e, (ModelManagerError, ValueError)):
                raise ModelManagerError(f"Failed to initialize model manager: {str(e)}")
            raise

    def add_model(
        self,
        model_name: str,
        alias: str = "champion",
        artifact_path: str = "model_path",
        model_dir: Optional[Union[Path, str]] = None
    ) -> None:
        """Add a model from MLflow to local storage.
        
        Args:
            model_name: Name of the model in MLflow registry
            alias: Model version alias
            artifact_path: Path to model artifacts in MLflow
            model_dir: Custom directory to save the model
            
        Raises:
            ModelManagerError: If model addition fails
            ValueError: If model_name is empty
        """
        if not model_name:
            raise ValueError("Model name cannot be empty")
            
        try:
            if model_dir:
                model_dir = Path(model_dir)
                
            self.model_config_manager.add_model(
                model_name=model_name,
                alias=alias,
                artifact_path=artifact_path,
                model_dir=model_dir
            )
            logger.info(f"Successfully added model {model_name}")
            
        except Exception as e:
            raise ModelManagerError(f"Failed to add model {model_name}: {str(e)}")

    def model_update_available(self, model_name: str, alias: str ) -> bool:
        return self.model_config_manager.check_for_update(model_name, alias)
    
    def update_serve(
        self,
        model_name: str,

        port: Optional[int] = 8080,
        gguf_relative_path: Optional[str] = "artifacts/model.gguf",
    ) -> None:
       self.delete_serve_model(model_name)
       self.add_serve(model_name, port, gguf_relative_path, update=True)
    
    def add_serve(
        self,
        model_name: str,
        port: Optional[int] = 8080,
        gguf_relative_path: Optional[str] = "artifacts/model.gguf",
        update: bool = False,
        alias: Optional[str] = "champion",
    ) -> None:
        """Start serving a model through LLaMA.cpp server.
        
        Args:
            model_name: Name of the model to serve
            server_port: Port for the server API
            ui_port: Port for the web UI
            
        Raises:
            ModelManagerError: If serving fails
            ValueError: If model not found or ports invalid
        """
        try:
            status = self.serve_manager.get_status(model_name)
            if status and not update:
                logger.info(f"Model {model_name} is already running")
                return
            elif status and update:
                logger.info(f"Updating model {model_name}")
                self.serve_manager.delete_serve(model_name)
            elif status:
                logger.info(f"Model {model_name} is already running")
                return
            # Validate model exists
            if model_name not in self.model_config_manager.list_models():
                self.add_model(model_name , alias=alias)

            # Validate ports
            if not (1024 <= port <= 65535):
                raise ValueError("Ports must be between 1024 and 65535")
                
            # Get model path and verify it exists
            model_path = self.model_config_manager.get_model_path(model_name)
            model_file = model_path / gguf_relative_path
            if not model_file.exists():
                raise ValueError(f"Model file not found: {model_file}")

            
            
            # Start server
            self.serve_manager.add_serve(
                model_id=model_name,
                model_name=model_file.name,
                model_path=model_file.parent,
                port=port,
            )
            logger.info(f"Started serving model {model_name} on ports {port}")
            
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ModelManagerError(f"Failed to serve model {model_name}: {str(e)}")
            raise

    def stop_serve_model(self, model_name: str) -> None:
        """Stop a running model server.
        
        Args:
            model_name: Name of the model to stop
            
        Raises:
            ModelManagerError: If stopping fails
            ValueError: If model not found
        """
        try:
            if model_name not in self.model_config_manager.list_models():
                raise ValueError(f"Model {model_name} not found in configuration")
                
            self.serve_manager.stop(model_name)
            logger.info(f"Stopped model {model_name}")
            
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ModelManagerError(f"Failed to stop model {model_name}: {str(e)}")
            raise
    
    def delete_serve_model(self, model_name: str) -> None:
        try:
            if model_name not in self.model_config_manager.list_models():
                raise ValueError(f"Model {model_name} not found in configuration")
                
            self.serve_manager.delete(model_name)
            logger.info(f"Deleted model {model_name}")
            
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ModelManagerError(f"Failed to delete model {model_name}: {str(e)}")
            raise

    def delete_model(self, model_name: str) -> None:
        """Delete a model and its configuration.
        
        This removes both the model files and configuration entries.
        
        Args:
            model_name: Name of the model to delete
            
        Raises:
            ModelManagerError: If deletion fails
            ValueError: If model not found
        """
        try:
            if model_name not in self.model_config_manager.list_models():
                raise ValueError(f"Model {model_name} not found in configuration")
                
            # Stop the model if it's running
            try:
                self.serve_manager.stop(model_name)
            except Exception as e:
                logger.warning(f"Failed to stop model {model_name}: {str(e)}")
                
            # Delete model files and configuration
            try:
                self.serve_manager.delete(model_name)
            except Exception as e:
                logger.warning(f"Failed to delete server instance: {str(e)}")
                
            self.model_config_manager.delete_model(model_name)
            logger.info(f"Deleted model {model_name}")
            
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ModelManagerError(f"Failed to delete model {model_name}: {str(e)}")
            raise

    def delete_all_serve_models(self) -> None:
        self.serve_manager.delete_all()


