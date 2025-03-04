from typing import Dict, List, Optional, Union
from pathlib import Path
import os
import shutil
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.artifacts import download_artifacts
from pydantic import BaseModel, model_validator, Field, ValidationError
from loguru import logger

from serve.utils.model_config import ModelConfig


@dataclass
class ModelVersionInfo:
    """Information about a model version."""
    run_id: str
    version: str
    status: str
    alias: str


class MLFlowConfig(BaseModel):
    """MLFlow configuration for connecting to MLflow server.
    
    Attributes:
        tracking_uri: URI for MLflow tracking server
        registry_uri: URI for MLflow model registry (defaults to tracking_uri)
        username: Username for MLflow authentication
        password: Password for MLflow authentication
    """
    tracking_uri: str = Field(..., description="URI for MLflow tracking server")
    registry_uri: Optional[str] = Field(None, description="URI for MLflow model registry")
    username: str = Field(..., description="Username for MLflow authentication")
    password: str = Field(..., description="Password for MLflow authentication")

    @model_validator(mode='after')
    def set_registry_uri(self) -> "MLFlowConfig":
        """Set registry URI to tracking URI if not provided."""
        if self.registry_uri is None:
            self.registry_uri = self.tracking_uri
        return self
    
    @model_validator(mode='after')
    def set_environment_variables(self) -> "MLFlowConfig":
        """Set MLflow environment variables from config values."""
        env_vars = {
            "MLFLOW_TRACKING_URI": self.tracking_uri,
            "MLFLOW_TRACKING_USERNAME": self.username,
            "MLFLOW_TRACKING_PASSWORD": self.password
        }
        
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = value
            elif key not in os.environ:
                raise ValueError(f"{key} must be set either in config or environment")
                
        return self

    @classmethod
    def from_env(cls) -> "MLFlowConfig":
        """Create configuration from environment variables.
        
        Returns:
            MLFlowConfig: Configuration instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = {
            "MLFLOW_TRACKING_URI": "tracking URI",
            "MLFLOW_TRACKING_USERNAME": "username",
            "MLFLOW_TRACKING_PASSWORD": "password"
        }
        
        config_dict = {}
        for env_var, description in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                raise ValueError(f"Missing {description}: {env_var} environment variable must be set")
            config_dict[env_var.lower().replace('mlflow_', '')] = value
            
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI")
        if registry_uri:
            config_dict['registry_uri'] = registry_uri
            
        try:
            return cls(**config_dict)
        except ValidationError as e:
            logger.error(f"Invalid MLflow configuration: {str(e)}")
            raise


class MLflowModelConfigManager:
    """Manager for MLflow model configurations and artifacts.
    
    This class handles the interaction between local model storage and MLflow,
    including downloading models, checking for updates, and managing model configurations.
    """
    
    def __init__(self, mlflow_client: Optional[MlflowClient] = None, config_path: Optional[Path] = None) -> None:
        """Initialize MLflow model configuration manager.
        
        Args:
            mlflow_client: Custom MLflow client instance
            config_path: Path to the configuration file
            
        Raises:
            ValidationError: If MLflow configuration is invalid
            MlflowException: If unable to connect to MLflow server
        """
        try:
            if mlflow_client is None:
                mlflow_config = MLFlowConfig.from_env()
                self.mlflow_client = MlflowClient(
                    tracking_uri=mlflow_config.tracking_uri,
                    registry_uri=mlflow_config.registry_uri
                )
            else:
                self.mlflow_client = mlflow_client

            self.model_config = ModelConfig(config_path=config_path)
            logger.info("Initialized MLflow model configuration manager")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow manager: {str(e)}")
            raise

    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: Path) -> None:
        """Download model artifacts from MLflow.
        
        Args:
            run_id: Run ID of the model
            artifact_path: Path to the model artifacts in MLflow
            dst_path: Local destination path for artifacts
            
        Raises:
            MlflowException: If download fails
        """
        try:
            mlflow.set_tracking_uri(self.mlflow_client.tracking_uri)
            download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=str(dst_path.resolve())
            )
            logger.info(f"Downloaded artifacts to {dst_path}")
            
        except Exception as e:
            logger.error(f"Failed to download artifacts: {str(e)}")
            raise MlflowException(f"Failed to download artifacts: {str(e)}")

    def add_model(self,
                 model_name: str,
                 alias: str = "champion",
                 artifact_path: str = "model_path",
                 model_dir: Optional[Path] = None) -> Path:
        """Download model and update local configuration.
        
        Args:
            model_name: Name of the model in registry
            alias: Model version alias
            artifact_path: Path to model artifacts in MLflow
            model_dir: Custom directory to save the model
            
        Returns:
            Path: Path where the model is saved
            
        Raises:
            ValueError: If model version not found
            MlflowException: If download fails
        """
        try:
            #check if the model is already exists and updated
            try:
                if self.check_for_update(model_name, alias):
                    logger.info(f"Model {model_name} is already up to date")
                    return self.get_model_path(model_name)
            except Exception as e:

                model_version = self.mlflow_client.get_model_version_by_alias(model_name, alias)
                if not model_version:
                    raise ValueError(f"No model version found for {model_name} with alias {alias}")
                    
                logger.info(f"Found model version: {model_version.version} for {model_name}")
                run_id = model_version.run_id
                
                model_dir = model_dir or self.model_config.config_path.parent / "models" / model_name
                
                # Download artifacts
                self.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=model_dir)
                    
                # Update configuration
                self.model_config.update_model_info(
                    run_id=run_id,
                    model_name=model_name,
                    alias=alias,
                    model_dir=model_dir
                )
                
                return model_dir
            
        except MlflowException as e:
            logger.error(f"MLflow error while adding model {model_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while adding model {model_name}: {str(e)}")
            raise

    def check_for_update(self, model_name: str, alias: str = "champion") -> bool:
        """Check if a model needs updating.
        
        Args:
            model_name: Name of the model in registry
            alias: Model version alias
            
        Returns:
            bool: True if update needed, False otherwise
            
        Raises:
            ValueError: If model version not found
            MlflowException: If unable to check version
        """
        try:
            model_version = self.mlflow_client.get_model_version_by_alias(model_name, alias)
            if not model_version:
                raise ValueError(f"No model version found for {model_name} with alias {alias}")
                
            current_run_id = model_version.run_id
            stored_config = self.model_config.load_model_config(model_name)
            
            if not stored_config.run_id:
                logger.info(f"No local version found for {model_name}")
                return True
                
            needs_update = stored_config.run_id != current_run_id
            if needs_update:
                logger.info(f"Update available for {model_name}")
            return needs_update
            
        except MlflowException as e:
            logger.error(f"MLflow error checking {model_name} version: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error checking {model_name} version: {str(e)}")
            raise

    def check_for_updates(self) -> Dict[str, Optional[bool]]:
        """Check for updates for all models.
        
        Returns:
            Dict[str, Optional[bool]]: Model names mapped to update status
            (None indicates check failed)
        """
        try:
            all_configs = self.model_config.load_config()
            if not all_configs:
                logger.info("No models configured locally")
                return {}
                
            updates = {}
            for model_name, stored_config in all_configs.items():
                try:
                    updates[model_name] = self.check_for_update(
                        model_name, 
                        stored_config.alias or "champion"
                    )
                except Exception as e:
                    logger.warning(f"Failed to check updates for {model_name}: {str(e)}")
                    updates[model_name] = None
                    
            return updates
            
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
            raise

    def delete_model(self, model_name: str) -> None:
        """Delete model files and remove from configuration.
        
        Args:
            model_name: Name of the model to delete
            
        Raises:
            ValueError: If model not found in configuration
            OSError: If unable to delete model files
        """
        try:
            stored_config = self.model_config.load_model_config(model_name)
            if not stored_config.model_name:
                raise ValueError(f"Model {model_name} not found in configuration")
                
            # Delete model directory if it exists
            if stored_config.model_dir:
                model_path = Path(stored_config.model_dir)
                if model_path.exists():
                    try:
                        shutil.rmtree(model_path)
                        logger.info(f"Deleted model directory: {model_path}")
                    except OSError as e:
                        logger.error(f"Failed to delete model directory: {str(e)}")
                        raise
            
            # Remove from configuration
            config = self.model_config.load_config()
            if model_name in config:
                del config[model_name]
                self.model_config.save_config({k: v.model_dump() for k, v in config.items()})
                logger.info(f"Removed {model_name} from configuration")
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            raise

    def get_model_status(self, model_name: Optional[str] = None) -> Dict[str, dict]:
        """Get current status for one or all models.
        
        Args:
            model_name: Specific model name or None for all models
            
        Returns:
            Dict[str, dict]: Status information for requested model(s)
            
        Raises:
            ValueError: If specified model not found
        """
        try:
            if model_name:
                status = self._get_single_model_status(model_name)
                if not status:
                    raise ValueError(f"Model {model_name} not found")
                return {model_name: status}
            
            # Get status for all models
            all_configs = self.model_config.load_config()
            if not all_configs:
                return {}
                
            statuses = {}
            for name in all_configs:
                try:
                    status = self._get_single_model_status(name)
                    if status:
                        statuses[name] = status
                except Exception as e:
                    logger.warning(f"Failed to get status for {name}: {str(e)}")
                    statuses[name] = {"error": str(e)}
                    
            return statuses
            
        except Exception as e:
            logger.error(f"Error getting model status: {str(e)}")
            raise

    def _get_single_model_status(self, model_name: str) -> dict:
        """Get detailed status for a single model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            dict: Status information including version, location, and update status
            
        Raises:
            ValueError: If model not found
            MlflowException: If unable to get model version
        """
        try:
            stored_config = self.model_config.load_model_config(model_name)
            if not stored_config.model_name:
                raise ValueError(f"Model {model_name} not found in configuration")
                
            # Get current version from MLflow
            model_version = self.mlflow_client.get_model_version_by_alias(
                model_name,
                stored_config.alias or "champion"
            )
            
            if not model_version:
                raise ValueError(f"No model version found in MLflow for {model_name}")
                
            status = {
                "local_run_id": stored_config.run_id,
                "current_run_id": model_version.run_id,
                "model_path": str(stored_config.model_dir),
                "alias": stored_config.alias,
                "needs_update": stored_config.run_id != model_version.run_id,
                "mlflow_version": model_version.version,
                "mlflow_status": model_version.status
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status for {model_name}: {str(e)}")
            raise

    def update_model(self, model_name: str, new_alias: str) -> None:
        """Update the alias for a model.
        
        Args:
            model_name: Name of the model
            new_alias: New alias to set
            
        Raises:
            ValueError: If model not found
        """
        try:
            stored_config = self.model_config.load_model_config(model_name)
            if not stored_config.model_name:
                raise ValueError(f"Model {model_name} not found in configuration")
                
            # check if the model is already needs update
            if self.check_for_update(model_name, new_alias):
                # Update model 
                self.delete_model(model_name)
                self.add_model(model_name, alias=new_alias)
            else:
                logger.info(f"Model {model_name} is already up to date")
            
        except Exception as e:
            logger.error(f"Error updating alias for {model_name}: {str(e)}")
            raise

    def list_models(self) -> List[str]:
        """Get list of all configured models.
        
        Returns:
            List[str]: Names of all configured models
        """
        try:
            config = self.model_config.load_config()
            return list(config.keys())
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise

    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path: Path to the model directory
            
        Raises:
            ValueError: If model not found or path not set
        """
        try:
            model_path = self.model_config.get_model_path(model_name)
            if not model_path:
                raise ValueError(f"No path found for model {model_name}")
            return model_path
        except Exception as e:
            logger.error(f"Error getting path for {model_name}: {str(e)}")
            raise
