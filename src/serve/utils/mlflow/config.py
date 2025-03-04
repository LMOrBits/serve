from typing import Optional

from mlflow import MlflowClient
import mlflow
from pydantic import BaseModel, model_validator
import os

from pathlib import Path
import os

from mlflow.exceptions import MlflowException
from serve.utils.model_config import ModelConfig

from loguru import logger

from mlflow.artifacts import download_artifacts

class MLFlowConfig(BaseModel):
    """MLFlow configuration."""
    tracking_uri: str
    registry_uri: Optional[str] = None
    username: str
    password: str

    @model_validator(mode='after')
    def set_registry_uri(self) -> "MLFlowConfig":
        """Set registry URI if not provided."""
        if self.registry_uri is None:
            self.registry_uri = self.tracking_uri
        return self
    
    @model_validator(mode='after')
    def set_username_password(self) -> "MLFlowConfig":
        """Set username and password if not provided."""
        if self.tracking_uri is None:
           self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        elif self.tracking_uri is None:
            os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        if self.username is None:
            self.username = os.getenv("MLFLOW_TRACKING_USERNAME")
        elif self.username is None:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
        if self.password is None:
            self.password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        elif self.password is None:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
        return self

    
    @classmethod
    def from_env(cls) -> "MLFlowConfig":
        """Initialize from environment variables."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        registry_uri = os.getenv("MLFLOW_REGISTRY_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME") 
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI environment variable must be set")
        if not username:
            raise ValueError("MLFLOW_TRACKING_USERNAME environment variable must be set")
        if not password:
            raise ValueError("MLFLOW_TRACKING_PASSWORD environment variable must be set")

        return cls(
            tracking_uri=tracking_uri,
            username=username,
            password=password,
            registry_uri=registry_uri
        )

class MLflowModelConfigManager:
    def __init__(self, mlflow_client: Optional[MlflowClient] = None, config_path: Optional[Path] = None) -> None:
        """Initialize MLflow model configuration manager.
        
        Args:
            mlflow_client (Optional[MlflowClient]): MLflow client instance
            config_path (Optional[Path]): Path to the config file
        """
        if mlflow_client is None:
            mlflow_config = MLFlowConfig.from_env()
            self.mlflow_client = MlflowClient(tracking_uri=mlflow_config.tracking_uri, registry_uri=mlflow_config.registry_uri)
        else:
            self.mlflow_client = mlflow_client

        self.model_config = ModelConfig(config_path=config_path)

    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: Path) -> None:
        """Download artifacts from MLflow.
        
        Args:
            run_id (str): Run ID of the model
            artifact_path (str): Path to the model artifacts in MLflow
        """ 
        mlflow.set_tracking_uri(self.mlflow_client.tracking_uri)
        download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=str(dst_path.resolve()))

    def add_model(self, model_name: str, alias: str = "champion", artifact_path: str = "model_path", model_dir: Optional[Path] = None) -> Path:
        """Download model and update configuration.
        
        Args:
            model_name (str): Name of the model in registry
            alias (str): Model version alias
            artifact_path (str): Path to the model artifacts in MLflow
            model_dir (Optional[Path]): Custom directory to save the model
            
        Returns:
            Path: Path where the model is saved
        """
        try:
            model_version = self.mlflow_client.get_model_version_by_alias(model_name, alias)
            if not model_version:
                raise ValueError(f"No model version found for {model_name} with alias {alias}")
                
            logger.info(f"Model version: {model_version}")
            run_id = model_version.run_id
            
            if model_dir is None:
                model_dir = self.model_config.config_path.parent / "models" / model_name

            # Download the model
            try:
                self.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=model_dir)
            except Exception as e:
                raise MlflowException(f"Failed to download artifacts: {str(e)}")
                
            # Update configuration after successful download
            self.model_config.update_model_info(
                run_id=run_id,
                model_name=model_name,
                alias=alias,
                model_dir=model_dir
            )
            
            return model_dir
            
        except MlflowException as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while adding model {model_name}: {str(e)}")
            raise

    def check_for_update(self, model_name: str, alias: str = "champion") -> bool:
        """Check if model needs updating.
        
        Args:
            model_name (str): Name of the model in registry
            alias (str): Model version alias
            
        Returns:
            bool: True if update needed, False otherwise
        """
        try:
            model_version = self.mlflow_client.get_model_version_by_alias(model_name, alias)
            if not model_version:
                raise ValueError(f"No model version found for {model_name} with alias {alias}")
                
            current_run_id = model_version.run_id
            
            stored_config = self.model_config.load_model_config(model_name)
            if not stored_config:
                return True  # No local version exists, update needed
                
            stored_run_id = stored_config.run_id
            return stored_run_id != current_run_id
            
        except MlflowException as e:
            logger.error(f"Error checking model {model_name} version: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while checking for updates: {str(e)}")
            raise
    
    def check_for_updates(self) -> dict:
        """Check for updates for all models.
        
        Returns:
            dict: Dictionary of model names and their update status
        """
        try:
            all_configs = self.model_config.load_config()
            if not all_configs:
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
            model_name (str): Name of the model to delete
        """
        try:
            stored_config = self.model_config.load_model_config(model_name)
            if not stored_config:
                raise ValueError(f"Model {model_name} not found in configuration")
                
            if stored_config.model_dir:
                model_path = Path(stored_config.model_dir)
                if model_path.exists():
                    import shutil
                    try:
                        shutil.rmtree(model_path)
                    except Exception as e:
                        logger.error(f"Failed to delete model directory: {str(e)}")
                        raise
            
            # Remove from config
            config = self.model_config.load_config()
            if model_name in config:
                del config[model_name]
                self.model_config.save_config({k: v.model_dump() for k, v in config.items()})
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            raise

    def get_model_status(self, model_name: Optional[str] = None) -> dict:
        """Get current status for one or all models.
        
        Args:
            model_name (Optional[str]): Specific model name or None for all models
            
        Returns:
            dict: Status information for requested model(s)
        """
        try:
            if model_name:
                status = self._get_single_model_status(model_name)
                if not status:
                    raise ValueError(f"Model {model_name} not found")
                return status
            
            # Get status for all models
            all_configs = self.model_config.load_config()
            if not all_configs:
                return {}
                
            statuses = {}
            for name in all_configs.keys():
                try:
                    statuses[name] = self._get_single_model_status(name)
                except Exception as e:
                    logger.warning(f"Failed to get status for {name}: {str(e)}")
                    statuses[name] = {"error": str(e)}
            return statuses
            
        except Exception as e:
            logger.error(f"Error getting model status: {str(e)}")
            raise

    def _get_single_model_status(self, model_name: str) -> dict:
        """Get status for a single model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model status information
        """
        stored_config = self.model_config.load_model_config(model_name)
        if not stored_config:
            raise ValueError(f"Model {model_name} not found in configuration")
            
        current_version = None
        needs_update = False
        try:
            current_version = self.mlflow_client.get_model_version_by_alias(
                model_name, 
                stored_config.alias or "champion"
            )
            needs_update = self.check_for_update(model_name, stored_config.alias or "champion")
        except MlflowException as e:
            logger.warning(f"Failed to get latest version for {model_name}: {str(e)}")

        return {
            "model_name": model_name,
            "current_run_id": stored_config.run_id,
            "model_dir": str(stored_config.model_dir) if stored_config.model_dir else None,
            "current_alias": stored_config.alias,
            "latest_version": current_version.version if current_version else None,
            "needs_update": needs_update,
            "model_exists": stored_config.model_dir.exists() if stored_config.model_dir else False
        }

    def update_model_alias(self, model_name: str, new_alias: str) -> None:
        """Update model alias in configuration.
        
        Args:
            model_name (str): Name of the model
            new_alias (str): New alias for the model
        """
        try:
            stored_config = self.model_config.load_model_config(model_name)
            if not stored_config:
                raise ValueError(f"Model {model_name} not found in configuration")
                
            # Verify the new alias exists in MLflow
            try:
                self.mlflow_client.get_model_version_by_alias(model_name, new_alias)
            except MlflowException:
                raise ValueError(f"Alias {new_alias} does not exist for model {model_name}")
                
            self.model_config.update_model_info(
                run_id=stored_config.run_id,
                model_name=model_name,
                alias=new_alias,
                model_dir=stored_config.model_dir
            )
        except Exception as e:
            logger.error(f"Error updating alias for model {model_name}: {str(e)}")
            raise

    def list_models(self) -> list[str]:
        """Get list of all managed models.
        
        Returns:
            list[str]: List of model names
        """
        try:
            config = self.model_config.load_config()
            return list(config.keys()) if config else []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise
