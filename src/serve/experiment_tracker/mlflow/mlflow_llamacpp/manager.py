from pathlib import Path
import os

from loguru import logger
from typing import Optional

from mlflow.tracking import MlflowClient

from serve.utils.mlflow.config import MLflowModelConfigManager,MLFlowConfig

class ModelMannager:
    def __init__(self,
                 config_path: Optional[Path | str] = None, 
                 mlflow_client: Optional[MlflowClient] = None, 
                 tracking_uri: Optional[str] = None, 
                 registry_uri: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None, 
                 ):

        self.mlflow_client = mlflow_client
        self.config_path = config_path
        
        mlflow_config = MLFlowConfig(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            username=username,
            password=password
        )
        mlflow_client = MlflowClient(
            tracking_uri=mlflow_config.tracking_uri, 
            registry_uri=mlflow_config.registry_uri
        )
        self.model_config_manager = MLflowModelConfigManager(
            mlflow_client=mlflow_client,
            config_path=config_path
        )


    def add_model(
        self,
        model_name: str,
        alias: str = "champion",
        artifact_path: str = "model_path",
        model_dir: Optional[Path] = None
    ):
        try:
            self.model_config_manager.add_model(
                model_name=model_name,
                alias=alias,
                artifact_path=artifact_path,
                model_dir=model_dir
            )
        except Exception as e:
            logger.error(f"Error adding model {model_name}: {str(e)}")
            raise e
        return

    def serve_model(self, 
                    model_name: str,
                    model_dir: Optional[Path] = None
                    ) -> Path:
        """Download model and update configuration.
        
        Args:
            model_name (str): Name of the model in registry
            alias (str): Model version alias
            artifact_path (str): Path to the model artifacts in MLflow
            model_dir (Optional[Path]): Custom directory to save the model
            
        Returns:
            Path: Path where the model is saved
        """

        return 






