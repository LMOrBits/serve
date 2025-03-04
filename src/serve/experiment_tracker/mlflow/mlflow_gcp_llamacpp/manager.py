from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from serve.utils.mlflow.config import MLflowModelConfig, ModelConfig
from serve.experiment_tracker.mlflow_gcp_llamacpp.download import download_model_artifact

class ModelManager:
    """
    Manager class for handling model operations including downloading, updating, and configuration.
    """
    def __init__(
        self,
        model_name: str,
        alias: str = "champion",
        artifact_path: str = "model_path",
        gcs_bucket: str = "mlflow-artifacts-bucket",
        local_dir: Optional[Path] = None,
        gcs_credentials: Optional[Path] = None
    ):
        self.model_name = model_name
        self.alias = alias
        self.artifact_path = artifact_path
        self.gcs_bucket = gcs_bucket
        self.local_dir = local_dir
        self.gcs_credentials = gcs_credentials
        
        # Initialize MLflow model config
        self.mlflow_model_config = MLflowModelConfig(model_name)
        self.model_config = ModelConfig()

    def download_model(self, force: bool = False) -> Path:
        """
        Download the model artifacts. If force is True, download regardless of update status.
        Otherwise, only download if the model needs updating.
        
        Args:
            force (bool): Force download even if model is up to date
            
        Returns:
            Path: Local path to the downloaded model
        """
        if force or self.needs_update():
            logger.info(f"Downloading model {self.model_name}")
            return download_model_artifact(
                model_name=self.model_name,
                alias=self.alias,
                artifact_path=self.artifact_path,
                gcs_bucket=self.gcs_bucket,
                local_dir=self.local_dir,
                gcs_credentials=self.gcs_credentials
            )
        else:
            logger.info("Model is up to date")
            return self.local_dir or Path(__file__).parents[3] / "models"

    def needs_update(self) -> bool:
        """
        Check if the model needs to be updated.
        
        Returns:
            bool: True if model needs update, False otherwise
        """
        return self.mlflow_model_config.model_needs_update(self.alias)

    def get_model_path(self) -> Path:
        """
        Get the local path to the model.
        
        Returns:
            Path: Local path to the model
        """
        return self.local_dir or Path(__file__).parents[3] / "models"

    def ensure_model_available(self, force_download: bool = False) -> Path:
        """
        Ensure the model is available locally, downloading if necessary.
        
        Args:
            force_download (bool): Force download even if model exists
            
        Returns:
            Path: Path to the model directory
        """
        return self.download_model(force=force_download)

class ModelTracker:
    """
    Simple tracker for managing and monitoring multiple models.
    Keeps track of model availability and handles downloads when needed.
    """
    def __init__(self, default_gcs_bucket: str = "mlflow-artifacts-bucket"):
        self.default_gcs_bucket = default_gcs_bucket
        self.models: Dict[str, ModelManager] = {}
        self.model_config = ModelConfig()
        self._load_tracked_models()

    def _load_tracked_models(self):
        """Load all models from the model config"""
        try:
            config = self.model_config.load_config()
            for model_name in config:
                if model_name not in self.models:
                    self.models[model_name] = ModelManager(
                        model_name=model_name,
                        gcs_bucket=self.default_gcs_bucket
                    )
        except Exception as e:
            logger.error(f"Error loading models from config: {e}")
    
    def add_model(self, model_name: str, alias: str = "champion", artifact_path: str = "model_path", gcs_bucket: str = "mlflow-artifacts-bucket", local_dir: Optional[Path] = None, gcs_credentials: Optional[Path] = None):
        """Add a new model to the tracker"""
        self.models[model_name] = ModelManager(
            model_name=model_name,
            alias=alias,
            artifact_path=artifact_path,
            gcs_bucket=gcs_bucket,
            local_dir=local_dir,
            gcs_credentials=gcs_credentials
        )

    def get_model_status(self) -> Dict[str, dict]:
        """
        Get the status of all tracked models.
        Returns a dictionary with model names as keys and their status information.
        """
        status = {}
        for model_name, manager in self.models.items():
            model_path = manager.get_model_path()
            status[model_name] = {
                "available": model_path.exists(),
                "needs_update": manager.needs_update(),
                "path": str(model_path)
            }
        return status

    def ensure_model(self, model_name: str, download_if_missing: bool = True) -> ModelManager:
        """
        Get a model manager, creating and downloading if necessary.
        """
        if model_name not in self.models:
            logger.info(f"Creating new model manager for {model_name}")
            self.models[model_name] = ModelManager(
                model_name=model_name,
                gcs_bucket=self.default_gcs_bucket
            )

        manager = self.models[model_name]
        
        if download_if_missing and manager.needs_update():
            logger.info(f"Downloading missing or outdated model: {model_name}")
            manager.ensure_model_available()

        return manager

    def update_all_models(self) -> Dict[str, bool]:
        """
        Update all tracked models that need updates.
        Returns a dictionary with model names and whether they were updated successfully.
        """
        results = {}
        for model_name, manager in self.models.items():
            try:
                if manager.needs_update():
                    manager.ensure_model_available()
                    results[model_name] = True
                else:
                    results[model_name] = "already_up_to_date"
            except Exception as e:
                logger.error(f"Error updating model {model_name}: {e}")
                results[model_name] = False
        return results

    def list_models(self) -> List[str]:
        """List all tracked model names."""
        return list(self.models.keys())
