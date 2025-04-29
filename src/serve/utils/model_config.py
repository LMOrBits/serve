from typing import Dict, Optional, Union
from pathlib import Path
import os
import json
from loguru import logger

from pydantic import BaseModel, Field, field_validator



class ModelConfigStatus(BaseModel):
    """Configuration status for a model instance.
    
    Attributes:
        run_id: Unique identifier for the model run
        model_name: Name of the model
        model_dir: Directory containing the model files
        alias: Alternative name/alias for the model
    """
    run_id: Optional[str] = Field(None, description="Unique identifier for the model run")
    model_name: Optional[str] = Field(None, description="Name of the model")
    model_dir: Optional[Union[Path, str]] = Field(None, description="Directory containing the model files")
    alias: Optional[str] = Field(None, description="Alternative name/alias for the model")

    @field_validator('model_dir')
    @classmethod
    def validate_model_dir(cls, v: Optional[Union[Path, str]]) -> Optional[Path]:
        """Validate and convert model directory to a Path object."""
        if v is None:
            return None
        return Path(v)

    def model_dump(self) -> dict:
        """Convert the model configuration to a dictionary.
        
        Returns:
            dict: Dictionary representation of the model configuration
        """
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "alias": self.alias
        }


class ModelConfig:
    """Manager for model configurations.
    
    This class handles reading, writing, and managing model configurations,
    including their storage locations and metadata.
    """
    
    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        """Initialize the model configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.is_dir():
                config_path = config_path / "config.json"
        self.config_path = config_path or Path(__file__).parents[4] / "models" / "config.json"
        self.config = self.get_config()
        logger.info(f"Initialized model configuration manager with config path: {self.config_path}")

    def get_config(self) -> Dict[str, ModelConfigStatus]:
        """Get the current configuration.
        
        Returns:
            Dict[str, ModelConfigStatus]: Dictionary mapping model names to their configurations
            
        Raises:
            OSError: If unable to create necessary directories
            JSONDecodeError: If configuration file is corrupted
        """
        try:
            self._ensure_config_exists()
            self._ensure_all_model_dir_exists()
            return self.load_config()
        except Exception as e:
            logger.error(f"Failed to get configuration: {str(e)}")
            raise
    
    def _ensure_config_exists(self) -> None:
        """Ensure the configuration file exists with default structure.
        
        Creates necessary directories and an empty configuration file if they don't exist.
        
        Raises:
            OSError: If unable to create directories or write configuration file
        """
        try:
            if not self.config_path.parent.exists():
                os.makedirs(self.config_path.parent, exist_ok=True)
                logger.info(f"Created configuration directory: {self.config_path.parent}")
            
            if not self.config_path.exists():
                self.save_config({})
                logger.info(f"Created empty configuration file: {self.config_path}")
        except OSError as e:
            logger.error(f"Failed to ensure configuration exists: {str(e)}")
            raise

    def _ensure_model_dir_exists(self, model_name: str) -> None:
        """Ensure the model directory exists for a specific model.
        
        Args:
            model_name: Name of the model
            
        Raises:
            OSError: If unable to create model directory
        """
        try:
            model_status = self.load_model_config(model_name)
            if model_status.model_dir is None:
                model_dir = self.config_path.parent / model_name
            else:
                model_dir = Path(model_status.model_dir)
                
            if not model_dir.exists():
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Created model directory: {model_dir}")
        except OSError as e:
            logger.error(f"Failed to ensure model directory exists for {model_name}: {str(e)}")
            raise

    def _ensure_all_model_dir_exists(self) -> None:
        """Ensure directories exist for all configured models.
        
        Raises:
            OSError: If unable to create any model directory
        """
        try:
            for model_status in self.load_config().values():
                if model_status.model_name:
                    self._ensure_model_dir_exists(model_status.model_name)
        except Exception as e:
            logger.error(f"Failed to ensure all model directories exist: {str(e)}")
            raise

    def load_model_config(self, model_name: str) -> ModelConfigStatus:
        """Load configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfigStatus: Configuration status for the model
            
        Raises:
            JSONDecodeError: If configuration file is corrupted
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                if model_name not in config_data:
                    logger.warning(f"No configuration found for model: {model_name}")
                    return ModelConfigStatus()
                return ModelConfigStatus(**config_data[model_name])
        except Exception as e:
            logger.error(f"Failed to load model configuration for {model_name}: {str(e)}")
            raise
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to a model's directory.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Optional[Path]: Path to the model directory if it exists, None otherwise
            
        Raises:
            JSONDecodeError: If configuration file is corrupted
        """
        try:
            model_config = self.load_model_config(model_name)
            return model_config.model_dir
        except Exception as e:
            logger.error(f"Failed to get model path for {model_name}: {str(e)}")
            raise
    
    def load_config(self) -> Dict[str, ModelConfigStatus]:
        """Load the complete configuration.
        
        Returns:
            Dict[str, ModelConfigStatus]: Dictionary mapping model names to their configurations
            
        Raises:
            JSONDecodeError: If configuration file is corrupted
        """
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            return {k: ModelConfigStatus(**v) for k, v in data.items()} if data else {}
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def save_config(self, config: Dict) -> None:
        """Save the configuration to file.
        
        Args:
            config: Configuration dictionary to save
            
        Raises:
            OSError: If unable to write configuration file
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise

    def update_config(self, model_name: str, model_status: ModelConfigStatus) -> None:
        """Update configuration for a specific model.
        
        Args:
            model_name: Name of the model
            model_status: New configuration status for the model
            
        Raises:
            JSONDecodeError: If configuration file is corrupted
            OSError: If unable to write configuration file
        """
        try:
            config = self.load_config()
            config[model_name] = model_status
            config = {k: v.model_dump() for k, v in config.items()} if config else {}
            self.save_config(config)
            logger.info(f"Updated configuration for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to update configuration for {model_name}: {str(e)}")
            raise

    def update_model_info(self,
                         run_id: str,
                         model_name: str,
                         alias: str,
                         model_dir: Optional[Path] = None) -> None:
        """Update information for a model.
        
        Args:
            run_id: Unique identifier for the model run
            model_name: Name of the model
            alias: Alternative name/alias for the model
            model_dir: Optional custom directory for the model
            
        Raises:
            ValueError: If required parameters are invalid
            OSError: If unable to create model directory
        """
        try:
            if not run_id or not model_name or not alias:
                raise ValueError("run_id, model_name, and alias are required")
                
            model_dir = model_dir or self.config_path.parent / "models" / model_name
            model_status = ModelConfigStatus(
                run_id=run_id,
                model_name=model_name,
                alias=alias,
                model_dir=model_dir
            )
            
            self._ensure_model_dir_exists(model_name)
            self.update_config(model_name, model_status)
            logger.info(f"Updated model info for {model_name} with run_id {run_id}")
        except Exception as e:
            logger.error(f"Failed to update model info: {str(e)}")
            raise