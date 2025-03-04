
from typing import Optional
from pydantic import BaseModel
import os

import json
from pathlib import Path
import os

class ModelConfigStatus(BaseModel):
    run_id: Optional[str] = None
    model_name: Optional[str] = None
    model_dir: Optional[Path | str] = None
    alias: Optional[str] = None

    def model_post_init(self, __context) -> None:
        if isinstance(self.model_dir, str):
            self.model_dir = Path(self.model_dir)

    def model_dump(self):
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_dir": str(self.model_dir),
            "alias": self.alias
        }


class ModelConfig:
    def __init__(self,  config_path: Optional[Path] = None):
        if config_path is not None:
            if isinstance(config_path, str):
                config_path = Path(config_path)
            if config_path.is_dir():
                config_path = config_path / "config.json"
        self.config_path = config_path or Path(__file__).parents[4] / "models" / "config.json"
        self.config = self.get_config()

    def get_config(self):
        self._ensure_config_exists()
        self._ensure_all_model_dir_exists()
        return self.load_config()
    
    def _ensure_config_exists(self):
        """Ensure the config file exists with default structure"""
        if not self.config_path.parent.exists():
            os.makedirs(self.config_path.parent, exist_ok=True)
        
        if not self.config_path.exists():
            self.save_config({})

    def _ensure_model_dir_exists(self, model_name:str):
        """Ensure the model directory exists"""
        model_status = self.load_model_config(model_name)
        if model_status.model_dir is None or not Path(model_status.model_dir).exists():
            os.makedirs(self.config_path.parent / model_name, exist_ok=True)

    def _ensure_all_model_dir_exists(self):
        """Ensure the model directory exists"""
        for model_status in self.load_config().values():
            self._ensure_model_dir_exists(model_status.model_name)

    def load_model_config(self, model_name:str) -> ModelConfigStatus:
        """Load the current configuration"""
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
            if model_name not in config_data:
                return ModelConfigStatus()
            return ModelConfigStatus(**config_data[model_name])

    def load_config(self):
        """Load the current configuration"""
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        return {k: ModelConfigStatus(**v) for k, v in data.items()} if data else {}

    def save_config(self, config):
        """Save the configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def update_config(self, model_name:str, model_status:ModelConfigStatus):
        """Add a new model to the configuration"""
        config = self.load_config()
        config[model_name] = model_status
        config = {k: v.model_dump() for k, v in config.items()} if config else {}
        self.save_config(config)

    def update_model_info(self, run_id: str, model_name: str, alias: str , model_dir: Optional[Path] = None):
        """Update the current model information"""
        model_dir = model_dir or self.config_path.parent / "models" / model_name
        model_status = ModelConfigStatus(run_id=run_id, model_name=model_name, alias=alias, model_dir=model_dir)
        self._ensure_model_dir_exists(model_name)
        self.update_config(model_name, model_status)