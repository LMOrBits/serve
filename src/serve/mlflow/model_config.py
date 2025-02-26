import json
from pathlib import Path
import os

class ModelConfig:
    def __init__(self):
        self.config_path = Path(__file__).parents[3] / "models" / "config.json"
        self._ensure_config_exists()

    def _ensure_config_exists(self):
        """Ensure the config file exists with default structure"""
        if not self.config_path.parent.exists():
            os.makedirs(self.config_path.parent, exist_ok=True)
        
        if not self.config_path.exists():
            default_config = {
                "current_model": {
                    "run_id": None,
                    "model_name": None,
                    "alias": None
                }
            }
            self.save_config(default_config)

    def load_config(self):
        """Load the current configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_config(self, config):
        """Save the configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def update_model_info(self, run_id: str, model_name: str, alias: str):
        """Update the current model information"""
        config = self.load_config()
        config["current_model"] = {
            "run_id": run_id,
            "model_name": model_name,
            "alias": alias
        }
        self.save_config(config) 