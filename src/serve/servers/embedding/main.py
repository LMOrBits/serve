from pathlib import Path
import shutil
from typing import Optional, List, Union, Dict
from loguru import logger

from pydantic import BaseModel, field_validator, FieldValidationInfo
from serve._cli.task import TaskCLI
from mlflow import MlflowClient
import json
from mlflow.artifacts import download_artifacts
from serve.utils.mlflow.model import get_model, get_model_run_id

class EmbeddingConfig(BaseModel):
    model_name: str
    alias: str
    model_path: Path
    run_id: str

    def model_dump(self):
        return {
            "model_name": self.model_name,
            "alias": self.alias,
            "model_path": str(self.model_path),
            "run_id": self.run_id
        }
class EmbeddingManager():

    def __init__(self , desrie_path: Path, mlflow_client: MlflowClient):
        Path(desrie_path).mkdir(parents=True, exist_ok=True)
        configs_dir = Path(desrie_path) / "embedding_configs.json"
        self.condir = configs_dir
        if not configs_dir.exists():
            configs = {}
            with open(configs_dir, "w") as f:
                json.dump(configs, f)
        self.configs = self.get_configs()
        self.desrie_path = Path(desrie_path).resolve().absolute()
        self.mlflow_client = mlflow_client
        self.task_cli = TaskCLI(Path(__file__).parent)
        self.artifact_path = "serve"

    def get_configs(self):
        with open(self.condir, "r") as f:
            configs = json.load(f) 
        return configs
    
    def config_update(self, config: EmbeddingConfig):
        self.configs = self.get_configs()
        self.configs[config.model_name] = config.model_dump()
        with open(self.condir, "w") as f:
            json.dump(self.configs, f, indent=4)

    def run_serve(self, model_name: str, docker: bool = True):
        if model_name in self.configs:
            if docker:
                self.task_cli.run("serve", model_name=model_name , model_path=self.configs[model_name]["model_path"])
            else:
                self.task_cli.run("local", model_name=model_name , model_path=self.configs[model_name]["model_path"])
                
        else:
            raise ValueError(f"Model {model_name} not found")

    def add_serve(self, model_name: str, alias: str, force: bool = False, docker: bool = True):
        if model_name in self.configs and not force:
            self.run_serve(model_name, docker)

        else: 
            if force:
                logger.info(f"Deleting old model {model_name} from {self.desrie_path}")
                model_path = self.desrie_path / model_name
                if model_path.exists():
                    shutil.rmtree(model_path)
            
            logger.info(f"Downloading model {model_name} from mlflow")
            model_path , run_id = get_model(self.mlflow_client, model_name, alias, self.desrie_path, self.artifact_path)
            logger.info(f"Model {model_name} downloaded to {model_path}")
            self.config_update(EmbeddingConfig(model_name=model_name, alias=alias, model_path=model_path/"serve", run_id=run_id))
            logger.info(f"Running model {model_name} with alias {alias}")
            self.run_serve(model_name, docker)
    
    def new_model_status(self, model_name: str, alias: str):
        try:
            if model_name in self.configs:
                run_id = get_model_run_id(self.mlflow_client, model_name, alias)
                if run_id != self.configs[model_name]["run_id"]:
                    return True
                else:
                    return False
            else:
                return True
        except Exception as e:
            logger.warning(f"Error checking model {model_name} with alias {alias}: {e}")
            return True
    
    def update_model(self, model_name: str, alias: str,  docker: bool = True):
        if self.new_model_status(model_name, alias):
            logger.info(f"Updating model {model_name} with alias {alias}")
            self.delete_serve(model_name)
            self.add_serve(model_name, alias, force=True, docker=docker)
        else:
            self.run_serve(model_name, docker)

    def stop_serve(self, model_name: str):
        self.task_cli.run("stop", model_name=model_name)

    def delete_serve(self, model_name: str):
        self.task_cli.run("delete", model_name=model_name)
    
    def delete_all_serve(self):
        for model_name in self.configs:
            self.delete_serve(model_name)
    
    def stop_all_serve(self):
        for model_name in self.configs:
            self.stop_serve(model_name)
        
        
