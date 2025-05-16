import os
from pathlib import Path
import shutil
from typing import Optional, List, Union, Dict
from loguru import logger

from pydantic import BaseModel 
from serve._cli.task import TaskCLI
from mlflow import MlflowClient
import json
from serve.utils.mlflow.model import get_model, get_model_run_id



class LlamaCppConfig(BaseModel):
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
class LlamaCppServer():

    def __init__(self , desrie_path: Path, mlflow_client: MlflowClient , gcp: bool = False):
        Path(desrie_path).mkdir(parents=True, exist_ok=True)
        configs_dir = Path(desrie_path) / "lm_configs.json"
        self.condir = configs_dir
        if not configs_dir.exists():
            configs = {}
            with open(configs_dir, "w") as f:
                json.dump(configs, f)
        self.configs = self.get_configs()
        self.desrie_path = Path(desrie_path).resolve().absolute()
        self.mlflow_client = mlflow_client
        self.task_cli = TaskCLI(Path(__file__).parent)
        self.artifact_path = "model_path"

    def get_configs(self):
        with open(self.condir, "r") as f:
            configs = json.load(f) 
        return configs
    
    def config_update(self, config: LlamaCppConfig):
        self.configs = self.get_configs()
        self.configs[config.model_name] = config.model_dump()
        with open(self.condir, "w") as f:
            json.dump(self.configs, f, indent=4)

    def run_serve(self, model_name: str, port: int = 8080):
        if model_name in self.configs:
            self.task_cli.run("serve", model_id=model_name , model_path=self.desrie_path / self.configs[model_name]["model_path"], port=port)
        else:
            raise ValueError(f"Model {model_name} not found")

    def add_serve(self, model_name: str, alias: str, force: bool = False ,port: int = 8080):
        if model_name in self.configs and not force:
            self.run_serve(model_name , port)

        else: 
            if force:
                logger.info(f"Deleting old model {model_name} from {self.desrie_path}")
                model_path = self.desrie_path / model_name
                if model_path.exists():
                    shutil.rmtree(model_path)
            
            logger.info(f"Downloading model {model_name} from mlflow")
            credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            mlflow_gcp = os.getenv("MLFLOW_GCS_BUCKET")
            gcp = False
            if mlflow_gcp is not None or credentials is not None:
                if mlflow_gcp is None:
                    logger.info("No MLFLOW_GCS_BUCKET environment variable found, using default model path")
                if credentials is None:
                    logger.info("No GOOGLE_APPLICATION_CREDENTIALS environment variable found, using default model path")
                else:
                    logger.info("Using GCP credentials and GCS bucket")
                    gcp = True
            model_path , run_id = get_model(self.mlflow_client, model_name, alias, self.desrie_path, self.artifact_path, gcp)
            logger.info(f"Model {model_name} downloaded to {model_path}")
            self.config_update(LlamaCppConfig(model_name=model_name, alias=alias, model_path=model_path/"model_path"/"artifacts", run_id=run_id))
            logger.info(f"Running model {model_name} with alias {alias}")
            self.run_serve(model_name , port)
    
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
    
    def update_model(self, model_name: str, alias: str , port: int = 8080):
        if self.new_model_status(model_name, alias):
            logger.info(f"Updating model {model_name} with alias {alias}")
            self.delete_serve(model_name)
            self.add_serve(model_name, alias, force=True, port=port)
        else:
            self.run_serve(model_name)

    def stop_serve(self, model_name: str):
        self.task_cli.run("stop", model_id=model_name)

    def delete_serve(self, model_name: str):
        self.task_cli.run("delete", model_id=model_name)
    
    def delete_all_serve(self):
        for model_name in self.configs:
            self.delete_serve(model_name)
    
    def stop_all_serve(self):
        for model_name in self.configs:
            self.stop_serve(model_name)

   
