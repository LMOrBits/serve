from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from serve._cli.task import TaskCLI

class LlamaCppServerConfig(BaseModel):
    server_port: Optional[int] = 8000
    ui_port: Optional[int] = 8080
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[Path | str] = None


class LlamaCppServer:
    def __init__(
            self,
            ):
        self.cli = TaskCLI(Path(__file__).parent)
        self.configs = []

    def add_serve(self,
                  server_port: Optional[int] = 8000,
                  ui_port: Optional[int] = 8080,
                  model_name: Optional[str] = None,
                  model_id: Optional[str] = None,
                  model_path: Optional[Path | str] = None):
        """Add a serve"""
        self.serve(server_port, ui_port, model_name, model_id)
        self.configs.append(LlamaCppServerConfig(server_port=server_port, ui_port=ui_port, model_name=model_name, model_id=model_id, model_path=model_path))

    def delete_serve(self, model_id):
        """Delete a serve by model_id"""
        self.cli.run("delete", MODEL_ID=model_id)
        self.configs = [config for config in self.configs if config.model_id != model_id]


    def serve(self,
              model_path: Optional[Path | str] = None,
              server_port: Optional[int] = 8000,
              ui_port: Optional[int] = 8080,
              model_name: Optional[str] = None,
              model_id: Optional[str] = None
              ):
        """Serve a model"""
        self.cli.run("serve", 
                     MODEL_PATH=model_path,
                     SERVER_PORTS=server_port,
                     UI_PORT=ui_port,
                     MODEL_NAME=model_name,
                     MODEL_ID=model_id)
        
    def status_model_id(self, model_id):
        self.cli.run("status", MODEL_ID=model_id , ALL=False)

    def status(self, model_id):
        all_model_ids = [config.model_id for config in self.configs]
        statuses = []
        for model_id in all_model_ids:
            statuses.append(self.status_model_id(model_id))
        return statuses

    def stop(self, model_id):
        self.cli.run("stop", MODEL_ID=model_id)

    def delete(self, model_id):
        self.cli.run("delete", MODEL_ID=model_id)
