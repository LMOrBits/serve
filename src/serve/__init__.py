from serve.servers.llamacpp.main import llama_cpp
from serve.experiment_tracker.mlflow.mlflow_llamacpp.cli import mlflow_gcp_llamacpp

clis = [
  llama_cpp,
  mlflow_gcp_llamacpp
]