import os
import mlflow
from pathlib import Path
from configparser import ConfigParser


def config_init():
    config_mlflow = ConfigParser()
    dir_path = Path(__file__).parents[3] / 'secrets' / 'my_auth_config.ini'
    print("---------------\n",dir_path)
    config_mlflow.read(dir_path)
    os.environ['MLFLOW_TRACKING_USERNAME'] = config_mlflow['mlflow']['TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config_mlflow['mlflow']['TRACKING_PASSWORD']
    return config_mlflow

