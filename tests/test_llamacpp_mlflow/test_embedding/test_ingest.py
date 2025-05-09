import pytest

import os
from dotenv import load_dotenv
import pytest
from pathlib import Path

from .test_mlflow_config import tracking_uri, mlflow_client 


def test_ingest_from_mlflow_project(mlflow_client):
   

