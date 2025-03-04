import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
from google.cloud import storage
import serve.mlflow.config as config
import os
from tqdm import tqdm
from serve.mlflow.model_config import ModelConfig
from serve.mlflow.check import model_needs_update
from loguru import logger
from typing import Optional

class TqdmWriter:
    def __init__(self, file_obj, total):
        self.file_obj = file_obj
        self.pbar = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
    def write(self, data):
        self.file_obj.write(data)
        self.pbar.update(len(data))
    def flush(self):
        self.file_obj.flush()
    def close(self):
        self.pbar.close()

def download_from_gcs(
    gcs_bucket: str,
    source_path: str,
    destination_path: Path,
    credentials: Optional[Path] = None
) -> list[Path]:
    """
    Download files from Google Cloud Storage. Can handle both single files and directories.
    
    Args:
        gcs_bucket (str): Name of the GCS bucket
        source_path (str): Path in GCS to download from
        destination_path (Path): Local path to save files to
        credentials: Optional pre-configured storage client
    
    Returns:
        list[Path]: List of paths to downloaded files
    """
    if credentials is None:
        credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    bucket = storage.Client(credentials=credentials).bucket(gcs_bucket)
    downloaded_files = []
    
    # List all blobs with the given prefix
    blobs = bucket.list_blobs(prefix=source_path)
    
    for blob in blobs:
        if blob.name == source_path or blob.name.startswith(source_path + '/'):
            # Calculate relative path from source_path
            rel_path = blob.name[len(source_path):].lstrip('/')
            if not rel_path:  # This is the directory itself
                continue
                
            local_path = destination_path / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {blob.name} to {local_path}")
            with open(local_path, "wb") as file_obj:
                writer = TqdmWriter(file_obj, blob.size)
                blob.download_to_file(writer)
                writer.close()
            
            downloaded_files.append(local_path)
    
    return downloaded_files



    

        
   
