from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download
import os
from loguru import logger

def download_model_artifact(
    repo_id: str,
    desired_path: Path,
    model_url: Optional[str] = None,
):
    """
    Download a model from Hugging Face Hub to a specified path.
    
    Args:
        repo_id: The name of the model on Hugging Face Hub (e.g. "bert-base-uncased")
        desired_path: The path where the model should be stored
        model_url: Optional URL to download a specific model file
        
    Returns:
        Path: The path where the model was downloaded
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(desired_path, exist_ok=True)
        
        if model_url:
            # If a specific model URL is provided, download that file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_url.split('/')[-1],
                local_dir=desired_path,
                local_dir_use_symlinks=False
            )
        else:
            # Download the complete model repository
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=desired_path,
                local_dir_use_symlinks=False
            )
        paths = []
        for guff in Path(local_path).glob("**/*.gguf"):
            paths.append(guff)

        if len(paths) == 0:
            logger.warning(f"No .gguf file found in {local_path}")
        model_path = Path(local_path) / "model_path" / "artifacts"
        model_path.mkdir(parents=True, exist_ok=True)
        if len(paths) == 1:
            paths[0].rename(model_path / "model.gguf")
        elif len(paths) > 1:
            logger.warning(f"Multiple .gguf files found in {local_path}, using the first one")
            paths[0].rename(model_path / "model.gguf")
        return model_path
        
    except Exception as e:
        raise Exception(f"Failed to download model {repo_id}: {str(e)}")
    

