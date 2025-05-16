from pathlib import Path
from google.cloud import storage
import os
from tqdm import tqdm

from loguru import logger
from typing import Optional, List, Union
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

class DownloadError(Exception):
    """Custom exception for download-related errors."""
    pass

class TqdmWriter:
    """Custom file writer with progress bar for downloads.
    
    This class wraps a file object with tqdm progress bar functionality
    to show download progress.
    
    Attributes:
        file_obj: The file object to write to
        pbar: tqdm progress bar instance
    """
    
    def __init__(self, file_obj, total_bytes: int):
        """Initialize the writer with a file object and total size.
        
        Args:
            file_obj: File object to write to
            total_bytes: Total size of the file in bytes
        """
        self.file_obj = file_obj
        self.pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="Downloading",
            miniters=1
        )
        
    def write(self, data: bytes) -> int:
        """Write data to file and update progress bar.
        
        Args:
            data: Bytes to write
            
        Returns:
            int: Number of bytes written
        """
        bytes_written = self.file_obj.write(data)
        self.pbar.update(len(data))
        return bytes_written
        
    def flush(self) -> None:
        """Flush the file buffer."""
        self.file_obj.flush()
        
    def close(self) -> None:
        """Close the progress bar."""
        self.pbar.close()

def download_from_gcs(
    gcs_bucket: str,
    source_path: str,
    destination_path: Union[str, Path],
    credentials: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Download files from Google Cloud Storage.
    
    This function can handle both single files and directories. For directories,
    it preserves the directory structure when downloading.
    
    Args:
        gcs_bucket: Name of the GCS bucket
        source_path: Path in GCS to download from
        destination_path: Local path to save files to
        credentials: Path to Google Cloud credentials file
            If None, uses GOOGLE_APPLICATION_CREDENTIALS environment variable
            
    Returns:
        List[Path]: List of paths to downloaded files
        
    Raises:
        DownloadError: If download fails
        ValueError: If invalid parameters provided
        DefaultCredentialsError: If credentials not found/invalid
    """
    try:
        # Validate inputs
        if not gcs_bucket:
            raise ValueError("GCS bucket name cannot be empty")
        if not source_path:
            raise ValueError("Source path cannot be empty")
            
        # Convert paths to proper types
        destination_path = Path(destination_path)
        print('destination_path', destination_path)
        if credentials:
            credentials = str(Path(credentials).resolve())
        else:
            credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials:
                raise ValueError(
                    "No credentials provided and GOOGLE_APPLICATION_CREDENTIALS "
                    "environment variable not set"
                )
        credentials = service_account.Credentials.from_service_account_file(
            credentials
        ) 
        # Initialize GCS client and get bucket
        try:
            storage_client = storage.Client(credentials=credentials)
            bucket: Bucket = storage_client.bucket(gcs_bucket)
        except DefaultCredentialsError as e:
            raise DefaultCredentialsError(
                f"Failed to initialize GCS client with credentials: {str(e)}"
            )
        except Exception as e:
            raise DownloadError(f"Failed to access GCS bucket: {str(e)}")
            
        downloaded_files: List[Path] = []
        
        # List all blobs with the given prefix
        try:
            blobs: List[Blob] = list(bucket.list_blobs(prefix=source_path))
            if not blobs:
                raise DownloadError(
                    f"No files found at gs://{gcs_bucket}/{source_path}"
                )
                
            logger.info(
                f"Found {len(blobs)} files to download from "
                f"gs://{gcs_bucket}/{source_path}"
            )
            
            # Download each blob
            for blob in blobs:
                if blob.name == source_path or blob.name.startswith(source_path + '/'):
                    # Calculate relative path from source_path
                    rel_path = blob.name[len(source_path):].lstrip('/')
                    if not rel_path:  # Skip directory itself
                        continue
                        
                    local_path = destination_path / rel_path
                    
                    # Create parent directories
                    try:
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        raise DownloadError(
                            f"Failed to create directory {local_path.parent}: {str(e)}"
                        )
                    
                    # Download file with progress bar
                    logger.info(f"Downloading {blob.name} to {local_path}")
                    try:
                        with open(local_path, "wb") as file_obj:
                            writer = TqdmWriter(file_obj, blob.size)
                            blob.download_to_file(writer)
                            writer.close()
                        downloaded_files.append(local_path)
                    except (OSError, GoogleAPIError) as e:
                        raise DownloadError(
                            f"Failed to download {blob.name}: {str(e)}"
                        )
                        
            if not downloaded_files:
                raise DownloadError(
                    f"No files were downloaded from gs://{gcs_bucket}/{source_path}"
                )
                
            logger.info(
                f"Successfully downloaded {len(downloaded_files)} files to "
                f"{destination_path}"
            )
            return downloaded_files
            
        except GoogleAPIError as e:
            raise DownloadError(f"GCS API error: {str(e)}")
            
    except Exception as e:
        if not isinstance(e, (DownloadError, ValueError, DefaultCredentialsError)):
            raise DownloadError(f"Unexpected error during download: {str(e)}")
        raise



    

        
   
