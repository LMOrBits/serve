from loguru import logger
from google.cloud import storage
import os


def upload_artifact_to_mlflow_gcs(
    run,
    base_artifact_uri,
    artifact_subpath,
    local_file_path,
    chunk_size=10 * 1024 * 1024,
):
    """
    Uploads a file directly to the MLflow artifact location in GCS.

    Parameters:
      run: MLflow run object or run ID (string). If run object, the run_id is taken from run.info.run_id.
      base_artifact_uri: Base GCS URI for MLflow artifacts, e.g. "gs://your-bucket/mlflow-artifacts"
      artifact_subpath: The subdirectory (e.g. "model") under the run's artifact folder where the file will be stored.
      local_file_path: Local path to the file to upload.
      chunk_size: (Optional) Chunk size in bytes for uploading (default is 10 MB).

    The file will be uploaded to:
      gs://<bucket>/<artifact_root>/<run_id>/<artifact_subpath>/<filename>
    """
    # Determine the run_id from the provided run
    if isinstance(run, str):
        run_id = run
    else:
        run_id = run.info.run_id

    # Ensure the base_artifact_uri starts with "gs://"
    if not base_artifact_uri.startswith("gs://"):
        raise ValueError("base_artifact_uri must start with 'gs://'")

    # Remove "gs://" and split into bucket and artifact root.
    # Example: "gs://your-bucket/mlflow-artifacts" -> bucket: "your-bucket", artifact_root: "mlflow-artifacts"
    path_without_prefix = base_artifact_uri[5:]
    parts = path_without_prefix.split("/", 1)
    bucket_name = parts[0]
    artifact_root = parts[1] if len(parts) > 1 else ""
    artifact_root = artifact_root.rstrip("/")  # Remove trailing slash if present

    # Get the filename from the local file path
    filename = os.path.basename(local_file_path)

    # Construct the full blob (object) path.
    # If an artifact root is provided, the structure is:
    #   <artifact_root>/<run_id>/<artifact_subpath>/<filename>
    if artifact_root:
        blob_path = f"{artifact_root}/{run_id}/{artifact_subpath}/{filename}"
    else:
        blob_path = f"{run_id}/{artifact_subpath}/{filename}"

    # Initialize the GCS client and select the bucket
    client = storage.Client()  # Make sure your GCP credentials are properly set
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Set the chunk size for resumable upload (if needed)
    logger.info(f"Uploading {local_file_path} to gs://{bucket_name}/{blob_path} ...")
    blob.upload_from_filename(local_file_path)
    logger.info("Upload complete.")


def upload_large_model(bucket_name, local_file_path, destination_blob_path):
    """
    Uploads a large file directly to GCS using a specified chunk size.

    Parameters:
      bucket_name (str): Your GCS bucket name.
      local_file_path (str): Path to your local model file.
      destination_blob_path (str): Destination path in the bucket.
      chunk_size (int): Size of each upload chunk (default: 10 MB).
    """

    client = storage.Client()  # Assumes your environment has valid GCP credentials
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    logger.info(
        f"Uploading {local_file_path} to gs://{bucket_name}/{destination_blob_path} ..."
    )
    blob.upload_from_filename(local_file_path)
    logger.info("Upload complete.")
