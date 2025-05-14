import asyncio

from dotenv import load_dotenv
from google.auth.exceptions import TransportError
from google.cloud import storage
from google.cloud.exceptions import exceptions as gcs_exceptions

import json
import os

from nicewebrl.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

GOOGLE_CREDENTIALS = "./google-cloud-key.json"


def initialize_storage_client(bucket_name: str):
  storage_client = storage.Client.from_service_account_json(GOOGLE_CREDENTIALS)

  bucket = storage_client.bucket(bucket_name)
  return bucket


def list_files(bucket):
  blobs = bucket.list_blobs()
  print("Files in bucket:")
  for blob in blobs:
    print(blob.name)


def download_files(bucket, destination_folder):
  blobs = bucket.list_blobs()
  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  for blob in blobs:
    file_path = os.path.join(destination_folder, blob.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    blob.download_to_filename(file_path)
    print(f"Downloaded {blob.name} to {file_path}")


async def save_data_to_gcs(data, blob_filename, bucket_name: str):
  try:
    bucket = initialize_storage_client(bucket_name)
    blob = bucket.blob(blob_filename)

    # Run the blocking upload in a thread pool
    await asyncio.to_thread(
      blob.upload_from_string, data=json.dumps(data), content_type="application/json"
    )

    logger.info(f"Saved {blob_filename} in bucket {bucket.name}")
    return True  # Successfully saved
  except (TransportError, gcs_exceptions.GoogleCloudError) as e:
    logger.info(f"Error saving to GCS: {e}")
  except Exception as e:
    logger.info(f"Unexpected error: {e}")
    logger.info("Skipping GCS upload")

  return False  # Failed to save


async def save_file_to_gcs(local_filename, blob_filename, bucket_name: str):
  try:
    bucket = initialize_storage_client(bucket_name)
    blob = bucket.blob(blob_filename)

    # Run the blocking upload in a thread pool
    await asyncio.to_thread(blob.upload_from_filename, local_filename)

    logger.info(f"Saved {blob_filename} in bucket {bucket.name}")
    return True  # Successfully saved
  except Exception as e:
    logger.info(f"Unexpected error: {e}")
    logger.info("Skipping GCS upload")

  return False  # Failed to save


async def save_to_gcs_with_retries(
  files_to_save, max_retries=5, retry_delay=5, bucket_name: str = ""
):
  """Save multiple files to Google Cloud Storage with retry logic.

  Args:
      files_to_save: List of filenames
      max_retries: Number of retry attempts
      retry_delay: Seconds to wait between retries

  Returns:
      bool: True if all files were saved successfully, False otherwise
  """
  assert bucket_name != "", "Bucket name is required"
  for attempt in range(max_retries):
    try:
      # Try to save all files
      for local_file in files_to_save:
        saved = await save_file_to_gcs(
          local_filename=local_file, blob_filename=local_file, bucket_name=bucket_name
        )
        if not saved:
          raise Exception(f"Failed to save {local_file}")

      logger.info(f"Successfully saved data to GCS on attempt {attempt + 1}")
      return True

    except Exception as e:
      if attempt < max_retries - 1:
        logger.info(f"Error saving to GCS: {e}. Retrying in {retry_delay} seconds...")
        await asyncio.sleep(retry_delay)
      else:
        logger.info(f"Failed to save to GCS after {max_retries} attempts: {e}")
        return False
