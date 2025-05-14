"""
This script was used to download the user data from Google Cloud Storage.

Call from `examples/xland-LLM-assistant/` directory with:
python download_google_data.py
"""

import os
from google.cloud import storage
import fnmatch
import config


##############################
# User Data
##############################
def initialize_storage_client(bucket_name: str):
  storage_client = storage.Client.from_service_account_json(
    config.GOOGLE_CREDENTIALS
  )
  bucket = storage_client.bucket(bucket_name)
  return bucket


def download_user_files(
  bucket_name: str, pattern: str, destination_folder: str, prefix: str = "data/"
):
  # Create a client
  bucket = initialize_storage_client(bucket_name)

  # List all blobs in the bucket with the given prefix
  blobs = bucket.list_blobs(prefix=prefix)

  # Create the destination folder if it doesn't exist
  os.makedirs(destination_folder, exist_ok=True)

  # Download matching files
  for blob in blobs:
    if fnmatch.fnmatch(blob.name, pattern):
      destination_file = os.path.join(destination_folder, os.path.basename(blob.name))
      # Check if file already exists
      if os.path.exists(destination_file):
        print(f"File already exists: \n\t {destination_file}")
        continue
      blob.download_to_filename(destination_file)
      print(f"Downloaded: \n\t from: {blob.name} \n\t to: {destination_file}")


if __name__ == "__main__":
  import config
  from glob import glob

  bucket_name = config.BUCKET_NAME
  human_data_pattern = "*"
  download_user_files(
    bucket_name=bucket_name,
    pattern=human_data_pattern,
    destination_folder=f"{config.DATA_DIR}",
  )
  files = f"{config.DATA_DIR}/*{human_data_pattern}"
  files = list(set(glob(files)))
