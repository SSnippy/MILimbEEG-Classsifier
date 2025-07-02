import os
import requests
import zipfile
import shutil
import re

DATA_URL = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/x8psbz3f6x-2.zip'
ZIP_NAME = 'dataset.zip'
BASE_DIR = 'Datapoints'
RAW_DIR = os.path.join(BASE_DIR, 'raw')
BP_DIR = os.path.join(BASE_DIR, 'BPfilter')

# Download the zip file
print('Downloading dataset...')
with requests.get(DATA_URL, stream=True) as r:
    r.raise_for_status()
    with open(ZIP_NAME, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print('Download complete.')

# Create required directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(BP_DIR, exist_ok=True)

# Extract zip file
with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
    # List all items in the zip
    all_items = zip_ref.namelist()
    # Find the root folder inside the zip
    root_folder = os.path.commonprefix(all_items).split('/')[0]
    # Extract all items under root/S\d+/ into raw
    for item in all_items:
        match = re.match(rf'{root_folder}/S\d+/', item)
        if match:
            rel_path = os.path.relpath(item, root_folder)
            target_path = os.path.join(RAW_DIR, rel_path)
            if item.endswith('/'):
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(item) as src, open(target_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

print('Extraction complete. Directory structure is ready.') 