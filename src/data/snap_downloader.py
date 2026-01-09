import os
import urllib.request
from pathlib import Path

def download_file(url: str, dest: Path):
    """Download a file from a URL to a destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest}")

def decompress_gz(gz_path: Path, out_path: Path):
    import gzip, shutil
    print(f"Decompressing {gz_path} to {out_path} ...")
    with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed {gz_path} to {out_path}")

def ensure_file_exists(local_path: Path, url: str):
    if not local_path.exists():
        gz_path = local_path.with_suffix(local_path.suffix + ".gz")
        download_file(url, gz_path)
        decompress_gz(gz_path, local_path)
