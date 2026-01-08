import os
import urllib.request

DATA_DIR = "data/raw"
FILE_NAME = "the-verdict.txt"


def download_text(url: str):
    """
    Downloads text file from given URL into data/raw/.
    Skips download if file already exists.
    Returns local file path.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, FILE_NAME)

    if not os.path.exists(file_path):
        print("⬇️ Downloading dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("✅ Download complete.")
    else:
        print("📁 Dataset already exists. Skipping download.")

    return file_path
