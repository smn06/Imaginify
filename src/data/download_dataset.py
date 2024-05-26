import urllib.request
import tarfile
import os

# URL to download the dataset (example)
dataset_url = "https://example.com/dataset.tar.gz"
save_path = "./data/raw/"
os.makedirs(save_path, exist_ok=True)

# Download dataset
urllib.request.urlretrieve(dataset_url, os.path.join(save_path, "dataset.tar.gz"))

# Extract dataset (if in compressed format)
with tarfile.open(os.path.join(save_path, "dataset.tar.gz"), "r:gz") as tar:
    tar.extractall(path=save_path)
