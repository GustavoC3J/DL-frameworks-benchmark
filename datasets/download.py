import tarfile
import requests
import zipfile
import os

def download_and_extract(dataset_name, url):
    file_name = f"{dataset_name}.zip" if dataset_name != "cifar10" else f"{dataset_name}.tar.gz"
    
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Error downloading {dataset_name}: {response.status_code}")
        return
    
    # Extract the file
    extract_path = f"datasets/{dataset_name}"
    os.makedirs(extract_path, exist_ok=True)

    try:
        if dataset_name == "cifar10":
            with tarfile.open(file_name, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_path)
        else:
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                
    except Exception as e:
        print(f"Error extracting {dataset_name}: {e}")
        return
    
    # Remove the file
    os.remove(file_name)
    
    # Completion message
    print(f"{dataset_name} downloaded and extracted successfully.")

# Dataset URLs
datasets = {
    "fashion-mnist": "https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist",
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "yellow-taxi": "https://www.kaggle.com/api/v1/datasets/download/elemento/nyc-yellow-taxi-trip-data"
}

if __name__ == "__main__":
    # Download and extract each dataset
    for name, url in datasets.items():
        download_and_extract(name, url)
