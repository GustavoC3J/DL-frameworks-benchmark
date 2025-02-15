import requests
import zipfile
import os

def download_and_extract(dataset_name, url):
    file_name = f"{dataset_name}.zip"
    
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
    try:
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(f"datasets/{dataset_name}")
    except zipfile.BadZipFile:
        print(f"The downloaded file for {dataset_name} is not a valid ZIP file.")
        return
    
    # Remove the zip file
    os.remove(file_name)
    
    # Completion message
    print(f"{dataset_name} downloaded and extracted successfully.")

# Dataset URLs
datasets = {
    "fashion-mnist": "https://www.kaggle.com/api/v1/datasets/download/zalando-research/fashionmnist",
    "cifar100": "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/cifar100",
    "yellow-taxi": "https://www.kaggle.com/api/v1/datasets/download/elemento/nyc-yellow-taxi-trip-data"
}

if __name__ == "__main__":
    # Download and extract each dataset
    for name, url in datasets.items():
        download_and_extract(name, url)
