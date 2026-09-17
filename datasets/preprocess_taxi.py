"""
Builds the cached aggregation of the yellow taxi dataset:

    python -m datasets.preprocess_taxi

Reading and cleaning the raw CSV files takes minutes and several GB of memory, and the result is a
few thousand rows. Caching it once keeps that cost out of every experiment. The loader does the
same on its own the first time it runs, so this script is only useful to get it out of the way.
"""

from datasets.loader.dataset_loader.yellow_taxi_loader import CACHE_PATH, YellowTaxiDatasetLoader

if __name__ == "__main__":
    # The seed is unused here: the taxi split is chronological
    loader = YellowTaxiDatasetLoader(seed=0)

    for dataset_type in ("train", "test"):
        aggregated = loader.aggregate(dataset_type)
        print(f"{dataset_type}: {len(aggregated)} intervals cached in {CACHE_PATH}")
