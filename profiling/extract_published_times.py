"""
Summarizes the epoch times of results/final into profiling/data/published_epoch_times.csv, so the
analysis does not depend on that folder, which is replaced whenever the experiments are re-run.

    python profiling/extract_published_times.py

One row per successful run. The first epoch is reported apart: it carries JIT compilation and
warm-up, so the median excludes it.
"""

import glob
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(PROJECT_ROOT, "profiling", "data", "published_epoch_times.csv")


def collect():
    rows = []

    for directory in sorted(glob.glob(os.path.join(PROJECT_ROOT, "results", "final", "*", "*"))):
        history = os.path.join(directory, "train.csv")
        if not os.path.exists(history) or os.path.exists(os.path.join(directory, "error.txt")):
            continue

        _, backend, model, complexity, rest = os.path.basename(directory).split("_", 4)
        precision, seed = rest.rsplit("_", 1)
        epochs = pd.read_csv(history)["epoch_time"]

        rows.append({
            "backend": backend,
            "model": model,
            "complexity": complexity,
            "precision": precision,
            "seed": int(seed),
            "epochs": len(epochs),
            "first_epoch_time": round(epochs.iloc[0], 4),
            "median_epoch_time": round(epochs.iloc[1:].median(), 4),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = collect()
    table.to_csv(OUTPUT, index=False)
    print(f"{len(table)} runs written to {OUTPUT}")
