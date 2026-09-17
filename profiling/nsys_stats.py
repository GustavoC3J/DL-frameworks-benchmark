"""
Reads the `nsys profile --stats` summaries saved in profiling/data/*.log.

Each log holds several runs separated by a "==== <name> ====" line, and each run holds the
stats reports nsys prints (kernels, CUDA API calls, memory transfers).
"""

import re

import pandas as pd

RUN_SEPARATOR = re.compile(r"={20} (.+?) ={20}")
DATA_ROW = re.compile(r"^\s+\d+\.\d+\s")


def read_runs(path):
    """{run name: its section of the log}, in order."""
    parts = RUN_SEPARATOR.split(open(path).read())

    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts), 2)}


def report(section, name):
    """One stats report as a DataFrame, with the numbers already parsed.

    The last column keeps the name (kernel, operation...), which contains spaces.
    """
    block = section.split(f"Executing '{name}' stats report")[1]
    header = [line for line in block.splitlines() if line.strip().startswith("Time (%)") or line.strip().startswith("Total (MB)")][0]
    columns = [c.strip() for c in re.split(r"\s{2,}", header.strip())]

    rows = []
    for line in block.splitlines():
        if not DATA_ROW.match(line):
            if rows:
                break
            continue
        values = line.split()
        numbers = [float(v.replace(",", "")) for v in values[:len(columns) - 1]]
        rows.append(numbers + [" ".join(values[len(columns) - 1:])])

    return pd.DataFrame(rows, columns=columns)


def kernels(section):
    """GPU kernels of a run: name, total time in ms and number of launches."""
    table = report(section, "cuda_gpu_kern_sum")

    return pd.DataFrame({
        "kernel": table.iloc[:, -1],
        "ms": table["Total Time (ns)"] / 1e6,
        "launches": table["Instances"].astype(int),
    })


def transfers(section):
    """Memory copies of a run: direction, total time in ms and count."""
    table = report(section, "cuda_gpu_mem_time_sum")

    return pd.DataFrame({
        "operation": table.iloc[:, -1].str.replace(r"\[CUDA memcpy (.+)\]", r"\1", regex=True),
        "ms": table["Total Time (ns)"] / 1e6,
        "copies": table["Count"].astype(int),
    })


def summary(section):
    """One row per run: kernel time, launches and memory copies."""
    k, t = kernels(section), transfers(section)

    return {
        "kernel_ms": k.ms.sum(),
        "launches": int(k.launches.sum()),
        "distinct_kernels": len(k),
        "host_to_device_copies": int(t.loc[t.operation.str.startswith("Host-to-Device"), "copies"].sum()),
        "device_to_device_copies": int(t.loc[t.operation.str.startswith("Device-to-Device"), "copies"].sum()),
    }
