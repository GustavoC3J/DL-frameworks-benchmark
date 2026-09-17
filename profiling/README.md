# Analysis of the benchmark's own behaviour

Notes and measurements about how the benchmark runs, as opposed to the results it produces.

| | |
|---|---|
| [`profiling_analysis.ipynb`](profiling_analysis.ipynb) | Where the training time goes: how much of it the GPU is actually busy, why `torch-keras` is the slowest backend, and why its LSTM stopped being an outlier after a Keras update |
| `nsys_stats.py` | Reads the `nsys profile --stats` summaries used by the notebook |
| `data/profiling_new_env.log` | `nsys` summaries for `torch-keras` and `torch`, MLP and LSTM (Keras 3.15.1 / torch 2.10) |
| `data/profiling_old_env.log` | The same `torch-keras` LSTM run under Keras 3.8 / torch 2.6 and under Keras 3.15.1 / torch 2.10 |
| `data/profiling_epoch_times.csv` | Epoch times and losses of the profiled runs |
| `data/published_epoch_times.csv` | Epoch times of the 680 successful runs behind the paper, one row per run |
| `extract_published_times.py` | Regenerates that file from `results/final` |

The notebook only needs `pandas` and `matplotlib`, and reads the files in `data/`: it does not
depend on `results/`, which is replaced whenever the experiments are re-run.
`data/published_epoch_times.csv` holds the paper's measurements as they were at commit `cb012fd`;
running `python profiling/extract_published_times.py` rewrites it from the current `results/final`.

Each log is the output of running an experiment under `nsys`, once per backend and model:

```sh
nsys profile --trace=cuda,cudnn,cublas,nvtx --stats=true \
    python experiment.py torch-keras lstm simple fp32 --gpu-ids 2 --seed 42 --epochs 3
```

`--stats=true` prints the summaries these files hold; the binary reports it also writes (hundreds
of MB) are not kept here. For the version comparison the same command ran twice, activating a conda
environment with Keras 3.8 / torch 2.6 and then the current one.
