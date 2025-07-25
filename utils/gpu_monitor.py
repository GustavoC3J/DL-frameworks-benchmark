
import threading
import time
import csv

from pynvml import *

class GPUMonitor:
    """
    Monitors GPU metrics (utilization, memory usage, power consumption) and saves them to a CSV file.
    Runs in a separate thread and can be started/stopped.
    Uses NVIDIA's NVML library to access GPU metrics.
    """

    def __init__(self, gpu_indices=None, interval=1):
        """
        gpu_indices: GPU index list to monitor (None = all)
        interval: seconds between samples
        file_path: CSV file path to save metrics
        """
        # Parse to int for comparison with nvidia-smi output
        self.gpu_indices = [int(x) for x in gpu_indices] if gpu_indices else None
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self._handles = None


    def _init_nvml(self):
        nvmlInit()

        indices = list(range(nvmlDeviceGetCount())) if self.gpu_indices is None else self.gpu_indices

        self._handles = [nvmlDeviceGetHandleByIndex(i) for i in indices]
        self._monitor_indices = indices

    
    def _get_gpu_metrics(self):
        metrics = {}
        for idx, handle in zip(self._monitor_indices, self._handles):
            try:
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                power = nvmlDeviceGetPowerUsage(handle) / 1000  # mW -> W
                metrics[f"gpu_{idx}_utilization"] = util.gpu
                metrics[f"gpu_{idx}_memory_used"] = mem.used / 1024**2  # bytes -> MiB
                metrics[f"gpu_{idx}_power"] = power
            except NVMLError as e:
                raise NVMLError(f"Error reading GPU {idx}: {e}")
        return metrics

    def _monitor_loop(self):
        self._init_nvml()

        # Prepare column names
        fieldnames = ['timestamp']
        for idx in self._monitor_indices:
            fieldnames += [
                f'gpu_{idx}_utilization',
                f'gpu_{idx}_memory_used',
                f'gpu_{idx}_power'
            ]

        with open(self.file_path, 'w', newline='\n') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while not self._stop_event.is_set():
                # Take current timestamp
                t0 = time.time()
                now = t0 - self._start_time
                row = {'timestamp': now}

                # Add metrics and write to file
                row.update(self._get_gpu_metrics())
                writer.writerow(row)
                f.flush()

                # Wait for the next sample
                elapsed = time.time() - t0
                to_sleep = max(0, self.interval - elapsed)
                time.sleep(to_sleep)

        nvmlShutdown()


    def start(self, file_path, start_time=None):
        self._stop_event.clear()
        self.file_path = file_path
        self._start_time = start_time if start_time else time.time()

        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()


    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()


    def get_total_memory(self):
        """
        Returns a dictionary with the total memory (in MiB) of each monitored GPU.
        """

        self._init_nvml()

        total_mem = {}
        for idx, handle in zip(self._monitor_indices, self._handles):
            try:
                mem = nvmlDeviceGetMemoryInfo(handle)
                total_mem[idx] = mem.total / 1024**2  # bytes -> MiB
            except NVMLError as e:
                raise NVMLError(f"Error reading GPU-{idx}'s total memory: {e}")

        nvmlShutdown()

        return total_mem
