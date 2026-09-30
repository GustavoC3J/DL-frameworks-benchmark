import pandas as pd
from codecarbon import OfflineEmissionsTracker


class EnergyMonitor:
    """
    Measures the energy of CPU, GPU and RAM with codecarbon, one tracker per phase.
    A phase is split into consecutive codecarbon tasks (the training and validation stages of each
    epoch, or the whole test), and `<phase>_emissions.csv` gets a row per task.
    """

    def __init__(self, gpu_indices, output_dir, interval=1, country_iso_code="ESP"):
        """
        gpu_indices: comma-separated NVML indices of the GPUs to measure, as in GPUMonitor
        output_dir: directory of the CSV files
        interval: seconds between power measurements
        country_iso_code: only used to turn energy into CO2 emissions
        """
        self.gpu_ids = [int(x) for x in gpu_indices.split(",")]
        self.output_dir = output_dir
        self.interval = interval
        self.country_iso_code = country_iso_code
        self._tracker = None
        self._phase = None
        self._task = None
        self._rows = []


    def start(self, phase):
        self._phase = phase
        self._rows = []
        self._tracker = OfflineEmissionsTracker(
            project_name=phase,
            country_iso_code=self.country_iso_code,
            gpu_ids=self.gpu_ids,
            measure_power_secs=self.interval,
            tracking_mode="machine",
            # The rows are written here: codecarbon's own files would only hold the phase totals
            save_to_file=False,
            log_level="warning",
        )

        # Hardware detection and the first task set codecarbon up (~1 s), so both happen here
        # rather than inside the measured phase
        self._tracker.get_detected_hardware()
        self._tracker.start_task()
        self._tracker.stop_task()


    def begin(self, task, **labels):
        """Closes the open task and opens the next one right away, so no energy falls between them.
        The labels (the epoch, for instance) become columns of the task's row."""
        self.end()
        self._tracker.start_task()
        self._task = {"task": task, **labels}


    def end(self):
        if self._task is None:
            return

        task, self._task = self._task, None
        data = self._tracker.stop_task()

        if data is None:
            raise RuntimeError(f"codecarbon did not record the energy of the {task['task']} task of the {self._phase} phase")

        row = {**task, **data.values}

        # codecarbon averages the power over the tracker's whole life, not over the task
        for device in ("cpu", "gpu", "ram"):
            row[f"{device}_power"] = row[f"{device}_energy"] * 3.6e6 / row["duration"] # kWh -> W

        self._rows.append(row)


    def stop(self):
        if self._tracker is None:
            return

        try:
            self.end()
        finally:
            tracker, self._tracker = self._tracker, None
            tracker.stop()

        if not self._rows:
            raise RuntimeError(f"codecarbon did not record the energy of the {self._phase} phase")

        pd.DataFrame(self._rows).to_csv(f"{self.output_dir}/{self._phase}_emissions.csv", index=False)
