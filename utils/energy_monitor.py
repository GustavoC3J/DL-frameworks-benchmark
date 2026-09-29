from codecarbon import OfflineEmissionsTracker


class EnergyMonitor:
    """
    Measures the energy of CPU, GPU and RAM with codecarbon, one tracker per phase.
    Each phase writes its own `<phase>_emissions.csv`, with a single row.
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


    def start(self, phase):
        self._phase = phase
        self._tracker = OfflineEmissionsTracker(
            project_name=phase,
            country_iso_code=self.country_iso_code,
            gpu_ids=self.gpu_ids,
            measure_power_secs=self.interval,
            tracking_mode="machine",
            output_dir=self.output_dir,
            # codecarbon's append mode drops empty columns and misaligns the row, hence a file per phase
            output_file=f"{phase}_emissions.csv",
            save_to_file=True,
            log_level="warning",
        )

        # Hardware detection is lazy and slow, so it is forced here rather than inside the measured phase
        self._tracker.get_detected_hardware()
        self._tracker.start()


    def stop(self):
        if self._tracker is None:
            return

        tracker, self._tracker = self._tracker, None
        tracker.stop()

        # codecarbon swallows its own exceptions, so a failed measurement would go unnoticed
        if getattr(tracker, "final_emissions_data", None) is None:
            raise RuntimeError(f"codecarbon did not record the energy of the {self._phase} phase")
