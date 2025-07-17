import numpy as np
import time
import json
import random
from typing import List, Dict, Union, Optional

class SensorSimulator:
    """
    Simulates quantum sensor output for testing edge pipeline.
    Generates time-series data with configurable noise, drift, and sampling rate.
    """

    def __init__(
        self,
        sensor_name: str = "QuantumSensorSim",
        sampling_rate_hz: int = 100,
        signal_freq_hz: float = 5.0,
        amplitude: float = 1.0,
        noise_std: float = 0.05,
        drift_per_sec: float = 0.001,
        seed: Optional[int] = None
    ):
        self.sensor_name = sensor_name
        self.sampling_rate_hz = sampling_rate_hz
        self.signal_freq_hz = signal_freq_hz
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.drift_per_sec = drift_per_sec
        self.drift_value = 0.0
        
        # Pre-compute constants for optimization
        self.two_pi_freq = 2 * np.pi * self.signal_freq_hz
        self.drift_increment = self.drift_per_sec / self.sampling_rate_hz
        self.interval = 1.0 / self.sampling_rate_hz
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_sample(self, t: float) -> Dict[str, Union[float, str]]:
        """Generates a single time-stamped sensor sample."""
    

    def stream_data(self, duration_sec: int = 10, as_json: bool = False) -> None:
        """
        Streams data in real time for a given duration.

        :param duration_sec: Time to simulate in seconds
        :param as_json: Print output as JSON line
        """
        

    def generate_batch(self, num_samples: int = 1000) -> List[Dict[str, Union[float, str]]]:
        """
        Generate a batch of data samples (not in real time).
        Optimized for faster batch generation.

        :param num_samples: Number of data points to simulate
        :return: List of sample dictionaries
        """
       
      

    def export_to_json(self, file_path: str = "dummy_data.json", num_samples: int = 1000) -> None:
        """
        Generates and saves sensor data to a JSON file.

        :param file_path: Output path
        :param num_samples: Number of samples
        """
 

    def reset_drift(self) -> None:
        """Reset the drift value to zero."""
      

# ------------------ TEST BLOCK ------------------ #
if __name__ == "__main__":
    sim = SensorSimulator(
        sampling_rate_hz=10,
        signal_freq_hz=1.0,
        amplitude=1.5,
        noise_std=0.1,
        drift_per_sec=0.01,
        seed=42
    )

    # Stream to terminal
    sim.stream_data(duration_sec=5, as_json=True)

    # Export batch
    sim.export_to_json("tests/dummy_data.json", num_samples=500)