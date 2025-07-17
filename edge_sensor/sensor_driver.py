# sensor_driver.py
import time
import random
import logging
import math
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading

# Hardware interface imports with graceful fallback
try:
    import smbus2
    import RPi.GPIO as GPIO
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# Configure logger



@dataclass
class SensorConfig:
    """Configuration for sensor driver with sensible defaults."""



class SensorDriver:
    """Optimized sensor driver with caching, threading safety, and better error handling."""
    
    def __init__(self, config: Optional[Union[Dict, SensorConfig]] = None):
        # Handle both dict and SensorConfig inputs


    def _init_hardware(self) -> None:
        """Initialize hardware interfaces with proper error handling."""
      

    def _read_adc_raw(self, channel: int = None) -> float:
        """Read raw ADC value with optimized simulation."""


    def _apply_calibration(self, reading: float, sensor_id: str) -> float:
        """Apply calibration offset to raw reading."""
      

    def _should_read_new_value(self) -> bool:
        """Check if we need to read a new value based on caching and timing."""
      

    @contextmanager
    def _rate_limit(self):
        """Context manager for rate limiting sensor reads."""
   

    def read_sensor(self, sensor_id: str = "sensor_1", force_read: bool = False) -> float:
        """
        Read sensor value with caching and rate limiting.
        
        Args:
            sensor_id: Identifier for the sensor
            force_read: Bypass caching and force a new reading
            
        Returns:
            Calibrated sensor value
        """
      

    def read_multiple_sensors(self, sensor_ids: list) -> Dict[str, float]:
        """Read multiple sensors efficiently."""
    

    def update_calibration(self, sensor_id: str, offset: float) -> None:
        """Update calibration offset for a sensor."""
   

    def set_sampling_rate(self, rate: float) -> None:
        """Update sampling rate and recalculate interval."""
    

    def get_status(self) -> Dict:
        """Get current driver status."""
  

    def cleanup(self) -> None:
        """Clean up hardware resources."""
     

    def __enter__(self):
        """Context manager entry."""
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
       


# Example usage and testing
if __name__ == "__main__":
    # Example with custom configuration
    config = SensorConfig(
        sampling_rate=5.0,
        gain=2.0,
        enable_caching=True,
        max_reading_age=0.2
    )
    
    # Using context manager for automatic cleanup
  