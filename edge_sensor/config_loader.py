# edge_sensor/config_loader.py

import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import logging


@dataclass
class CalibrationSettings:
    offset: float
    scale_factor: float
    baseline_noise_threshold: float


@dataclass
class SensorConfig:
    type: str
    model: str
    sampling_rate_hz: int
    gain: float
    resolution_bits: int
    calibration: CalibrationSettings
    adc_channel: int
    integration_time_us: int
    temperature_compensation: bool


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


def _validate_config_dict(config_dict: Dict[str, Any]) -> None:
    """Validates the configuration dictionary structure."""
    

def _parse_sensor_config(config_dict: Dict[str, Any]) -> SensorConfig:
    """Parses the configuration dictionary into a SensorConfig object."""



def load_sensor_config(path: str = "config/sensor_config.yaml") -> SensorConfig:
    """
    Loads the sensor configuration from a YAML file and parses it into a dataclass.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        SensorConfig: Parsed configuration object
        
    Raises:
        ConfigValidationError: If configuration validation fails
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
   


# Optional: Add a function to load with fallback defaults
def load_sensor_config_with_defaults(
    path: str = "config/sensor_config.yaml",
    default_config: Optional[SensorConfig] = None
) -> SensorConfig:
    """
    Loads sensor configuration with fallback to default values.
    
    Args:
        path: Path to the YAML configuration file
        default_config: Default configuration to use if loading fails
        
    Returns:
        SensorConfig: Loaded or default configuration
    """
  