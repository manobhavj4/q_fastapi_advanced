import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class CommsConfig:
    """Configuration manager for communication settings (MQTT, S3, REST API)."""
    
    # Default configuration values
    DEFAULTS = {
        'mqtt': {
            'port': 8883,
            'client_id': 'default-client',
            'keepalive': 60,
            'qos': 1,
            'tls': {}
        },
        's3': {
            'region': 'us-east-1',
            'data_prefix': 'data/',
            'log_prefix': 'logs/'
        },
        'rest_api': {
            'enabled': False
        }
    }
    
    def __init__(self, config_path: str = "config/comms_config.yaml"):
        self.config_path = Path(config_path)
        self.mqtt: Dict[str, Any] = {}
        self.s3: Dict[str, Any] = {}
        self.rest_api: Dict[str, Any] = {}
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and parse configuration from YAML file."""
    
    
    def _parse_mqtt_config(self, mqtt_cfg: Dict[str, Any]) -> None:
        """Parse MQTT configuration with defaults."""
        defaults = self.DEFAULTS['mqtt']
        
      
    
    def _parse_tls_config(self, tls_cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Parse TLS configuration for MQTT."""
        return {
            'ca_cert': tls_cfg.get('ca_cert'),
            'certfile': tls_cfg.get('certfile'),
            'keyfile': tls_cfg.get('keyfile')
        }
    
    def _parse_s3_config(self, s3_cfg: Dict[str, Any]) -> None:
        """Parse S3 configuration with defaults."""
   
    
    def _parse_rest_api_config(self, api_cfg: Dict[str, Any]) -> None:
        """Parse REST API configuration with defaults."""
 
    
    def validate_config(self) -> None:
        """Validate required configuration values."""

    
    def get_mqtt_config(self) -> Dict[str, Any]:
        """Get MQTT configuration."""
        return self.mqtt.copy()
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration."""
        return self.s3.copy()
    
    def get_rest_api_config(self) -> Dict[str, Any]:
        """Get REST API configuration."""
        return self.rest_api.copy()
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def __str__(self) -> str:
        """String representation of configuration."""
      
    
    def __repr__(self) -> str:
        """Developer representation of configuration."""
        return f"CommsConfig(config_path='{self.config_path}')"


# Example usage and testing
if __name__ == "__main__":
    try:
        config = CommsConfig()
        config.validate_config()
        
        print("[MQTT Config]:", config.get_mqtt_config())
        print("[S3 Config]:", config.get_s3_config())
        print("[REST API Config]:", config.get_rest_api_config())
        
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Configuration error: {e}")# edge_sensor/config_loader.py

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
 