
"""
Edge Sensor Runtime Package - Optimized for Performance

This package integrates:
- Sensor data acquisition
- Signal processing  
- AI model inference (drift correction, anomaly detection, feature extraction)
- Communication via MQTT
- Logging of sensor and inference data

Performance optimizations:
- Lazy loading of heavy components
- Import-time optimization
- Minimal initialization overhead
- Cached module loading
"""

import sys
from typing import TYPE_CHECKING, Any
import warnings
from functools import lru_cache

# Suppress unnecessary warnings for performance
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Type checking imports - zero runtime cost
if TYPE_CHECKING:
    from .sensor_driver import SensorDriver
    from .sensor_simulator import SensorSimulator
    from .signal_processor import SignalProcessor
    from .ai_models.drift_compensation import DriftCompensator
    from .ai_models.anomaly_detector import AnomalyDetector
    from .ai_models.fft_feature_extractor import FFTFeatureExtractor
    from .ai_models.model_utils import load_model, save_model
    from .mqtt_client import MQTTClient
    from .data_logger import DataLogger
    from .config_loader import load_sensor_config
    from .comms_config_loader import load_comms_config


class LazyLoader:
    """
    High-performance lazy loader for expensive imports.
    Modules are only loaded when first accessed, reducing startup time.
    """
    
    __slots__ = ('_module_name', '_attribute', '_cached_object')
    
    def __init__(self, module_name: str, attribute: str = None):
        self._module_name = module_name
        self._attribute = attribute
        self._cached_object = None
    
    def __getattr__(self, name: str) -> Any:
        if self._cached_object is None:
            self._load_module()
        return getattr(self._cached_object, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        if self._cached_object is None:
            self._load_module()
        return self._cached_object(*args, **kwargs)
    
    def _load_module(self):
        """Load module with error handling and caching."""
        try:
            module = __import__(self._module_name, fromlist=[''])
            if self._attribute:
                self._cached_object = getattr(module, self._attribute)
            else:
                self._cached_object = module
        except ImportError as e:
            # Fallback for missing dependencies
            self._cached_object = self._create_fallback(self._module_name, self._attribute)
            import logging
            logger = logging.getLogger("EdgeSensor")
            logger.warning(f"Module {self._module_name} not available, using fallback: {e}")
    
    @staticmethod
    def _create_fallback(module_name: str, attribute: str = None):
        """Create a fallback object for missing modules."""
        class FallbackObject:
            def __init__(self, name):
                self._name = name
            
            def __call__(self, *args, **kwargs):
                raise ImportError(f"Module {module_name}.{attribute or ''} not available")
            
            def __getattr__(self, name):
                raise ImportError(f"Module {module_name}.{attribute or ''} not available")
        
        return FallbackObject(f"{module_name}.{attribute or ''}")


# Optimized lazy loading for sensor components
@lru_cache(maxsize=1)
def _get_sensor_driver():
    """Cached lazy loader for SensorDriver."""
    return LazyLoader('edge_sensor.sensor_driver', 'SensorDriver')

@lru_cache(maxsize=1) 
def _get_sensor_simulator():
    """Cached lazy loader for SensorSimulator."""
    return LazyLoader('edge_sensor.sensor_simulator', 'SensorSimulator')

@lru_cache(maxsize=1)
def _get_signal_processor():
    """Cached lazy loader for SignalProcessor."""
    return LazyLoader('edge_sensor.signal_processor', 'SignalProcessor')


# Optimized lazy loading for AI models (typically the heaviest imports)
@lru_cache(maxsize=1)
def _get_drift_compensator():
    """Cached lazy loader for DriftCompensator."""
    return LazyLoader('edge_sensor.ai_models.drift_compensation', 'DriftCompensator')

@lru_cache(maxsize=1)
def _get_anomaly_detector():
    """Cached lazy loader for AnomalyDetector."""
    return LazyLoader('edge_sensor.ai_models.anomaly_detector', 'AnomalyDetector')

@lru_cache(maxsize=1)
def _get_fft_feature_extractor():
    """Cached lazy loader for FFTFeatureExtractor."""
    return LazyLoader('edge_sensor.ai_models.fft_feature_extractor', 'FFTFeatureExtractor')

@lru_cache(maxsize=1)
def _get_model_utils():
    """Cached lazy loader for model utilities."""
    return LazyLoader('edge_sensor.ai_models.model_utils')


# Optimized lazy loading for communication and logging
@lru_cache(maxsize=1)
def _get_mqtt_client():
    """Cached lazy loader for MQTTClient."""
    return LazyLoader('edge_sensor.mqtt_client', 'MQTTClient')

@lru_cache(maxsize=1)
def _get_data_logger():
    """Cached lazy loader for DataLogger."""
    return LazyLoader('edge_sensor.data_logger', 'DataLogger')


# Optimized lazy loading for configuration
@lru_cache(maxsize=1)
def _get_config_loaders():
    """Cached lazy loader for configuration functions."""
    sensor_config = LazyLoader('edge_sensor.config_loader', 'load_sensor_config')
    comms_config = LazyLoader('edge_sensor.comms_config_loader', 'load_comms_config')
    return sensor_config, comms_config


# High-performance module attribute access
class OptimizedModuleWrapper:
    """
    Module wrapper that provides lazy loading with attribute-style access.
    Optimized for minimal overhead when accessing frequently used components.
    """
    
    def __init__(self):
        # Pre-cache lightweight loaders
        self._sensor_config_loader, self._comms_config_loader = _get_config_loaders()
    
    @property
    def SensorDriver(self):
        """Lazy-loaded SensorDriver class."""
        return _get_sensor_driver()
    
    @property
    def SensorSimulator(self):
        """Lazy-loaded SensorSimulator class."""
        return _get_sensor_simulator()
    
    @property
    def SignalProcessor(self):
        """Lazy-loaded SignalProcessor class."""
        return _get_signal_processor()
    
    @property
    def DriftCompensator(self):
        """Lazy-loaded DriftCompensator class."""
        return _get_drift_compensator()
    
    @property
    def AnomalyDetector(self):
        """Lazy-loaded AnomalyDetector class."""
        return _get_anomaly_detector()
    
    @property
    def FFTFeatureExtractor(self):
        """Lazy-loaded FFTFeatureExtractor class."""
        return _get_fft_feature_extractor()
    
    @property
    def MQTTClient(self):
        """Lazy-loaded MQTTClient class."""
        return _get_mqtt_client()
    
    @property
    def DataLogger(self):
        """Lazy-loaded DataLogger class."""
        return _get_data_logger()
    
    @property
    def load_model(self):
        """Lazy-loaded load_model function."""
        return _get_model_utils().load_model
    
    @property
    def save_model(self):
        """Lazy-loaded save_model function."""
        return _get_model_utils().save_model
    
    @property
    def load_sensor_config(self):
        """Lazy-loaded sensor config loader."""
        return self._sensor_config_loader
    
    @property
    def load_comms_config(self):
        """Lazy-loaded communications config loader."""
        return self._comms_config_loader


# Create optimized module wrapper instance
_module_wrapper = OptimizedModuleWrapper()

# Expose components through module-level attributes for backward compatibility
SensorDriver = _module_wrapper.SensorDriver
SensorSimulator = _module_wrapper.SensorSimulator
SignalProcessor = _module_wrapper.SignalProcessor
DriftCompensator = _module_wrapper.DriftCompensator
AnomalyDetector = _module_wrapper.AnomalyDetector
FFTFeatureExtractor = _module_wrapper.FFTFeatureExtractor
MQTTClient = _module_wrapper.MQTTClient
DataLogger = _module_wrapper.DataLogger
load_model = _module_wrapper.load_model
save_model = _module_wrapper.save_model
load_sensor_config = _module_wrapper.load_sensor_config
load_comms_config = _module_wrapper.load_comms_config

# Optimized __all__ definition for better import performance
__all__ = [
    "SensorDriver",
    "SensorSimulator", 
    "SignalProcessor",
    "DriftCompensator",
    "AnomalyDetector",
    "FFTFeatureExtractor",
    "load_model",
    "save_model",
    "MQTTClient",
    "DataLogger",
    "load_sensor_config",
    "load_comms_config",
]

# Performance monitoring and utilities
def get_loaded_modules():
    """
    Get information about which modules have been loaded.
    Useful for performance optimization and debugging.
    """
  

def preload_critical_modules():
    """
    Preload critical modules for applications requiring minimal latency.
    Call this during application startup for performance-critical applications.
    """
    

def preload_ai_modules():
    """
    Preload AI/ML modules for applications requiring immediate AI inference.
    These are typically the heaviest modules to load.
    """
    

def get_package_info():
    """Get package performance and status information."""
    import time


# Minimal logging setup - only if requested
def _setup_minimal_logging():
    """Setup minimal logging with performance optimizations."""
    import global_services
    

# Module initialization metadata
__version__ = "1.0.0-optimized"
__author__ = "Edge Sensor Team"
__description__ = "High-performance edge sensor runtime with lazy loading optimizations"

# Performance hint for import optimization
def __getattr__(name: str):
    """
    Fallback attribute access for any missed lazy loading.
    This ensures backward compatibility while maintaining performance.
    """



