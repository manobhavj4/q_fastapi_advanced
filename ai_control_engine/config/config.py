import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from functools import lru_cache
from pathlib import Path
import threading

logger = logging.getLogger("AIEngineConfig")

"""
import os
AI_ENGINE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'ai_engine_config.yaml')

"""


class OptimizedConfigLoader:
    """
    High-performance configuration loader with caching and optimized access patterns.
    
    Optimizations:
    - LRU caching for repeated config access
    - Single YAML parse with intelligent caching
    - Fast path resolution with pathlib
    - Thread-safe operations
    - Minimal memory allocations
    - Pre-computed access patterns
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Load and validate the AI Engine configuration with performance optimizations.
        
        Args:
            config_path: Optional path to YAML config. Defaults to ai_engine_config.yaml 
                        in config/ directory.
        """
        # Use pathlib for faster path operations
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent / 'ai_engine_config.yaml'
        
        # Thread safety for concurrent access
        self._config_lock = threading.RLock()
        
        # Load and cache configuration once
        self.config = self._load_config()
        
        # Pre-cache commonly accessed sections for O(1) access
        self._thresholds_cache = self.config.get('thresholds', {})
        self._models_cache = self.config.get('models', {})
        self._flags_cache = self.config.get('flags', {})
        
        # Pre-validate critical sections exist
        self._validate_config_structure()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration with optimized error handling and caching.
        
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            # Use pathlib for faster file operations
            config_text = self.config_path.read_text(encoding='utf-8')
            
            # Single YAML parse - most expensive operation
            config = yaml.safe_load(config_text)
            
            if config is None:
                config = {}
                
            logger.info(f"✅ Config loaded from: {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"❌ Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error loading config: {e}")
            raise

    def _validate_config_structure(self) -> None:
        """Validate configuration structure on load for fast access later."""
        required_sections = ['thresholds', 'models', 'flags']
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"⚠️ Missing config section: {section}")
                self.config[section] = {}

    @lru_cache(maxsize=128)
    def get_threshold(self, key: str) -> Union[int, float]:
        """
        Retrieve a threshold value by key with LRU caching for O(1) repeated access.
        
        Args:
            key: Threshold key name
            
        Returns:
            Threshold value
            
        Raises:
            ValueError: If threshold key not found
        """
        try:
            return self._thresholds_cache[key]
        except KeyError:
            raise ValueError(f"Threshold '{key}' not found in config.")

    @lru_cache(maxsize=64)
    def get_model_path(self, model_name: str) -> str:
        """
        Return the file path for a registered model with caching.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model file path
            
        Raises:
            ValueError: If model path not found
        """
        try:
            return self._models_cache[model_name]
        except KeyError:
            raise ValueError(f"Model path for '{model_name}' not found in config.")

    @lru_cache(maxsize=32)
    def get_flag(self, flag_name: str) -> Union[bool, str, int, float]:
        """
        Return a boolean or string flag with caching.
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            Flag value (defaults to False if not found)
        """
        return self._flags_cache.get(flag_name, False)

    def get_all(self) -> Dict[str, Any]:
        """
        Return the full parsed config (no caching needed as it's already in memory).
        
        Returns:
            Complete configuration dictionary
        """
        return self.config

    # Additional optimized methods for common access patterns
    
    @lru_cache(maxsize=16)
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get an entire configuration section with caching.
        
        Args:
            section_name: Name of the configuration section
            
        Returns:
            Section dictionary (empty dict if not found)
        """
        return self.config.get(section_name, {})

    def get_threshold_batch(self, keys: tuple) -> Dict[str, Union[int, float]]:
        """
        Get multiple thresholds in a single call for better performance.
        
        Args:
            keys: Tuple of threshold keys (tuple for hashability)
            
        Returns:
            Dictionary mapping keys to threshold values
        """
        result = {}
        for key in keys:
            try:
                result[key] = self.get_threshold(key)
            except ValueError:
                logger.warning(f"Threshold '{key}' not found, skipping")
        return result

    def get_model_paths_batch(self, model_names: tuple) -> Dict[str, str]:
        """
        Get multiple model paths in a single call.
        
        Args:
            model_names: Tuple of model names
            
        Returns:
            Dictionary mapping model names to paths
        """
        result = {}
        for model_name in model_names:
            try:
                result[model_name] = self.get_model_path(model_name)
            except ValueError:
                logger.warning(f"Model '{model_name}' not found, skipping")
        return result

    @lru_cache(maxsize=8)
    def has_threshold(self, key: str) -> bool:
        """
        Fast check if threshold exists without raising exceptions.
        
        Args:
            key: Threshold key name
            
        Returns:
            True if threshold exists, False otherwise
        """
        return key in self._thresholds_cache

    @lru_cache(maxsize=8)
    def has_model(self, model_name: str) -> bool:
        """
        Fast check if model path exists without raising exceptions.
        
        Args:
            model_name: Model name
            
        Returns:
            True if model exists, False otherwise
        """
        return model_name in self._models_cache

    @lru_cache(maxsize=8) 
    def has_flag(self, flag_name: str) -> bool:
        """
        Fast check if flag exists without raising exceptions.
        
        Args:
            flag_name: Flag name
            
        Returns:
            True if flag exists, False otherwise
        """
        return flag_name in self._flags_cache

    def reload_config(self) -> None:
        """
        Reload configuration from disk and clear caches.
        Useful for configuration changes without restart.
        """
        with self._config_lock:
            # Clear all caches
            self.get_threshold.cache_clear()
            self.get_model_path.cache_clear()
            self.get_flag.cache_clear()
            self.get_section.cache_clear()
            self.has_threshold.cache_clear()
            self.has_model.cache_clear()
            self.has_flag.cache_clear()
            
            # Reload configuration
            self.config = self._load_config()
            
            # Update cached sections
            self._thresholds_cache = self.config.get('thresholds', {})
            self._models_cache = self.config.get('models', {})
            self._flags_cache = self.config.get('flags', {})
            
            logger.info("🔄 Configuration reloaded and caches cleared")

    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache statistics for performance monitoring.
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        return {
            'get_threshold': self.get_threshold.cache_info()._asdict(),
            'get_model_path': self.get_model_path.cache_info()._asdict(),
            'get_flag': self.get_flag.cache_info()._asdict(),
            'get_section': self.get_section.cache_info()._asdict(),
            'has_threshold': self.has_threshold.cache_info()._asdict(),
            'has_model': self.has_model.cache_info()._asdict(),
            'has_flag': self.has_flag.cache_info()._asdict()
        }

    def optimize_for_access_pattern(self, 
                                  common_thresholds: Optional[tuple] = None,
                                  common_models: Optional[tuple] = None,
                                  common_flags: Optional[tuple] = None) -> None:
        """
        Pre-warm caches with commonly accessed keys for optimal performance.
        
        Args:
            common_thresholds: Tuple of frequently accessed threshold keys
            common_models: Tuple of frequently accessed model names  
            common_flags: Tuple of frequently accessed flag names
        """
        logger.info("🚀 Pre-warming caches for optimal performance...")
        
        # Pre-warm threshold cache
        if common_thresholds:
            for key in common_thresholds:
                try:
                    self.get_threshold(key)
                except ValueError:
                    pass
        
        # Pre-warm model path cache
        if common_models:
            for model in common_models:
                try:
                    self.get_model_path(model)
                except ValueError:
                    pass
        
        # Pre-warm flag cache
        if common_flags:
            for flag in common_flags:
                self.get_flag(flag)
        
        logger.info("✅ Cache pre-warming completed")


# Factory function for easy instantiation with common optimizations
def create_optimized_config_loader(config_path: Optional[str] = None,
                                 pre_warm: bool = True,
                                 common_thresholds: Optional[tuple] = None,
                                 common_models: Optional[tuple] = None,
                                 common_flags: Optional[tuple] = None) -> OptimizedConfigLoader:
    """
    Create an optimized config loader with optional pre-warming.
    
    Args:
        config_path: Path to configuration file
        pre_warm: Whether to pre-warm caches
        common_thresholds: Common threshold keys to pre-cache
        common_models: Common model names to pre-cache
        common_flags: Common flag names to pre-cache
        
    Returns:
        Optimized configuration loader instance
    """
    loader = OptimizedConfigLoader(config_path)
    
    if pre_warm:
        loader.optimize_for_access_pattern(
            common_thresholds=common_thresholds,
            common_models=common_models, 
            common_flags=common_flags
        )
    
    return loader


# Backward compatibility alias
ConfigLoader = OptimizedConfigLoader


# Performance comparison utility
def benchmark_config_access(loader: OptimizedConfigLoader, 
                          iterations: int = 1000) -> Dict[str, float]:
    """
    Benchmark configuration access performance.
    
    Args:
        loader: Configuration loader instance
        iterations: Number of benchmark iterations
        
    Returns:
        Performance timing results
    """
    import time
    
    # Sample keys for benchmarking
    threshold_keys = list(loader._thresholds_cache.keys())[:5]
    model_keys = list(loader._models_cache.keys())[:5]
    flag_keys = list(loader._flags_cache.keys())[:5]
    
    results = {}
    
    if threshold_keys:
        start = time.perf_counter()
        for _ in range(iterations):
            for key in threshold_keys:
                try:
                    loader.get_threshold(key)
                except ValueError:
                    pass
        results['threshold_access_time'] = time.perf_counter() - start
    
    if model_keys:
        start = time.perf_counter()
        for _ in range(iterations):
            for key in model_keys:
                try:
                    loader.get_model_path(key)
                except ValueError:
                    pass
        results['model_access_time'] = time.perf_counter() - start
    
    if flag_keys:
        start = time.perf_counter()
        for _ in range(iterations):
            for key in flag_keys:
                loader.get_flag(key)
        results['flag_access_time'] = time.perf_counter() - start
    
    return results


if __name__ == "__main__":
    # Example usage and performance demonstration
    try:
        # Create optimized loader
        loader = create_optimized_config_loader(
            pre_warm=True,
            common_thresholds=('drift_threshold', 'error_threshold'),
            common_models=('lstm_drift_predictor', 'qec_decoder_cnn'),
            common_flags=('enable_logging', 'debug_mode')
        )
        
        # Demonstrate fast access
        print("⚡ Configuration Access Demo:")
        print(f"Threshold example: {loader.get_threshold('drift_threshold') if loader.has_threshold('drift_threshold') else 'N/A'}")
        print(f"Model path example: {loader.get_model_path('lstm_drift_predictor') if loader.has_model('lstm_drift_predictor') else 'N/A'}")
        print(f"Flag example: {loader.get_flag('enable_logging')}")
        
        # Show cache statistics
        cache_stats = loader.get_cache_stats()
        print(f"\n📊 Cache Statistics: {cache_stats}")
        
        # Performance benchmark
        benchmark_results = benchmark_config_access(loader)
        print(f"\n🏃 Performance Benchmark: {benchmark_results}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo requires a valid config file. Error: {e}")



"""  usage in controller.py

from config.config import ConfigLoader

config = ConfigLoader()

fidelity_threshold = config.get_threshold("fidelity_min")
model_path = config.get_model_path("digital_twin")
logging_enabled = config.get_flag("enable_logging")
"""