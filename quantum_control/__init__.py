"""
Quantum Control Package

This package provides AI-integrated quantum control tools for simulation,
pulse optimization, error correction, and job execution using Qiskit and QuTiP.
"""

import os
import sys
from typing import Dict, Any, Optional
from functools import lru_cache
import warnings

# Metadata
__version__ = "0.1.0"
__author__ = "GenXZ Quantum Research"

# Performance optimizations
_LAZY_IMPORTS = {}
_CONFIG_CACHE = {}

def _lazy_import(module_name: str, class_name: str):
    """Lazy import with caching for better performance."""


def _get_module_path(module_name: str) -> str:
    """Get the full module path for dynamic imports."""
    return f"{__name__}.{module_name}"

# Core module lazy loading properties
class _LazyModuleLoader:
    """Lazy module loader with property-based access."""
    
    @property
    def QuantumDotSimulator(self):
        return _lazy_import("simulator", "QuantumDotSimulator")
    
    @property
    def AutoTuner(self):
        return _lazy_import("auto_tuner", "AutoTuner")
    
    @property
    def GateOptimizer(self):
        return _lazy_import("gate_optimizer", "GateOptimizer")
    
    @property
    def PulseOptimizer(self):
        return _lazy_import("pulse_optimizer", "PulseOptimizer")
    
    @property
    def FidelityDriftPredictor(self):
        return _lazy_import("fidelity_drift_predictor", "FidelityDriftPredictor")
    
    @property
    def QECDecoder(self):
        return _lazy_import("qec_decoder", "QECDecoder")
    
    @property
    def QuantumJobManager(self):
        return _lazy_import("jobs", "QuantumJobManager")
    
    @property
    def QiskitRunner(self):
        return _lazy_import("qiskit_runner", "QiskitRunner")
    
    @property
    def QuantumJobAPI(self):
        return _lazy_import("quantum_job_api", "QuantumJobAPI")
    
    @property
    def ThermalNoiseEstimator(self):
        return _lazy_import("noise_modeling.thermal_noise_estimator", "ThermalNoiseEstimator")

# Create the lazy loader instance
_loader = _LazyModuleLoader()

# Export core classes through lazy loading
def __getattr__(name: str):
    """Dynamic attribute access for lazy loading."""
  

# Optimized configuration loading with caching
@lru_cache(maxsize=8)
def load_sim_config(config_path: str = "config/sim_config.yaml") -> Dict[str, Any]:
    """
    Loads simulator configuration YAML with caching for performance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # Check cache first
   
# Optimized bulk imports for development/testing
def _import_all_modules():
    """Import all modules at once for development environments."""
    modules = [
        'simulator', 'auto_tuner', 'gate_optimizer', 'pulse_optimizer',
        'fidelity_drift_predictor', 'qec_decoder', 'jobs', 'qiskit_runner',
        'quantum_job_api', 'noise_modeling.thermal_noise_estimator',
        'utils', 'circuits.circuit_templates', 'circuits.qaoa_example'
    ]
    


# Performance utilities
def get_import_status() -> Dict[str, bool]:
    """Get the current import status of all modules."""
    return {key: value is not None for key, value in _LAZY_IMPORTS.items()}

def clear_import_cache():
    """Clear the import cache (useful for testing)."""
  

def preload_core_modules():
    """Preload core modules for performance-critical applications."""



# Define __all__ for explicit imports
