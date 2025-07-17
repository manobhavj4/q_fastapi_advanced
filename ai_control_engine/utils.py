import os
import json
import time
import logging
import yaml
from functools import wraps, lru_cache
from typing import Any, Dict, Callable, Optional, Union
from pathlib import Path
import threading
from contextlib import contextmanager


# ------------------------------------------------------------------------------
# Constants & Configuration
# ------------------------------------------------------------------------------
_LOGGER_CACHE = {}
_LOGGER_LOCK = threading.Lock()
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_JSON_INDENT = 4


# ------------------------------------------------------------------------------
# Optimized Logger Setup
# ------------------------------------------------------------------------------
def get_logger(name: str = "AIControlEngine", level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Thread-safe cached logger factory with optimized singleton pattern.
    Uses thread-local storage to avoid lock contention.
    """
    


# ------------------------------------------------------------------------------
# High-Performance Timing Utilities
# ------------------------------------------------------------------------------
def timeit(func: Callable = None, *, log_level: int = logging.DEBUG) -> Callable:
    """
    Optimized timing decorator with minimal overhead.
    Uses perf_counter for higher precision and configurable log levels.
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
       


@contextmanager
def timer(name: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks with zero decorator overhead.
    """
   



# ------------------------------------------------------------------------------
# Optimized File I/O Utilities
# ------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _get_path_object(filepath: str) -> Path:
    """Cached Path object creation to avoid repeated string parsing."""
 


def load_json(filepath: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Optimized JSON loading with Path objects and better error handling.
    """
 


def save_json(data: Dict[str, Any], 
              filepath: str, 
              indent: int = DEFAULT_JSON_INDENT,
              encoding: str = 'utf-8',
              ensure_ascii: bool = False) -> None:
    """
    Optimized JSON saving with atomic writes and better performance.
    """
    

# ------------------------------------------------------------------------------
# Optimized YAML Utilities
# ------------------------------------------------------------------------------
# Cache YAML loader for better performance
_yaml_loader = yaml.SafeLoader


def load_yaml(filepath: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Optimized YAML loading with cached loader and Path objects.
    """



def save_yaml(data: Dict[str, Any], 
              filepath: str, 
              encoding: str = 'utf-8',
              default_flow_style: bool = False) -> None:
    """
    Optimized YAML saving with atomic writes.
    """



# ------------------------------------------------------------------------------
# Optimized Path Utilities
# ------------------------------------------------------------------------------
def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Optimized directory creation with Path objects.
    Returns Path object for chaining operations.
    """
 

@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Cached project root detection with Path objects.
    Only computed once per program execution.
    """
    


# ------------------------------------------------------------------------------
# Optimized Debug Utilities
# ------------------------------------------------------------------------------
def print_json(data: Dict[str, Any], 
               indent: int = DEFAULT_JSON_INDENT,
               sort_keys: bool = False) -> None:
    """
    Optimized JSON pretty-printing with configurable options.
    """
   


def print_yaml(data: Dict[str, Any], 
               default_flow_style: bool = False) -> None:
    """
    Pretty-print YAML data to console.
    """



# ------------------------------------------------------------------------------
# Batch File Operations (New Optimization)
# ------------------------------------------------------------------------------
def load_configs_batch(filepaths: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    Optimized batch loading of configuration files.
    Automatically detects JSON/YAML format and loads efficiently.
    """



# ------------------------------------------------------------------------------
# Performance Monitoring (New Feature)
# ------------------------------------------------------------------------------
class PerformanceMonitor:
    """
    Lightweight performance monitoring with minimal overhead.
    """
    
    def __init__(self):
       
    
    def start(self, name: str):
        """Start timing an operation."""
    
    def stop(self, name: str):
        """Stop timing and record the operation."""
    
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation."""
     
       
    
    def report(self) -> Dict[str, Dict[str, float]]:
        """Generate performance report."""
      


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


# ------------------------------------------------------------------------------
# Example Usage with Optimizations
# ------------------------------------------------------------------------------
@timeit
def optimized_computation():
    """Example showing optimized computation patterns."""
    # Simulate some work



def example_usage():
    """Demonstrate optimized usage patterns."""
    # Batch config loading
 
    

if __name__ == "__main__":
   