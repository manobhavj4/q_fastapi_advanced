import os
import logging
import threading
import time
from typing import Union, Dict, Any, Optional, Tuple
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from weakref import WeakValueDictionary
import pickle
import hashlib

import torch
import onnxruntime as ort

try:
    import mlflow.pyfunc
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Optimize ONNX Runtime settings globally


# Configure optimized logging


class ModelCache:
    """High-performance thread-safe model cache with LRU eviction and weak references."""
    
    def __init__(self, max_size: int = 20, max_memory_mb: int = 2048):
      
    
    def _generate_cache_key(self, model_path: str, format: str, use_gpu: bool = False) -> str:
        """Generate unique cache key with file hash for integrity."""
        # Include file modification time and size for cache invalidation
 
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Thread-safe cache retrieval with access time tracking."""
        
    
    def put(self, cache_key: str, model: Any, estimated_size: int = 0):
        """Thread-safe cache storage with memory management."""
       
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model memory footprint."""
   
    
    def _evict_if_needed(self, new_model_size: int):
        """Evict least recently used models to make space."""
        # Check if eviction needed

    
    def clear(self):
        """Clear all cached models."""
     
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
      


class OptimizedModelLoader:
    """
    Ultra-high-performance model loader with advanced caching, lazy loading,
    and concurrent model loading capabilities.
    
    Optimizations:
    - Thread-safe LRU cache with weak references
    - Lazy initialization and model validation
    - Concurrent model loading with thread pool
    - Memory-mapped file loading for large models
    - Optimized ONNX and PyTorch configurations
    - Automatic cache invalidation on file changes
    """
    
    def __init__(self, 
                 model_dir: str = "ai_control_engine/models/",
                 cache_size: int = 20,
                 max_cache_memory_mb: int = 2048,
                 enable_concurrent_loading: bool = True,
                 max_workers: int = 4):
        
       
    def _get_optimal_onnx_providers(self) -> list:
        """Get optimal ONNX execution providers based on hardware."""
       
    
    @lru_cache(maxsize=128)
    def _validate_model_path(self, model_path: str) -> Tuple[bool, str]:
        """Cached path validation to avoid repeated filesystem calls."""
       
    def load_model(self,
                   model_name: str,
                   format: str = "pytorch",
                   use_gpu: bool = False,
                   force_reload: bool = False) -> Union[torch.nn.Module, ort.InferenceSession, Any]:
        """
        High-performance model loading with caching and optimization.
        
        Args:
            model_name: Model file name or MLflow URI
            format: Format type ('pytorch', 'onnx', 'mlflow')
            use_gpu: Whether to use GPU (PyTorch only)
            force_reload: Force reload bypassing cache
            
        Returns:
            Loaded model object
        """
       
    
    def _load_pytorch_model_optimized(self, model_file: str, use_gpu: bool = False) -> torch.nn.Module:
        """Optimized PyTorch model loading with memory mapping and device optimization."""
        
    
    def _load_onnx_model_optimized(self, model_file: str) -> ort.InferenceSession:
        """Optimized ONNX model loading with provider selection and session options."""
       
    
    def _load_mlflow_model_optimized(self, model_uri: str) -> Any:
        """Optimized MLflow model loading with error handling."""
        
    
    def preload_models(self, model_configs: list) -> Dict[str, bool]:
        """
        Preload multiple models concurrently for optimal startup performance.
        
        Args:
            model_configs: List of dicts with 'name', 'format', 'use_gpu' keys
            
        Returns:
            Dict mapping model names to success status
        """
       
        
        # Concurrent loading
        
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        
    
    def clear_cache(self):
        """Clear all cached models and reset statistics."""
        
    
    def shutdown(self):
        """Gracefully shutdown the model loader."""
       


# Factory function for easy instantiation
def create_optimized_loader(model_dir: str = "ai_control_engine/models/",
                          cache_size: int = 20,
                          enable_preload: bool = True) -> OptimizedModelLoader:
    """Create an optimized model loader with sensible defaults."""
   


# Example usage with optimizations
if __name__ == "__main__":
    # Create optimized loader
   