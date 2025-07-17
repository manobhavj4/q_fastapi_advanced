import os
import logging
import argparse
import time
import threading
from typing import Optional, Dict, Any, Union
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import pickle
import warnings
warnings.filterwarnings("ignore")

import mlflow
from mlflow import pyfunc
from mlflow.models.signature import infer_signature
from ai_control_engine.config.config import MODEL_ARTIFACT_PATH, ML_EXPERIMENT_NAME
from ai_control_engine.utils import load_yaml_config

# Optimized logging configuration


# Global optimization settings
MLFLOW_CACHE = {}
CACHE_LOCK = threading.RLock()
MAX_WORKERS = 4


class OptimizedModelRegistry:
    """
    High-performance model registry with caching, lazy loading, and concurrent operations.
    Optimized for minimal computation time and reduced complexity.
    """
    
    def __init__(self, experiment_name: str = ML_EXPERIMENT_NAME, max_workers: int = MAX_WORKERS):
      
    
    @lru_cache(maxsize=1)
    def _initialize_experiment(self):
        """Initialize MLflow experiment with caching."""
        
    
    def _get_cached_model(self, model_path: str, model_type: str) -> Any:
        """Load model with intelligent caching."""
        
    
    def _load_model_by_type(self, model_path: str, model_type: str) -> Any:
        """Optimized model loading by type."""
       
    def _infer_signature_cached(self, model: Any, input_example: Any, model_type: str) -> Optional[Any]:
        """Cache signature inference for repeated operations."""
        
          
    
    def _log_model_optimized(self, model: Any, model_type: str, model_name: str, 
                           signature: Optional[Any] = None, metadata: Optional[Dict] = None) -> str:
        """Optimized model logging with type-specific optimizations."""
       f"📤 Model logged in {log_time:.3f}s")
            
      
    
    def register_model_fast(self, model_path: str, model_name: str, model_type: str = "pytorch",
                           input_example: Optional[Any] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fast model registration with optimizations.
        
        Args:
            model_path: Path to the saved model
            model_name: Unique name for registration
            model_type: Model framework type
            input_example: Example input for signature inference
            metadata: Optional metadata dictionary
            
        Returns:
            Registration result with timing information
        """
        
       
    
    def batch_register_models(self, model_configs: list) -> list:
        """
        Register multiple models concurrently for maximum throughput.
        
        Args:
            model_configs: List of dictionaries with model configuration
            
        Returns:
            List of registration results
        """
        
    
    def clear_cache(self):
        """Clear all caches to free memory."""
    


# Global optimized registry instance
_global_registry = None
_registry_lock = threading.Lock()


def get_optimized_registry() -> OptimizedModelRegistry:
    """Get or create global optimized registry instance."""
 


def register_model(model_path: str, model_name: str, model_type: str = "pytorch",
                  input_example: Optional[Any] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Optimized model registration function (backward compatible).
    
    This function maintains the same interface as the original but with significant
    performance optimizations including caching, concurrent operations, and
    reduced computational complexity.
    """



def main():
    """Optimized main function with enhanced argument parsing."""
   