import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache, wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.entities import RegisteredModel, ModelVersion

# Optional: Load from .env or config
from ai_control_engine.config.config import AI_ENGINE_CONFIG

# Configure high-performance logging


@dataclass
class RegistryMetrics:
    """Performance metrics for registry operations."""



def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)



def cache_client_results(maxsize=128, ttl_seconds=300):
    """LRU cache with time-to-live for client results."""
   

class OptimizedMlflowClient:
    """High-performance MLflow client with connection pooling and caching."""
    
    def __init__(self, tracking_uri: str, registry_uri: str):
       
        
    @lru_cache(maxsize=2)
    def get_client(self, client_type: str = "default") -> MlflowClient:
        """Get cached MLflow client with connection pooling."""
        
    
    def clear_cache(self):
        """Clear client cache."""
        with self._client_lock:
            self._client_cache.clear()
            self.get_client.cache_clear()


class OptimizedModelRegistry:
    """
    High-performance model registry with advanced optimizations:
    - Connection pooling and client caching
    - Result caching with TTL
    - Concurrent operations support
    - Performance metrics tracking
    - Batch operations for efficiency
    """
    
    def __init__(self, tracking_uri: str = None, registry_uri: str = None, 
                 max_workers: int = 4, enable_caching: bool = True):
        
        # Configuration
       
    
    def _configure_mlflow(self):
        """Configure MLflow settings for optimal performance."""
       
    
    @timing_decorator
    def register_model(self, model_uri: str, model_name: str, 
                      tags: Dict[str, str] = None, description: str = "",
                      async_execution: bool = False) -> Any:
        """
        Register model with optimized performance and optional async execution.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            tags: Optional tags to apply
            description: Optional description
            async_execution: Whether to execute asynchronously
            
        Returns:
            Registration result
        """
        
    def _register_model_sync(self, model_uri: str, model_name: str,
                           tags: Dict[str, str] = None, description: str = "") -> Any:
        """Synchronous model registration with error handling."""
      
    def _apply_model_metadata(self, model_name: str, tags: Dict[str, str] = None,
                            description: str = ""):
        """Apply tags and description efficiently in batch."""
     
    @timing_decorator
    @cache_client_results(maxsize=64, ttl_seconds=120)
    def list_models(self, max_results: int = 1000) -> List[RegisteredModel]:
        """
        List registered models with caching and pagination support.
        
        Args:
            max_results: Maximum number of models to return
            
        Returns:
            List of registered models
        """
       
    
    @timing_decorator
    @cache_client_results(maxsize=128, ttl_seconds=60)
    def get_latest_model_version(self, model_name: str, 
                               stage: str = "None") -> Optional[ModelVersion]:
        """
        Get latest model version with caching.
        
        Args:
            model_name: Name of the registered model
            stage: Stage filter (None, Staging, Production, Archived)
            
        Returns:
            Latest model version or None
        """
       
    
    @timing_decorator
    def get_model_versions_batch(self, model_names: List[str], 
                               stage: str = "None") -> Dict[str, Optional[ModelVersion]]:
        """
        Get latest versions for multiple models concurrently.
        
        Args:
            model_names: List of model names
            stage: Stage filter
            
        Returns:
            Dictionary mapping model names to their latest versions
        """
        def get_version(name):
        
    
    @timing_decorator
    def transition_model_stage(self, model_name: str, version: int, 
                             stage: str, archive_existing: bool = True) -> bool:
        """
        Transition model stage with enhanced error handling.
        
        Args:
            model_name: Name of the registered model
            version: Model version number
            stage: Target stage
            archive_existing: Whether to archive existing versions
            
        Returns:
            Success status
        """
       
    
    @timing_decorator
    def delete_model_version(self, model_name: str, version: int,
                           confirm_deletion: bool = False) -> bool:
        """
        Delete model version with safety checks.
        
        Args:
            model_name: Name of the registered model
            version: Version to delete
            confirm_deletion: Safety confirmation flag
            
        Returns:
            Success status
        """
      
    
    @timing_decorator
    def delete_registered_model(self, model_name: str,
                              confirm_deletion: bool = False) -> bool:
        """
        Delete entire registered model with safety checks.
        
        Args:
            model_name: Name of the model to delete
            confirm_deletion: Safety confirmation flag
            
        Returns:
            Success status
        """
        
    
    @contextmanager
    def batch_operations(self):
        """Context manager for batch operations with optimized client reuse."""
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        
    def clear_caches(self):
        """Clear all caches to free memory."""
      
    def optimize_for_throughput(self):
        """Apply throughput optimizations."""
        # Increase thread pool size
       
    def shutdown(self):
        """Graceful shutdown with cleanup."""
      
# Factory function for easy instantiation
def create_optimized_registry(tracking_uri: str = None, registry_uri: str = None,
                            max_workers: int = 4, auto_optimize: bool = True) -> OptimizedModelRegistry:
    """Create an optimized model registry."""
    

# Backward compatibility alias
ModelRegistry = OptimizedModelRegistry


# Example usage and testing
if __name__ == "__main__":
    # Performance testing
   
