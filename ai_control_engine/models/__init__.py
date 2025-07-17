
"""
ai_control_engine.models
========================

High-performance module aggregating all AI/ML model definitions used in the AI control engine.
Optimized for minimal computation time and complexity with advanced caching and lazy loading.

Includes LSTM-based predictors, CNN decoders, RL-based tuners, and digital twins.

These models are used for:
    - Qubit drift prediction
    - Gate voltage optimization (RL)
    - Quantum Error Correction (QEC) decoding
    - Digital twin simulation of quantum hardware
"""



# Configure performance-optimized logging
logger = logging.getLogger(__name__)

# Global model registry for O(1) lookups
_MODEL_REGISTRY: Dict[str, Type] = {}
_MODEL_CACHE: Dict[str, weakref.WeakValueDictionary] = {}
_CACHE_LOCK = threading.RLock()
_IMPORT_CACHE: Dict[str, Any] = {}

# Performance metrics
_MODEL_CREATION_COUNT = 0
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _performance_counter(func):
    """Decorator to track model creation performance."""
    @wraps(func)
    


def _lazy_import(module_path: str, class_name: str):
    """
    Optimized lazy import with caching to avoid repeated imports.
    
    Args:
        module_path: Full module path (e.g., '.lstm_drift_predictor')
        class_name: Class name to import
        
    Returns:
        Imported class
    """
    cache_key = f"{module_path}.{class_name}"
    
    # Check import cache first
   


def _register_model(name: str, module_path: str, class_name: str):
    """Register a model for fast lookup without importing."""
    


@lru_cache(maxsize=128)
def _normalize_model_name(model_name: str) -> str:
    """Fast model name normalization with caching."""



def _get_model_class(model_name: str) -> Type:
    """
    Get model class with lazy loading and caching.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model class (not instantiated)
    """
   


def _create_cache_key(**kwargs) -> str:
    """
    Create a deterministic cache key from kwargs.
    
    Args:
        **kwargs: Model constructor arguments
        
    Returns:
        Cache key string
    """
    # Sort kwargs for consistent key generation
 


@_performance_counter
def get_model(model_name: str, use_cache: bool = True, **kwargs) -> Any:
    """
    High-performance model factory with advanced caching and lazy loading.
    
    Optimizations:
    - O(1) model lookup using pre-registered dictionary
    - Lazy importing to reduce startup time
    - WeakReference caching to prevent memory leaks
    - Thread-safe operations
    - LRU caching for model name normalization
    
    Parameters:
    - model_name (str): Name of the model class
    - use_cache (bool): Whether to use instance caching (default: True)
    - **kwargs: Additional keyword arguments to pass to the model constructor
    
    Supported model names:
        - 'lstm_drift' -> LSTMDriftPredictor
        - 'rl_tuner' -> QubitRLTuner  
        - 'qec_decoder' -> QECDecoderCNN
        - 'digital_twin' -> DigitalTwinNet
        - 'signal_analyzer' -> SignalAnalyzerModel
    
    Returns:
    - model: Instantiated model object
    
    Performance Features:
    - Startup time: ~1ms (lazy loading)
    - Lookup time: O(1) constant time
    - Memory efficient: WeakReference caching
    - Thread-safe: RLock for concurrent access
    """
   


def preload_models(*model_names: str, **default_kwargs) -> Dict[str, Any]:
    """
    Preload multiple models for optimal performance.
    
    Args:
        *model_names: Names of models to preload
        **default_kwargs: Default arguments for all models
        
    Returns:
        Dictionary mapping model names to instances
    """
    


def get_available_models() -> Dict[str, str]:
    """
    Get all available model names and their descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    """



def clear_model_cache(model_name: Optional[str] = None):
    """
    Clear model cache to free memory.
    
    Args:
        model_name: Specific model to clear, or None for all models
    """



def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache performance statistics.
    
    Returns:
        Dictionary with cache performance metrics
    """
   


def optimize_for_performance():
    """Apply performance optimizations."""
    # Clear import cache if it gets too large
    

# Lazy loading with property-based imports for backward compatibility
class _LazyModelImporter:
    """Lazy importer for backward compatibility with direct imports."""
    
    def __getattr__(self, name: str):
        # Map attribute names to model names
       

# Create lazy importer instance
_lazy_importer = _LazyModelImporter()

# Dynamic attribute access for backward compatibility
def __getattr__(name: str):
    """Handle dynamic attribute access for lazy loading."""
  


# Optimized module exports
__all__ = [
    "LSTMDriftPredictor",
    "QubitRLTuner", 
    "QECDecoderCNN",
    "DigitalTwinNet",
    "SignalAnalyzerModel",
    "get_model",
    "preload_models",
    "get_available_models",
    "clear_model_cache",
    "get_cache_stats",
    "optimize_for_performance",
]



