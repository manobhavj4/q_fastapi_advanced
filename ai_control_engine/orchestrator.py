import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import threading
from collections import deque
import weakref

from ai_control_engine.controller import Controller
from ai_control_engine.digital_twin import DigitalTwin
from ai_control_engine.model_manager import ModelManager
from ai_control_engine.config.config import get_config
from ai_control_engine.utils import timestamp_now, log_json

# Configure logging with better performance


# Constants for optimization
DEFAULT_DECISION_INTERVAL = 1.0
MAX_WORKER_THREADS = 4
BATCH_SIZE = 10
CACHE_SIZE = 100
PREDICTION_TIMEOUT = 0.5


@dataclass
class SystemState:
    """Optimized state container with slots for memory efficiency."""
    __slots__ = ['drift', 'error_rate', 'tuning_action', 'feedback_signal', 'timestamp']
    
    drift: Optional[float] = None
    error_rate: Optional[float] = None
    tuning_action: Optional[Dict[str, Any]] = None
    feedback_signal: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CycleResult:
    """Lightweight result container."""
    __slots__ = ['timestamp', 'input_data', 'prediction', 'state', 'processing_time']
    
    timestamp: float
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    state: SystemState
    processing_time: float


class OptimizedOrchestrator:
    """
    High-performance orchestrator with the following optimizations:
    - Async/await for concurrent processing
    - Thread pooling for CPU-intensive tasks
    - Batch processing capabilities
    - Memory-efficient data structures
    - Caching for frequently accessed data
    - Reduced logging overhead
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logger.info("🚦 Initializing Optimized Orchestrator")
        
        # Load runtime configurations
        self.config = config or get_config()
        self.decision_interval = self.config.get("decision_interval", DEFAULT_DECISION_INTERVAL)
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)
        
        # Initialize subsystems
        self._initialize_subsystems()
        
        # Performance tracking
        self.cycle_times = deque(maxlen=100)  # Keep last 100 cycle times
        self.total_cycles = 0
        
        # State management
        self.current_state = SystemState()
        self._state_lock = threading.RLock()
        
        # Caching for predictions (using weak references to prevent memory leaks)
        self._prediction_cache = weakref.WeakValueDictionary()
        
        # Batch processing queue
        self._batch_queue = deque(maxlen=BATCH_SIZE * 2)
        self._batch_lock = threading.Lock()
        
        # Performance flags
        self.enable_logging = self.config.get("enable_detailed_logging", False)
        self.enable_caching = self.config.get("enable_caching", True)
        
        logger.info("✅ Orchestrator initialized successfully")
    
    def _initialize_subsystems(self):
        """Initialize subsystems with error handling."""
        try:
            self.controller = Controller(self.config)
            self.digital_twin = DigitalTwin(self.config)
            self.model_manager = ModelManager(self.config["model_paths"])
            
            # Load models concurrently
            self._load_models_concurrent()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize subsystems: {e}")
            raise
    
    def _load_models_concurrent(self):
        """Load models concurrently for faster startup."""
        model_futures = {
            "drift_model": self.executor.submit(self.model_manager.load_model, "lstm_drift"),
            "qec_decoder": self.executor.submit(self.model_manager.load_model, "qec_decoder"),
            "rl_tuner": self.executor.submit(self.model_manager.load_model, "qubit_rl_tuner")
        }
        
        # Wait for all models to load
        for name, future in model_futures.items():
            try:
                setattr(self, name, future.result(timeout=30))
                logger.info(f"✅ Loaded {name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
                raise
    
    def _create_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Create a cache key for input data."""
        # Simple hash-based cache key (can be improved based on data structure)
        return str(hash(frozenset(input_data.items())))
    
    async def _predict_digital_twin(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for digital twin prediction."""
        if self.enable_caching:
            cache_key = self._create_cache_key(input_data)
            if cache_key in self._prediction_cache:
                return self._prediction_cache[cache_key]
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        try:
            prediction = await asyncio.wait_for(
                loop.run_in_executor(self.executor, self.digital_twin.predict, input_data),
                timeout=PREDICTION_TIMEOUT
            )
            
            if self.enable_caching:
                self._prediction_cache[cache_key] = prediction
            
            return prediction
        except asyncio.TimeoutError:
           # logger.warning("⚠️ Digital twin prediction timed out, using cached result")
            return self._prediction_cache.get(self._create_cache_key(input_data), {})
    
    async def _process_models_concurrent(self, input_data: Dict[str, Any]) -> tuple:
        """Process all models concurrently for better performance."""
        loop = asyncio.get_event_loop()
        
        # Create tasks for concurrent execution
        tasks = [
            loop.run_in_executor(self.executor, self.drift_model.predict, input_data["signal"]),
            loop.run_in_executor(self.executor, self.qec_decoder.decode, input_data["qubit_output"]),
            self._predict_digital_twin(input_data)
        ]
        
        # Wait for all tasks to complete
        try:
            drift_value, error_rate, predicted_behavior = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            if isinstance(drift_value, Exception):
               # logger.error(f"❌ Drift prediction failed: {drift_value}")
                drift_value = self.current_state.drift  # Use last known value
            
            if isinstance(error_rate, Exception):
               # logger.error(f"❌ QEC decoding failed: {error_rate}")
                error_rate = self.current_state.error_rate
            
            if isinstance(predicted_behavior, Exception):
              #  logger.error(f"❌ Digital twin prediction failed: {predicted_behavior}")
                predicted_behavior = {}
            
            return drift_value, error_rate, predicted_behavior
            
        except Exception as e:
           # logger.error(f"❌ Concurrent processing failed: {e}")
            # Return last known values
            return self.current_state.drift, self.current_state.error_rate, {}
    
    async def run_cycle_async(self, input_data: Dict[str, Any]) -> CycleResult:
        """Optimized async cycle execution."""
        
    
    async def _log_result_async(self, result: CycleResult):
        """Asynchronous logging to avoid blocking main execution."""
        try:
            log_data = {
                "timestamp": result.timestamp,
                "input": result.input_data,
                "prediction": result.prediction,
                "state": {
                    "drift": result.state.drift,
                    "error_rate": result.state.error_rate,
                    "tuning_action": result.state.tuning_action,
                    "feedback_signal": result.state.feedback_signal
                },
                "processing_time": result.processing_time
            }
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, log_json, "cycle_log.json", log_data)
            
        except Exception as e:
           # logger.error(f"❌ Logging failed: {e}")
    
    def run_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for backward compatibility."""
       
    
    async def run_loop_async(self, data_stream):
        """
        Optimized async real-time loop with better resource management.
        """
       
    
    def run_loop(self, data_stream):
        """Synchronous wrapper for the real-time loop."""
    
    
    async def _cleanup(self):
        """Async cleanup of resources."""
     
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
      
    
    def __enter__(self):
    
    
    def __exit__(self, exc_type, exc_val, exc_tb):
    
    


# Backward compatibility alias
Orchestrator = OptimizedOrchestrator