
# ai_control_main/ai_control_main.py

from fastapi import APIRouter

router = APIRouter()

# path as /api/aicontrol/
@router.get("/")
async def run_ai_control_engine_job():
    return {"message": "AI control engine job is working!"}



# ai_control_main/ai_control_main.py

from fastapi import APIRouter

router = APIRouter()

# path as /api/aicontrol/
@router.get("/")
async def run_ai_control_engine_job():
    return {"message": "AI control engine job is working!"}


import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
import threading
from dataclasses import dataclass
from functools import lru_cache

# Core engine modules
from controller import AIController
from orchestrator import Orchestrator
from digital_twin import DigitalTwin
from model_manager import ModelManager
from utils import load_yaml_config, setup_logger, Timer
from config.config import CONFIG_PATH

"""
1	Collect live data from sensors
2	Predict expected state via digital twin
3	Run AI/ML inference using multiple models
4	Decide on control action via orchestrator
5	Apply decision to system
6	Log cycle data for traceability
"""

# Performance monitoring
@dataclass
class EngineMetrics:
    """Lightweight performance metrics."""
    cycle_count: int = 0
    total_time: float = 0.0
    sensor_time: float = 0.0
    inference_time: float = 0.0
    decision_time: float = 0.0
    
    @property
    def avg_cycle_time(self) -> float:
      
    
    @property
    def cycles_per_second(self) -> float:



class OptimizedAIControlEngine:
    """
    High-performance AI Control Engine with optimizations:
    - Concurrent execution of independent operations
    - Smart caching and object reuse
    - Minimal memory allocations
    - Predictive preloading
    - Lock-free performance monitoring
    """
    
    def __init__(self, config_path: str = CONFIG_PATH, max_workers: int = 4):
        _cache_time = 0
 
    
    @property
    @lru_cache(maxsize=1)
    def config(self) -> Dict[str, Any]:
        """Cached configuration loading with TTL."""
        
    
    @property
    def controller(self) -> AIController:
        """Lazy-loaded controller."""
     
    
    @property
    def orchestrator(self) -> Orchestrator:
        """Lazy-loaded orchestrator."""
       
    @property
    def twin(self) -> DigitalTwin:
        """Lazy-loaded digital twin."""
      
    
    @property
    def model_manager(self) -> ModelManager:
        """Lazy-loaded model manager."""
       
    
    async def warmup(self):
        """Pre-load and warm up all components for optimal performance."""
        
        
   
        
        
          
    
    
    async def _get_sensor_data_async(self) -> Dict[str, Any]:
        """Asynchronous sensor data retrieval."""
       
    
    async def _execute_control_cycle_async(self, 
                                         sensor_data: Optional[Dict[str, Any]] = None,
                                         warmup: bool = False) -> Tuple[Dict[str, Any], float]:
        """
        Optimized async control cycle with concurrent execution.
        
        Returns:
            Tuple of (decision, cycle_time)
        """
        
    async def run_control_loop(self):
        """
        High-performance async control loop with optimizations:
        - Adaptive sleep timing
        - Predictive scheduling
        - Graceful error recovery
        """
        
    
    async def _graceful_shutdown(self):
        """Graceful shutdown with resource cleanup."""
        
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        
    def request_shutdown(self):
        """Request graceful shutdown."""
       

# Factory function for easy initialization
def create_optimized_engine(config_path: str = CONFIG_PATH,
                          max_workers: int = 4,
                          auto_warmup: bool = True) -> OptimizedAIControlEngine:
    """Create and optionally warm up an optimized AI control engine."""
   

# High-performance async main
async def main():
    """Optimized main execution with performance monitoring."""
    # Create optimized engine
    

# Legacy synchronous wrapper for backward compatibility
def run_sync():
    """Synchronous wrapper for legacy compatibility."""
  


if __name__ == "__main__":
    # Run with high-performance async execution
    asyncio.run(main())


