
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import global_services
from typing import Tuple, Dict, List, Optional
from functools import lru_cache
from collections import deque
import threading


# Performance optimization constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32


class OptimizedQubitEnv:
    """
    High-performance environment for RL-based qubit tuning.
    Optimizations:
    - Vectorized operations
    - Pre-allocated memory
    - Cached computations
    - Reduced memory allocations
    """

    def __init__(self, simulator, action_scale: float = 1.0, 
                 max_episode_steps: int = 1000):
       
    def reset(self) -> np.ndarray:
        """Reset with pre-allocated buffers for zero-copy operations."""
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Optimized step with vectorized operations and minimal allocations."""
   


class EfficientActorCritic(nn.Module):
    """
    Optimized actor-critic network with:
    - Shared feature extraction
    - Efficient forward pass
    - Reduced parameter count
    - Optimized activation functions
    """

    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_size: int = 64, use_batch_norm: bool = False):
        super().__init__()
        
        # Shared feature extraction for efficiency
        

    def _initialize_weights(self):
        """Optimized weight initialization for faster convergence."""
       

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficient forward pass with shared feature extraction."""
        # Single pass through shared layers
       

    @torch.jit.script_method
    def forward_jit(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """JIT-compiled forward pass for maximum performance."""
       


class HighPerformanceQubitRLTuner:
    """
    Optimized RL agent with:
    - Efficient memory management
    - Vectorized operations
    - Optimized training loops
    - Smart caching
    - Reduced computational complexity
    """

    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 1e-3, gamma: float = 0.99,
                 batch_size: int = 32, buffer_size: int = 10000,
                 use_jit: bool = True):
        
     

    @torch.no_grad()
    def select_action(self, state: np.ndarray, 
                     exploration_noise: float = 0.0) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """
        Optimized action selection with:
        - Zero-copy operations
        - Efficient tensor operations
        - Optional exploration noise
        """
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, 
                      reward: float, next_state: np.ndarray, done: bool):
        """Add experience to replay buffer."""
        self.replay_buffer.append((state.copy(), action.copy(), reward, next_state.copy(), done))

    def update_vectorized(self, trajectory: Optional[List] = None, 
                         use_replay_buffer: bool = True) -> Dict[str, float]:
        """
        Optimized training update with:
        - Vectorized operations
        - Efficient gradient computation
        - Memory-efficient processing
        """
       


    @torch.jit.script_method
    def _compute_returns_vectorized(self, rewards: torch.Tensor, 
                                   dones: torch.Tensor) -> torch.Tensor:
        """Vectorized computation of discounted returns."""
       

    def update_legacy(self, trajectory: List) -> Dict[str, float]:
        """Legacy update method for backward compatibility."""
       
    # Alias for backward compatibility
    def update(self, trajectory: List) -> Dict[str, float]:
        """Main update method - uses optimized version by default."""
       
    @lru_cache(maxsize=1)
    def _get_model_info(self) -> Dict[str, int]:
        """Cached model information."""
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        
    def save_optimized(self, path: str, include_optimizer: bool = False):
        """Optimized model saving with optional optimizer state."""
       

    def load_optimized(self, path: str, load_optimizer: bool = False):
        """Optimized model loading with optional optimizer state."""
       

    # Legacy methods for backward compatibility
    def save(self, path: str):
        """Legacy save method."""
       
    def load(self, path: str):
        """Legacy load method."""
        
    def set_eval_mode(self):
        """Set model to evaluation mode for inference."""
       

    def set_train_mode(self):
        """Set model to training mode."""
       

    def clear_replay_buffer(self):
        """Clear the replay buffer to free memory."""
 


# Factory functions for easy instantiation
def create_optimized_qubit_tuner(state_dim: int, action_dim: int,
                                lr: float = 1e-3, gamma: float = 0.99,
                                use_gpu: bool = True) -> HighPerformanceQubitRLTuner:
    """Create an optimized qubit RL tuner."""
    


# Backward compatibility aliases
QubitEnv = OptimizedQubitEnv
ActorCritic = EfficientActorCritic
QubitRLTuner = HighPerformanceQubitRLTuner

