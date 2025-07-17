
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import GradScaler, autocast
from typing import Tuple, Dict, Any
import time
import numpy as np
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

from models.qec_decoder_cnn import QECDecoderCNN
from ai_control_engine.utils import set_seed, load_json_data
from ai_control_engine.config.config import QEC_DECODER_TRAIN_CONFIG as cfg
from ai_control_engine.registry.model_registry import save_model

import mlflow
import mlflow.pytorch

# Optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Set reproducibility
set_seed(cfg["seed"])


class OptimizedDataLoader:
    """High-performance data loading with caching and prefetching."""
    
    def __init__(self, dataset: TensorDataset, batch_size: int, shuffle: bool = True, 
                 pin_memory: bool = True, num_workers: int = 0, prefetch_factor: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Optimize num_workers based on system
        if num_workers == 0:
            num_workers = min(4, os.cpu_count() or 1)
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory and torch.cuda.is_available(),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
            drop_last=True  # Consistent batch sizes for optimization
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


@lru_cache(maxsize=2)
def load_qec_data_cached(path: str) -> TensorDataset:
    """
    Load quantum error correction data with caching for repeated access.
    Optimized for memory efficiency and speed.
    """
  


class FastTrainer:
    """Optimized training class with mixed precision and efficient operations."""
    
    def __init__(self, model: nn.Module, device: torch.device, use_amp: bool = True):
     
        
    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> float:
        """Optimized training epoch with mixed precision."""
    
    
    @torch.no_grad()  # Decorator for efficiency
    def evaluate(self, dataloader: DataLoader) -> float:
        """Optimized evaluation with vectorized operations."""
        


class OptimizedMLflowLogger:
    """Efficient MLflow logging with batched operations."""
    
    def __init__(self, use_mlflow: bool, config: Dict[str, Any]):
        self.use_mlflow = use_mlflow
        self.metrics_buffer = {}
        self.buffer_size = 10  # Log metrics in batches
        
        if self.use_mlflow:
            mlflow.start_run()
            # Log config once
            mlflow.log_params({k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))})
    
    def log_metric(self, key: str, value: float, step: int):
        """Buffer metrics for batch logging."""
        
    
    def _flush_metrics(self, key: str = None):
        """Flush buffered metrics to MLflow."""
       
    def finalize(self, model: nn.Module):
        """Flush remaining metrics and save model."""
       


def create_optimized_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create and optimize model for training."""
    


def create_optimized_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimized optimizer with better defaults."""
    # Use AdamW for better generalization and stability
    


def create_optimized_criterion() -> nn.Module:
    """Create optimized loss function."""
    # Label smoothing for better generalization
   


def optimize_data_loading(dataset: TensorDataset, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create optimized data loaders with efficient splitting."""
    
   


def main():
    """Optimized main training function."""
   

if __name__ == "__main__":
    main()


