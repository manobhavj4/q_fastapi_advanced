import os
import torch
import onnx
import onnxruntime as ort
import logging
import numpy as np
from typing import Union, List, Dict, Optional, Any
from pathlib import Path
from functools import lru_cache
import threading
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger("ModelManager")
logging.basicConfig(level=logging.INFO)

class OptimizedModelManager:
    """
    High-performance model manager with optimizations for edge deployment.
    
    Features:
    - Lazy loading for faster initialization
    - Memory-efficient caching
    - GPU/CPU optimization
    - Thread-safe operations
    - Batch processing support
    - Performance monitoring
    """
    
    def __init__(self, 
                 model_path: str, 
                 model_type: str = "torch", 
                 device: str = "auto",
                 enable_optimization: bool = True,
                 batch_size: int = 1,
                 cache_size: int = 128):
        """
        Initialize the optimized model manager.
        
        Args:
            model_path: Path to the saved model file
            model_type: 'torch' or 'onnx'
            device: 'cpu', 'cuda', or 'auto' for automatic selection
            enable_optimization: Enable various performance optimizations
            batch_size: Default batch size for inference
            cache_size: Size of LRU cache for repeated inputs
        """
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.device = self._select_device(device)
        self.enable_optimization = enable_optimization
        self.batch_size = batch_size
        self.cache_size = cache_size
        
        # Model objects (lazy loaded)
        self.model = None
        self.session = None
        self._is_loaded = False
        self._lock = threading.Lock()
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # Cache for model metadata
        self._input_names = None
        self._output_names = None
        self._input_shapes = None
        
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        logger.info(f"🚀 Initialized OptimizedModelManager for {self.model_type} model")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Optimizations: {'Enabled' if enable_optimization else 'Disabled'}")
    
    def _select_device(self, device: str) -> str:
        """Auto-select optimal device if 'auto' is specified."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_onnx_providers(self) -> List[str]:
        """Get optimal ONNX Runtime providers based on device."""
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the model with optimizations (thread-safe).
        
        Args:
            force_reload: Force reload even if already loaded
        """
        with self._lock:
            if self._is_loaded and not force_reload:
                return
            
            try:
                if self.model_type == "torch":
                    self._load_torch_model()
                elif self.model_type == "onnx":
                    self._load_onnx_model()
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")
                
                self._is_loaded = True
                logger.info(f"✅ Model loaded successfully from {self.model_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise
    
    def _load_torch_model(self) -> None:
        """Load PyTorch model with optimizations."""
        logger.info(f"🔹 Loading PyTorch model from {self.model_path}")
        
        # Load model
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()
        
        if self.enable_optimization:
            # Apply PyTorch optimizations
            self.model = torch.jit.optimize_for_inference(self.model)
            
            # Enable cuDNN benchmarking if using CUDA
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            
            logger.info("🚀 PyTorch optimizations applied")
    
    def _load_onnx_model(self) -> None:
        """Load ONNX model with optimizations."""
        logger.info(f"🔹 Loading ONNX model from {self.model_path}")
        
        # Validate ONNX model (optional, can be disabled for performance)
        if self.enable_optimization:
            onnx_model = onnx.load(str(self.model_path))
            onnx.checker.check_model(onnx_model)
        
        # Configure session options for optimization
        session_options = ort.SessionOptions()
        if self.enable_optimization:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Create inference session
        providers = self._get_onnx_providers()
        self.session = ort.InferenceSession(
            str(self.model_path), 
            sess_options=session_options,
            providers=providers
        )
        
        # Cache model metadata
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]
        self._input_shapes = [inp.shape for inp in self.session.get_inputs()]
        
        logger.info(f"🚀 ONNX model loaded with providers: {providers}")
    
    @lru_cache(maxsize=128)
    def _cached_predict(self, input_hash: int, input_data: Any) -> Any:
        """Cached prediction for repeated inputs."""
        return self._predict_internal(input_data)
    
    def _predict_internal(self, input_data: Any) -> Any:
        """Internal prediction method."""
        if self.model_type == "torch":
            return self._predict_torch(input_data)
        elif self.model_type == "onnx":
            return self._predict_onnx(input_data)
    
    def _predict_torch(self, input_data: torch.Tensor) -> torch.Tensor:
        """PyTorch prediction with optimizations."""
        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Expected torch.Tensor for PyTorch model")
        
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()
        
        # Move to device and optimize tensor
        input_data = input_data.to(self.device, non_blocking=True)
        
        # Use autocast for mixed precision if available
        with torch.no_grad():
            if self.device == "cuda" and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast():
                    output = self.model(input_data)
            else:
                output = self.model(input_data)
        
        return output
    
    def _predict_onnx(self, input_data: Union[Dict[str, np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """ONNX prediction with optimizations."""
        # Ensure model is loaded
        if not self._is_loaded:
            self.load_model()
        
        # Handle different input formats
        if isinstance(input_data, np.ndarray):
            if self._input_names is None:
                raise ValueError("Input names not available")
            input_dict = {self._input_names[0]: input_data}
        elif isinstance(input_data, dict):
            input_dict = input_data
        else:
            raise TypeError("Expected numpy array or dict for ONNX model")
        
        # Run inference
        result = self.session.run(self._output_names, input_dict)
        return result
    
    def predict(self, 
                input_data: Union[torch.Tensor, np.ndarray, Dict[str, np.ndarray]], 
                use_cache: bool = True) -> Union[torch.Tensor, List[np.ndarray]]:
        """
        Run optimized inference.
        
        Args:
            input_data: Input tensor/array or dict
            use_cache: Whether to use caching for repeated inputs
            
        Returns:
            Model prediction
        """
        import time
        start_time = time.time()
        
        try:
            # Use caching if enabled and input is hashable
            if use_cache and self.cache_size > 0:
                try:
                    input_hash = hash(input_data.tobytes() if hasattr(input_data, 'tobytes') else str(input_data))
                    result = self._cached_predict(input_hash, input_data)
                except (TypeError, AttributeError):
                    # Fallback to non-cached prediction
                    result = self._predict_internal(input_data)
            else:
                result = self._predict_internal(input_data)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def predict_batch(self, 
                     batch_data: List[Union[torch.Tensor, np.ndarray]], 
                     batch_size: Optional[int] = None) -> List[Any]:
        """
        Batch inference for improved throughput.
        
        Args:
            batch_data: List of input tensors/arrays
            batch_size: Override default batch size
            
        Returns:
            List of predictions
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            
            if self.model_type == "torch":
                # Stack tensors for batch processing
                stacked_input = torch.stack(batch)
                batch_result = self.predict(stacked_input, use_cache=False)
                results.extend(torch.unbind(batch_result))
            else:
                # Process individually for ONNX (can be optimized further)
                for item in batch:
                    results.append(self.predict(item, use_cache=False))
        
        return results
    
    @contextmanager
    def performance_monitor(self):
        """Context manager for performance monitoring."""
       
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.inference_count == 0:
            return {"total_inferences": 0, "avg_inference_time": 0.0}
        
        avg_time = self.total_inference_time / self.inference_count
        return {
            "total_inferences": self.inference_count,
            "total_time": self.total_inference_time,
            "avg_inference_time": avg_time,
            "inferences_per_second": 1.0 / avg_time if avg_time > 0 else 0.0
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._is_loaded:
            self.load_model()
        
        info = {
            "model_type": self.model_type,
            "device": self.device,
            "model_path": str(self.model_path),
            "optimizations_enabled": self.enable_optimization
        }
        
        if self.model_type == "onnx":
            info.update({
                "input_names": self._input_names,
                "output_names": self._output_names,
                "input_shapes": self._input_shapes
            })
        
        return info
    
    def get_model(self) -> Union[torch.nn.Module, ort.InferenceSession]:
        """Get the underlying model object."""
        if not self._is_loaded:
            self.load_model()
        
        return self.model if self.model_type == "torch" else self.session
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        with self._lock:
            if self.model_type == "torch" and self.model is not None:
                del self.model
                self.model = None
            elif self.model_type == "onnx" and self.session is not None:
                del self.session
                self.session = None
            
            self._is_loaded = False
            
            # Clear cache
            self._cached_predict.cache_clear()
            
            logger.info("🗑️  Model unloaded and memory freed")
    
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.unload_model()
        except:
            pass  # Ignore errors during cleanup


# Factory function for easy model creation
def create_model_manager(model_path: str, 
                        model_type: str = "auto",
                        device: str = "auto",
                        **kwargs) -> OptimizedModelManager:
    """
    Factory function to create an optimized model manager.
    
    Args:
        model_path: Path to model file
        model_type: 'torch', 'onnx', or 'auto' for auto-detection
        device: Device to use
        **kwargs: Additional arguments for ModelManager
        
    Returns:
        OptimizedModelManager instance
    """
    if model_type == "auto":
        path = Path(model_path)
        if path.suffix == ".onnx":
            model_type = "onnx"
        elif path.suffix in [".pt", ".pth"]:
            model_type = "torch"
        else:
            raise ValueError(f"Cannot auto-detect model type from {path.suffix}")
    
    return OptimizedModelManager(
        model_path=model_path,
        model_type=model_type,
        device=device,
        **kwargs
    )



""" For PyTorch model:

manager = ModelManager("models/lstm_drift.pt", model_type="torch", device="cpu")
x = torch.randn(1, 10, 8)  # Example input
output = manager.predict(x)

For ONNX model:

manager = ModelManager("models/lstm_drift.onnx", model_type="onnx")
input_dict = {"input": x.numpy()}  # x is a NumPy array
output = manager.predict(input_dict)
 """