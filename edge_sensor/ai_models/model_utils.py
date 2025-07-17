import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)


# Import with better error handling


class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def load_model(self, model_path: Path):
        """Load the model from file."""
      
    
    @abstractmethod
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
     
    
    @abstractmethod
    def get_input_shape(self) -> Union[Tuple, str]:
        """Get input shape information."""
     


class ONNXBackend(ModelBackend):
    """ONNX Runtime backend."""
    
    def __init__(self):
        if not ORT_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        self.session = None
        self.input_name = None
    
    def load_model(self, model_path: Path):
        """Load ONNX model with optimized session options."""
      
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
       
    
    def get_input_shape(self) -> Tuple:
        """Get ONNX input shape."""
       


class TFLiteBackend(ModelBackend):
    """TensorFlow Lite backend."""
    
    def __init__(self):
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite not available")
        self.interpreter = None
        self.input_details = None
        self.output_details = None
    
    def load_model(self, model_path: Path):
        """Load TFLite model with optimized settings."""
      
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run TFLite inference."""
  
    
    def get_input_shape(self) -> Tuple:
        """Get TFLite input shape."""
      


class TorchScriptBackend(ModelBackend):
    """TorchScript backend."""
    
    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        self.model = None
    
    def load_model(self, model_path: Path):
        """Load TorchScript model."""
        
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run TorchScript inference."""

    
    def get_input_shape(self) -> str:
        """Get TorchScript input shape (not statically determinable)."""
       


class ModelLoader:
    """Optimized model loader with backend abstraction."""
    
    BACKEND_MAP = {
        'onnx': ONNXBackend,
        'tflite': TFLiteBackend,
        'torchscript': TorchScriptBackend
    }
    
    def __init__(self, model_path: Union[str, Path], model_type: str = "onnx"):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ['onnx', 'tflite', 'torchscript']
        """
     
    
    def _validate_inputs(self):
        """Validate model path and type."""
      
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded model.
        
        Args:
            input_data: Input data as numpy array (preprocessed)
            
        Returns:
            Model predictions as numpy array
        """
   
    
    def get_input_shape(self) -> Union[Tuple, str]:
        """Get expected input shape for the model."""
     
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
       


# Utility functions
def normalize_input(data: np.ndarray, method: str = "min_max") -> np.ndarray:
    """
    Normalize input data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ['min_max', 'z_score', 'unit_norm']
    
    Returns:
        Normalized data array
    """



def batch_inference(model: ModelLoader, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """
    Run inference on large datasets in batches.
    
    Args:
        model: Loaded model instance
        data: Input data array (first dimension is batch)
        batch_size: Number of samples per batch
        
    Returns:
        Concatenated predictions
    """
  


# Example usage and testing
if __name__ == "__main__":
    # Create dummy input
    dummy_input = np.random.rand(1, 128).astype(np.float32)
    
    # Example usage (uncomment to test with actual models)
    """
    try:
        # ONNX example
        model = ModelLoader("models/anomaly_detector.onnx", model_type="onnx")
        print("Model info:", model.get_model_info())
        
        # Run inference
        output = model.run_inference(dummy_input)
        print("Prediction shape:", output.shape)
        
        # Batch inference example
        large_input = np.random.rand(100, 128).astype(np.float32)
        batch_output = batch_inference(model, large_input, batch_size=16)
        print("Batch prediction shape:", batch_output.shape)
        
    except Exception as e:
        logger.error(f"Error during model loading/inference: {e}")
    """
    