# anomaly_detector.py

import numpy as np
from sklearn.ensemble import IsolationForest
from joblib import load, dump
import logging
import os
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod

# Optional: Use ONNX Runtime or TensorFlow Lite if models are deployed in edge-optimized formats
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None

# Configure logging once at module level



class ModelHandler(ABC):
    """Abstract base class for different model handlers."""
    
    @abstractmethod
    def load_model(self, model_path: str):
      
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        


class SklearnHandler(ModelHandler):
    """Handler for scikit-learn models."""
    
    def __init__(self):
    
    
    def load_model(self, model_path: str):
    

    
    def predict(self, X: np.ndarray) -> np.ndarray:
       


class ONNXHandler(ModelHandler):
    """Handler for ONNX models."""
    
    def __init__(self):
        if ort is None:
            raise ImportError("onnxruntime is not installed.")
        self.model = None
        self.input_name = None
    
    def load_model(self, model_path: str):
      
    
    def predict(self, X: np.ndarray) -> np.ndarray:
     


class TFLiteHandler(ModelHandler):
    """Handler for TensorFlow Lite models."""
    
    def __init__(self):
        if tflite is None:
            raise ImportError("tflite_runtime is not installed.")
        self.model = None
        self.input_details = None
        self.output_details = None
    
    def load_model(self, model_path: str):
   
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Vectorized prediction for better performance
      


class AnomalyDetector:
    """Optimized anomaly detector with multiple backend support."""
    
    SUPPORTED_TYPES = {"sklearn", "onnx", "tflite"}
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "sklearn"):
        """
        Initialize the anomaly detector.
        
        :param model_path: Path to the pre-trained model
        :param model_type: One of ["sklearn", "onnx", "tflite"]
        """
      
    
    def _create_handler(self) -> ModelHandler:
        """Factory method to create appropriate model handler."""
      
    
    def _validate_model_path(self):
        """Validate that the model path exists."""
   
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and preprocess input data."""

    
    def train(self, X_train: np.ndarray, save_path: Optional[str] = None, 
              **model_params) -> 'AnomalyDetector':
        """
        Train an Isolation Forest model with optimized parameters.
        
        :param X_train: 2D numpy array of features
        :param save_path: Path to save the model
        :param model_params: Additional parameters for IsolationForest
        :return: Self for method chaining
        """
    
        
    
    def save_model(self, save_path: str):
        """Save the trained model."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies with input validation.
        
        :param X: 2D numpy array of input features
        :return: An array of anomaly flags: 1 (normal), -1 (anomaly)
        """

    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (sklearn only).
        
        :param X: 2D numpy array of input features
        :return: Anomaly scores (higher = more anomalous)
        """
   
    
    def is_anomaly(self, x: Union[np.ndarray, list]) -> bool:
        """
        Check if a single data point is anomalous.
        
        :param x: 1D feature vector
        :return: True if anomaly, False otherwise
        """

    def detect_batch(self, X: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in batch with optional threshold.
        
        :param X: 2D numpy array of input features
        :param threshold: Custom threshold for anomaly detection (sklearn only)
        :return: Tuple of (predictions, scores if available)
        """
    
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
       


# Example usage (commented for production)
if __name__ == "__main__":
    