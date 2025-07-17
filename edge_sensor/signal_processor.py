import numpy as np
import scipy.signal as signal
from scipy.stats import zscore
from typing import Tuple, Union, Optional, Dict, Any
import warnings


class SignalProcessor:
    """
    High-performance signal processing class with optimized algorithms.
    
    Supports filtering, smoothing, normalization, outlier removal, and windowing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SignalProcessor with configuration.
        
        Args:
            config: Dictionary with processing parameters
        """
        self.config = {
            "filter_type": "butterworth",
            "filter_order": 4,
            "cutoff_freq": 0.1,
            "sampling_rate": 100,
            "smoothing_window": 5,
            "normalization": "zscore",
            "remove_outliers": True,
            "zscore_threshold": 3.0,
            **config
        } if config else {
            "filter_type": "butterworth",
            "filter_order": 4,
            "cutoff_freq": 0.1,
            "sampling_rate": 100,
            "smoothing_window": 5,
            "normalization": "zscore",
            "remove_outliers": True,
            "zscore_threshold": 3.0
        }
        
        # Pre-compute filter coefficients for reuse
        self._filter_coeffs = None
        self._prepare_filter()
    
    def _prepare_filter(self) -> None:
        """Pre-compute filter coefficients for better performance."""
       
    
    @staticmethod
    def apply_fft(signal_data: np.ndarray, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT and return frequency and magnitude spectrum.
        
        Args:
            signal_data: Time-domain signal
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (frequencies, magnitude spectrum)
        """

    
    def apply_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply pre-computed low-pass Butterworth filter.
        
        Args:
            data: Input signal array
            
        Returns:
            Filtered signal array
        """
   
    
    def apply_moving_average(self, data: np.ndarray) -> np.ndarray:
        """
        Apply optimized moving average smoothing using uniform filter.
        
        Args:
            data: Input signal array
            
        Returns:
            Smoothed signal array
        """
    
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using z-score or min-max normalization.
        
        Args:
            data: Input signal array
            
        Returns:
            Normalized signal array
        """
      
    
    def remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Remove outliers using z-score thresholding with optimized implementation.
        
        Args:
            data: Input signal array
            
        Returns:
            Signal array with outliers removed
        """
        
    
    def apply_windowing(self, data: np.ndarray, window_size: int, overlap: float = 0.5) -> np.ndarray:
        """
        Split signal into overlapping windows using vectorized operations.
        
        Args:
            data: Input signal array
            window_size: Size of each window
            overlap: Overlap fraction between windows (0-1)
            
        Returns:
            2D array of windowed signals
        """
      
    
    def process(self, data: Union[list, np.ndarray]) -> np.ndarray:
        """
        Apply full processing pipeline with optimized order.
        
        Args:
            data: Input signal (list or numpy array)
            
        Returns:
            Processed signal array
        """
        # Convert to numpy array if needed
       
    
    def process_batch(self, data_batch: list) -> list:
        """
        Process multiple signals efficiently.
        
        Args:
            data_batch: List of signal arrays
            
        Returns:
            List of processed signal arrays
        """
  
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
   


# Example usage and performance demonstration
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    # Generate test signal
    np.random.seed(42)
    t = np.linspace(0, 10, 10000)
    raw_signal = (np.sin(2 * np.pi * 5 * t) + 
                  0.5 * np.sin(2 * np.pi * 15 * t) + 
                  np.random.normal(0, 0.3, len(t)))
    
    # Add some outliers

    
    # Create processor with optimized settings
