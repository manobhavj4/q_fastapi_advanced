import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
from typing import Tuple, Union, Optional
import warnings

class FFTFeatureExtractor:
    def __init__(self, sampling_rate: float, window_size: int = 1024, 
                 window_type: str = 'hann', overlap: float = 0.5):
        """
        Initialize the FFT feature extractor.

        Args:
            sampling_rate (float): Sampling frequency of the signal in Hz.
            window_size (int): Number of samples in one FFT window. Should be power of 2 for optimal performance.
            window_type (str): Type of window function ('hann', 'hamming', 'blackman', 'bartlett', 'none').
            overlap (float): Overlap ratio between consecutive windows (0.0 to 0.95).
        """
        if not self._is_power_of_two(window_size):
            warnings.warn(f"Window size {window_size} is not a power of 2. Consider using powers of 2 for optimal FFT performance.")
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_type = window_type
        self.overlap = np.clip(overlap, 0.0, 0.95)
        self.hop_length = int(window_size * (1 - overlap))
        
        # Pre-compute window function and frequencies
        self.window_func = self._get_window_function(window_type, window_size)
        self.frequencies = rfftfreq(window_size, d=1.0 / sampling_rate)
        
        # Pre-compute window normalization factor for power calculations
        self.window_norm = np.sum(self.window_func ** 2)
    
    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """Check if n is a power of 2."""
       
    
    def _get_window_function(self, window_type: str, size: int) -> np.ndarray:
        """Get the appropriate window function."""
  
    
    def extract_features(self, signal: Union[np.ndarray, list], 
                        return_phase: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract frequency-domain features from time-domain signal.

        Args:
            signal (np.ndarray or list): 1D time-domain signal array.
            return_phase (bool): Whether to return phase information along with magnitude.

        Returns:
            Tuple containing:
            - frequencies (np.ndarray): Frequency bins
            - magnitudes (np.ndarray): FFT magnitudes
            - phases (np.ndarray, optional): FFT phases if return_phase=True
        """
        
    
    def extract_features_windowed(self, signal: Union[np.ndarray, list], 
                                 return_times: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                                    Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract features using overlapping windows (STFT-like approach).
        
        Args:
            signal (np.ndarray or list): 1D time-domain signal array.
            return_times (bool): Whether to return time information for each window.
            
        Returns:
            Tuple containing:
            - frequencies (np.ndarray): Frequency bins
            - magnitudes (np.ndarray): 2D array of FFT magnitudes (time x frequency)
            - times (np.ndarray, optional): Time centers of each window if return_times=True
        """
       
    
    def extract_band_power(self, fft_freq: np.ndarray, fft_mag: np.ndarray,
                          band: Tuple[float, float], normalize: bool = True) -> float:
        """
        Compute the power of a specific frequency band.

        Args:
            fft_freq (np.ndarray): Frequencies from FFT.
            fft_mag (np.ndarray): Magnitudes from FFT.
            band (tuple): (low_freq, high_freq) tuple in Hz.
            normalize (bool): Whether to normalize by window function power.

        Returns:
            float: Power in the specified band.
        """
       
    
    def extract_multiple_band_powers(self, fft_freq: np.ndarray, fft_mag: np.ndarray,
                                   bands: list, normalize: bool = True) -> np.ndarray:
        """
        Compute power for multiple frequency bands efficiently.
        
        Args:
            fft_freq (np.ndarray): Frequencies from FFT.
            fft_mag (np.ndarray): Magnitudes from FFT.
            bands (list): List of (low_freq, high_freq) tuples in Hz.
            normalize (bool): Whether to normalize by window function power.
            
        Returns:
            np.ndarray: Array of powers for each band.
        """
        
    
    def get_peak_frequency(self, fft_freq: np.ndarray, fft_mag: np.ndarray,
                          freq_range: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Find the peak frequency and its magnitude.
        
        Args:
            fft_freq (np.ndarray): Frequencies from FFT.
            fft_mag (np.ndarray): Magnitudes from FFT.
            freq_range (tuple, optional): (low_freq, high_freq) to limit search range.
            
        Returns:
            Tuple[float, float]: (peak_frequency, peak_magnitude)
        """
        
    
    def get_spectral_centroid(self, fft_freq: np.ndarray, fft_mag: np.ndarray) -> float:
        """
        Calculate the spectral centroid (center of mass of spectrum).
        
        Args:
            fft_freq (np.ndarray): Frequencies from FFT.
            fft_mag (np.ndarray): Magnitudes from FFT.
            
        Returns:
            float: Spectral centroid in Hz.
        """
      