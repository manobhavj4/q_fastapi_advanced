import numpy as np
import logging
from typing import Optional, Union, Tuple
from functools import lru_cache
from dataclasses import dataclass
import warnings

# Optional: Use ML models for fitting or prediction
try:
    from sklearn.ensemble import RandomForestRegressor
    from joblib import load
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML functionality disabled.")



# Physical constants (pre-computed for performance)
KB = 1.380649e-23  # Boltzmann constant (J/K)
FOUR_KB = 4 * KB   # Pre-computed for thermal noise formula

@dataclass(frozen=True)
class NoiseParameters:
    """Immutable parameters for thermal noise calculation."""
 
    
    def __post_init__(self):
        """Validate parameters at initialization."""
      


class ThermalNoiseEstimator:
    """
    High-performance thermal noise estimator combining analytical and ML approaches.
    Optimized for quantum device applications with minimal computational overhead.
    """

    __slots__ = ('_params', '_ml_model', '_physics_noise_cache', '_alpha_cache')

    def __init__(self, temperature: float = 10e-3, resistance: float = 50, 
                 bandwidth: float = 1e6, ml_model_path: Optional[str] = None):
        """
        Initialize the thermal noise estimator.

        Args:
            temperature: Temperature in Kelvin (e.g., 10 mK)
            resistance: Resistance in ohms (e.g., 50 ohm)
            bandwidth: Bandwidth in Hz (e.g., 1 MHz)
            ml_model_path: Optional path to trained ML model
        """
    

    def _load_ml_model(self, model_path: str) -> None:
        """Load ML model with error handling."""
      

    @property
    def physics_noise(self) -> float:
        """
        Cached physics-based thermal noise calculation.
        Uses Johnson–Nyquist formula: P = 4kTRΔf
        """
       

    def predict_ml_noise(self, features: np.ndarray) -> Optional[float]:
        """
        Vectorized ML prediction with optimized error handling.

        Args:
            features: Feature vector for ML prediction

        Returns:
            Predicted noise power or None if unavailable
        """
      

    def estimate_combined_noise(self, features: Optional[np.ndarray] = None, 
                              alpha: float = 0.5) -> float:
        """
        High-performance combined noise estimation.

        Args:
            features: Input features for ML prediction
            alpha: Weight for ML prediction (0 = physics only, 1 = ML only)

        Returns:
            Combined estimated noise power
        """
        
        
        # Fast path: no ML model or features
     

    def batch_estimate(self, features_batch: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Vectorized batch processing for multiple feature sets.

        Args:
            features_batch: 2D array where each row is a feature vector
            alpha: Weight for ML prediction

        Returns:
            Array of noise estimates
        """
        

    def update_parameters(self, temperature: Optional[float] = None,
                         resistance: Optional[float] = None,
                         bandwidth: Optional[float] = None) -> None:
        """
        Update parameters and invalidate cache.

        Args:
            temperature: New temperature value
            resistance: New resistance value
            bandwidth: New bandwidth value
        """
    

    def get_noise_breakdown(self, features: Optional[np.ndarray] = None, 
                           alpha: float = 0.5) -> dict:
        """
        Get detailed breakdown of noise components for analysis.

        Returns:
            Dictionary with noise components and metadata
        """
      

    @staticmethod
    @lru_cache(maxsize=128)
    def theoretical_noise_limit(temperature: float, bandwidth: float) -> float:
        """
        Calculate theoretical minimum noise for given temperature and bandwidth.
        Cached for frequently used values.

        Args:
            temperature: Temperature in Kelvin
            bandwidth: Bandwidth in Hz

        Returns:
            Theoretical noise limit in watts
        """
      

    def __repr__(self) -> str:
        return (f"ThermalNoiseEstimator(T={self._params.temperature:.2e}K, "
                f"R={self._params.resistance}Ω, BW={self._params.bandwidth:.2e}Hz, "
                f"ML={'Yes' if self._ml_model else 'No'})")