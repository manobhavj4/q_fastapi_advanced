import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, fidelity, purity, tracedist, Bloch
from typing import Optional, Union, List, Tuple
import logging
from functools import wraps
from global_services.get_global_context import logger
# logger = logging.getLogger("QuantumUtils")

# -------------------------------------------
# Decorators and Utilities
# -------------------------------------------
def quantum_error_handler(func):
    """Decorator for consistent error handling in quantum operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
      

def validate_quantum_state(state: Union[Qobj, np.ndarray]) -> bool:
    """Validate if input is a valid quantum state."""
  

# -------------------------------------------
# Quantum State Metrics
# -------------------------------------------
class QuantumMetrics:
    """Optimized quantum state metrics calculator."""
    
    @staticmethod
    @quantum_error_handler
    def calculate_fidelity(state1: Qobj, state2: Qobj) -> Optional[float]:
        """Calculate quantum fidelity between two states."""
    
    
    @staticmethod
    @quantum_error_handler
    def calculate_trace_distance(rho1: Qobj, rho2: Qobj) -> Optional[float]:
        """Calculate trace distance between two density matrices."""
       
    
    @staticmethod
    @quantum_error_handler
    def calculate_purity(rho: Qobj) -> Optional[float]:
        """Calculate purity of a quantum state. Purity = Tr(rho^2)"""
       
    
    @staticmethod
    @quantum_error_handler
    def calculate_entanglement_measure(rho: Qobj) -> Optional[float]:
        """Calculate entanglement measure (von Neumann entropy)."""
   
    
    @staticmethod
    def batch_calculate_fidelity(states1: List[Qobj], states2: List[Qobj]) -> np.ndarray:
        """Calculate fidelity for multiple state pairs efficiently."""


# -------------------------------------------
# Visualization Class
# -------------------------------------------
class QuantumVisualizer:
    """Optimized quantum state visualization tools."""
    
    @staticmethod
    def __init__():
        # Configure matplotlib for better performance
   
    
    @staticmethod
    @quantum_error_handler
    def plot_bloch_vector(state: Qobj, title: str = "Bloch Sphere", 
                         save_path: Optional[str] = None) -> None:
        """Plot a single qubit state on the Bloch sphere."""
       
    
    @staticmethod
    @quantum_error_handler
    def plot_multiple_bloch_states(states: List[Qobj], labels: Optional[List[str]] = None,
                                  title: str = "Multiple States") -> None:
        """Plot multiple quantum states on the same Bloch sphere."""
      
    
    @staticmethod
    @quantum_error_handler
    def plot_fidelity_heatmap(fidelity_matrix: np.ndarray, 
                            time_labels: Optional[List] = None,
                            sensor_labels: Optional[List] = None,
                            title: str = "Fidelity Heatmap Over Time",
                            save_path: Optional[str] = None) -> None:
        """Plot optimized 2D heatmap of fidelity values."""
       
        

    
    @staticmethod
    @quantum_error_handler
    def plot_fidelity_timeseries(fidelity_data: Union[List, np.ndarray],
                               timestamps: Optional[Union[List, np.ndarray]] = None,
                               labels: Optional[List[str]] = None,
                               title: str = "Fidelity Over Time",
                               save_path: Optional[str] = None) -> None:
        """Optimized line plot of fidelity vs time with multiple series support."""
        
        # Handle single series or multiple series
      
        
      
    
    @staticmethod
    @quantum_error_handler
    def plot_state_evolution(states: List[Qobj], time_points: Optional[List] = None,
                           title: str = "Quantum State Evolution") -> None:
        """Plot the evolution of quantum state metrics over time."""
        
        
  

# -------------------------------------------
# Convenience Functions (backward compatibility)
# -------------------------------------------
calculate_fidelity = QuantumMetrics.calculate_fidelity
calculate_trace_distance = QuantumMetrics.calculate_trace_distance
calculate_purity = QuantumMetrics.calculate_purity
plot_bloch_vector = QuantumVisualizer.plot_bloch_vector
plot_fidelity_heatmap = QuantumVisualizer.plot_fidelity_heatmap
plot_fidelity_timeseries = QuantumVisualizer.plot_fidelity_timeseries