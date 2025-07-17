# quantum_control/circuits/circuit_templates.py

def get_default_template() -> dict:
    """
    Returns a default quantum circuit template.
    """



from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, RXGate, RYGate, RZGate
from qiskit.circuit import Parameter, ParameterVector
import numpy as np
from typing import List, Optional, Union, Dict, Any
from functools import lru_cache
from global_services.get_global_context import logger
# import logging

# logger = logging.getLogger(__name__)


class CircuitTemplates:
    """Optimized quantum circuit templates with caching and enhanced functionality."""
    
    # Circuit name constants for better maintainability
    BELL_STATE = "bell_state"
    QFT = "qft"
    DATA_ENCODING = "data_encoding"
    PARAMETERIZED_ANSATZ = "parameterized_ansatz"
    
    def __init__(self):
        """Initialize with circuit registry for caching."""
        self._circuit_cache: Dict[str, QuantumCircuit] = {}
        self._parameter_cache: Dict[str, ParameterVector] = {}
    
    @staticmethod
    @lru_cache(maxsize=32)
    def bell_pair(measure: bool = True) -> QuantumCircuit:
        """
        Generates a 2-qubit Bell state circuit: (|00⟩ + |11⟩)/√2
        
        Args:
            measure: Whether to include measurement operations
            
        Returns:
            QuantumCircuit: Bell state circuit
        """
       
    
    @staticmethod
    @lru_cache(maxsize=16)
    def qft_circuit(n_qubits: int, inverse: bool = False, 
                   do_swaps: bool = True) -> QuantumCircuit:
        """
        Generates a Quantum Fourier Transform circuit for `n_qubits`
        
        Args:
            n_qubits: Number of qubits
            inverse: Whether to generate inverse QFT
            do_swaps: Whether to include swap operations
            
        Returns:
            QuantumCircuit: QFT circuit
        """
     
    
    @staticmethod
    def data_encoding_circuit(data_vector: Union[List[float], np.ndarray], 
                            encoding_type: str = "rx") -> QuantumCircuit:
        """
        Encodes classical data into quantum state using rotation gates
        
        Args:
            data_vector: Data to encode
            encoding_type: Type of encoding ('rx', 'ry', 'rz', 'mixed')
            
        Returns:
            QuantumCircuit: Data encoding circuit
        """
        
    
    
    def parameterized_ansatz(self, num_qubits: int, 
                           layers: int = 1,
                           entanglement: str = "linear",
                           rotation_gates: List[str] = None,
                           use_parameters: bool = True) -> QuantumCircuit:
        """
        Enhanced parameterized circuit for variational algorithms
        
        Args:
            num_qubits: Number of qubits
            layers: Number of layers in the ansatz
            entanglement: Entanglement pattern ('linear', 'circular', 'full')
            rotation_gates: List of rotation gates to use ['rx', 'ry', 'rz']
            use_parameters: Whether to use symbolic parameters
            
        Returns:
            QuantumCircuit: Parameterized ansatz circuit
        """
   
        
  
    
    @staticmethod
    def _add_entanglement_layer(qc: QuantumCircuit, num_qubits: int, 
                              entanglement: str):
        """Add entanglement layer to circuit."""
  
    
    @staticmethod
    def measurement_circuit(n_qubits: int, 
                          classical_bits: Optional[int] = None) -> QuantumCircuit:
        """
        Creates a measurement circuit for n qubits
        
        Args:
            n_qubits: Number of qubits
            classical_bits: Number of classical bits (defaults to n_qubits)
            
        Returns:
            QuantumCircuit: Measurement circuit
        """
      
    
    def create_vqe_ansatz(self, num_qubits: int, 
                         params: Optional[List[float]] = None) -> QuantumCircuit:
        """
        Create a VQE-optimized ansatz circuit
        
        Args:
            num_qubits: Number of qubits
            params: Parameter values (optional)
            
        Returns:
            QuantumCircuit: VQE ansatz circuit
        """
        # Hardware-efficient ansatz for VQE
      
    
    def clear_cache(self):
        """Clear internal caches."""
       
    
    def get_circuit_info(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Get detailed information about a circuit
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            Dict containing circuit information
        """
    


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Initialize templates
   