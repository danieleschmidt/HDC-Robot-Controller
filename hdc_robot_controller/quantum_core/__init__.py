"""
Generation 6: Quantum Core Module
Native quantum computing integration for HDC Robot Controller.
"""

from .quantum_hdc_engine import (
    QuantumHDCEngine,
    QuantumClassicalHybrid,
    QuantumState,
    QuantumOperation,
    QuantumGate,
    demonstrate_quantum_hdc
)

__all__ = [
    'QuantumHDCEngine',
    'QuantumClassicalHybrid', 
    'QuantumState',
    'QuantumOperation',
    'QuantumGate',
    'demonstrate_quantum_hdc'
]