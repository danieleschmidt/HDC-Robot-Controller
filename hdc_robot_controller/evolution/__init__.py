"""
Generation 6: Evolutionary Breakthrough Module
============================================

Advanced evolutionary algorithms for self-improving robotic intelligence.
Implements genetic programming, neural architecture search, and adaptive optimization.
"""

from .genetic_optimizer import GeneticOptimizer, EvolutionaryStrategy
from .adaptive_architecture import AdaptiveNeuralArchitecture, NeuralEvolution
from .self_improving_algorithms import SelfImprovingHDC, AlgorithmicEvolution
from .quantum_genetic_algorithms import QuantumGeneticOptimizer
from .evolutionary_controller import EvolutionaryController

__all__ = [
    'GeneticOptimizer',
    'EvolutionaryStrategy', 
    'AdaptiveNeuralArchitecture',
    'NeuralEvolution',
    'SelfImprovingHDC',
    'AlgorithmicEvolution',
    'QuantumGeneticOptimizer',
    'EvolutionaryController'
]