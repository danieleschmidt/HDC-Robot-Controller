"""
HDC Robot Controller - Hyperdimensional Computing for Robust Robotic Control

High-performance, fault-tolerant robotic control using hyperdimensional computing.
"""

from .core.hypervector import HyperVector, HyperVectorSpace
from .core.operations import HDCOperations
from .core.memory import AssociativeMemory
from .controllers.base import HDCController
from .encoding.multimodal import MultiModalEncoder
from .learning.behavior import BehaviorLearner
from .utils.config import HDCConfig

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.ai"

__all__ = [
    "HyperVector",
    "HyperVectorSpace", 
    "HDCOperations",
    "AssociativeMemory",
    "HDCController",
    "MultiModalEncoder",
    "BehaviorLearner",
    "HDCConfig",
]