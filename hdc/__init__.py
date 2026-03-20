"""HDC: Hyperdimensional Computing for robot control."""

from .vectors import HyperdimensionalVector
from .memory import ItemMemory
from .encoder import SensorEncoder
from .classifier import OneShotClassifier

__all__ = [
    "HyperdimensionalVector",
    "ItemMemory",
    "SensorEncoder",
    "OneShotClassifier",
]

__version__ = "0.1.0"
