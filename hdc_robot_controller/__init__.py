"""
HDC Robot Controller - Hyperdimensional Computing for Robust Robotic Control

This package implements hyperdimensional computing (HDC) algorithms for real-time
robot control with one-shot learning and extreme fault tolerance capabilities.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"
__license__ = "BSD-3-Clause"

# Core HDC components
from .core.hypervector import HyperVector
from .core.encoding import MultiModalEncoder, SpatialEncoder, VisualEncoder, TemporalEncoder
from .core.memory import AssociativeMemory, EpisodicMemory, WorkingMemory
from .core.operations import HDCOperations

# Control components (optional imports)
try:
    from .control.fault_tolerant_controller import FaultTolerantController
    from .control.behavior_learner import BehaviorLearner
    from .control.hdc_controller import HDCController
    CONTROL_AVAILABLE = True
except ImportError:
    FaultTolerantController = None
    BehaviorLearner = None
    HDCController = None
    CONTROL_AVAILABLE = False

# Encoding modules (use core encoders)
from .core.encoding import MultiModalEncoder, SpatialEncoder, VisualEncoder, TemporalEncoder, RoboticEncoder
SensorFusionEncoder = MultiModalEncoder  # Alias for compatibility

# Utilities (optional imports)
try:
    from .utils.benchmarks import PerformanceBenchmark, FaultToleranceBenchmark
    from .utils.visualization import HDCVisualizer
    from .utils.logger import HDCLogger
    UTILS_AVAILABLE = True
except ImportError:
    PerformanceBenchmark = None
    FaultToleranceBenchmark = None
    HDCVisualizer = None
    HDCLogger = None
    UTILS_AVAILABLE = False

# ROS 2 integration
try:
    import rclpy
    from .ros2_integration.perception_node import HDCPerceptionNode
    from .ros2_integration.control_node import HDCControlNode
    from .ros2_integration.learning_node import HDCLearningNode
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Version info
def get_version():
    """Get the version of the HDC Robot Controller package."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "hdc_robot_controller",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "ros2_available": ROS2_AVAILABLE
    }

# Global configuration
DEFAULT_DIMENSION = 10000
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_LEARNING_RATE = 0.1

class HDCConfig:
    """Global configuration for HDC operations."""
    
    def __init__(self):
        self.dimension = DEFAULT_DIMENSION
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        self.learning_rate = DEFAULT_LEARNING_RATE
        self.enable_cuda = False
        self.enable_logging = True
        self.log_level = "INFO"
        
    def set_dimension(self, dimension: int):
        """Set the global hypervector dimension."""
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        self.dimension = dimension
        
    def set_similarity_threshold(self, threshold: float):
        """Set the global similarity threshold."""
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("Similarity threshold must be between -1 and 1")
        self.similarity_threshold = threshold
        
    def set_learning_rate(self, rate: float):
        """Set the global learning rate."""
        if rate < 0.0 or rate > 1.0:
            raise ValueError("Learning rate must be between 0 and 1")
        self.learning_rate = rate

# Global config instance
config = HDCConfig()

# Convenience functions
def create_random_hypervector(dimension: int = None, seed: int = None) -> HyperVector:
    """Create a random hypervector with the specified dimension."""
    dim = dimension if dimension is not None else config.dimension
    return HyperVector.random(dim, seed)

def create_zero_hypervector(dimension: int = None) -> HyperVector:
    """Create a zero hypervector with the specified dimension."""
    dim = dimension if dimension is not None else config.dimension
    return HyperVector.zero(dim)

def bundle_hypervectors(*vectors) -> HyperVector:
    """Bundle multiple hypervectors together."""
    if not vectors:
        raise ValueError("At least one hypervector is required for bundling")
    return HyperVector.bundle_vectors(list(vectors))

def bind_hypervectors(a: HyperVector, b: HyperVector) -> HyperVector:
    """Bind two hypervectors together."""
    return a.bind(b)

def similarity(a: HyperVector, b: HyperVector) -> float:
    """Calculate similarity between two hypervectors."""
    return a.similarity(b)

# Package validation
def validate_installation():
    """Validate that the HDC Robot Controller package is properly installed."""
    try:
        # Test core HDC operations
        hv1 = create_random_hypervector(1000, seed=42)
        hv2 = create_random_hypervector(1000, seed=43)
        
        # Test bundling
        bundled = bundle_hypervectors(hv1, hv2)
        assert bundled.dimension() == 1000
        
        # Test binding
        bound = bind_hypervectors(hv1, hv2)
        assert bound.dimension() == 1000
        
        # Test similarity
        sim = similarity(hv1, hv2)
        assert -1.0 <= sim <= 1.0
        
        # Test memory
        memory = AssociativeMemory(1000)
        memory.store("test", hv1)
        assert memory.contains("test")
        
        return True
        
    except Exception as e:
        print(f"HDC Robot Controller validation failed: {e}")
        return False

# Initialize logging if available
try:
    if UTILS_AVAILABLE and HDCLogger:
        logger = HDCLogger("hdc_robot_controller")
        logger.info(f"HDC Robot Controller v{__version__} initialized")
    else:
        logger = None
except:
    logger = None

__all__ = [
    # Core classes
    'HyperVector', 'MultiModalEncoder', 'AssociativeMemory', 'EpisodicMemory',
    'WorkingMemory', 'HDCOperations', 'FaultTolerantController', 'BehaviorLearner',
    'HDCController', 'SensorFusionEncoder', 'RoboticEncoder',
    
    # Utilities
    'PerformanceBenchmark', 'FaultToleranceBenchmark', 'HDCVisualizer', 'HDCLogger',
    
    # Configuration
    'HDCConfig', 'config',
    
    # Convenience functions
    'create_random_hypervector', 'create_zero_hypervector', 'bundle_hypervectors',
    'bind_hypervectors', 'similarity', 'get_version', 'get_info', 'validate_installation',
    
    # Constants
    'DEFAULT_DIMENSION', 'DEFAULT_SIMILARITY_THRESHOLD', 'DEFAULT_LEARNING_RATE',
    'ROS2_AVAILABLE'
]

# Conditional ROS 2 exports
if ROS2_AVAILABLE:
    __all__.extend(['HDCPerceptionNode', 'HDCControlNode', 'HDCLearningNode'])