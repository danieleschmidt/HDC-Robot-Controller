#!/usr/bin/env python3
"""
TERRAGON SDLC v4.0 - AUTONOMOUS SOFTWARE DEVELOPMENT LIFECYCLE

A comprehensive autonomous development framework implementing:
- Intelligent analysis and dynamic planning
- Progressive enhancement through generations  
- Hypothesis-driven development with statistical validation
- Comprehensive quality gates with auto-fixing
- Global-first implementation with i18n and compliance
- Self-improving patterns with adaptive optimization
- Research-grade experimental frameworks

This framework represents the state-of-the-art in autonomous software development,
enabling rapid, reliable, and scalable system evolution with minimal human intervention.
"""

__version__ = "4.0.0"
__author__ = "Terragon Labs"
__description__ = "Autonomous Software Development Lifecycle Framework"
__license__ = "BSD 3-Clause"

# Core components
from .autonomous_executor import AutonomousExecutor, execute_autonomous_sdlc
from .hypothesis_driven_development import (
    HypothesisGenerator,
    AutomatedExperimentRunner, 
    execute_hypothesis_driven_development
)
from .quality_orchestrator import (
    AutonomousQualityOrchestrator,
    execute_quality_gates
)
from .global_orchestrator import (
    GlobalOrchestrator,
    implement_global_first_features
)
from .self_improving_orchestrator import (
    SelfImprovingOrchestrator,
    implement_self_improving_patterns
)

# Utility functions
def get_version() -> str:
    """Get framework version."""
    return __version__

def get_framework_info() -> dict:
    """Get comprehensive framework information."""
    return {
        "name": "Terragon SDLC",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "capabilities": [
            "Autonomous code generation",
            "Intelligent project analysis", 
            "Progressive enhancement",
            "Hypothesis-driven development",
            "Statistical validation",
            "Comprehensive quality gates",
            "Auto-fixing and optimization",
            "Global i18n and compliance",
            "Self-improving systems",
            "Research framework integration"
        ],
        "supported_project_types": [
            "API services",
            "CLI applications", 
            "Web applications",
            "Libraries and frameworks",
            "Robotics systems",
            "Machine learning research",
            "Microservices",
            "Enterprise applications"
        ]
    }

# Main execution interface
async def execute_full_sdlc_pipeline(project_root=None, config=None):
    """Execute complete autonomous SDLC pipeline."""
    import asyncio
    from pathlib import Path
    
    if project_root is None:
        project_root = Path.cwd()
    
    results = {}
    
    # Phase 1: Autonomous SDLC execution
    print("🚀 Phase 1: Executing autonomous SDLC...")
    results['sdlc'] = await execute_autonomous_sdlc(project_root)
    
    # Phase 2: Hypothesis-driven development
    print("🔬 Phase 2: Running hypothesis-driven development...")
    results['hypothesis_testing'] = await execute_hypothesis_driven_development(project_root)
    
    # Phase 3: Quality gates validation
    print("🛡️ Phase 3: Validating quality gates...")
    results['quality'] = await execute_quality_gates(project_root, config)
    
    # Phase 4: Global-first implementation
    print("🌍 Phase 4: Implementing global-first features...")
    results['globalization'] = await implement_global_first_features(project_root)
    
    # Phase 5: Self-improving patterns
    print("🧠 Phase 5: Activating self-improving patterns...")
    results['self_improvement'] = await implement_self_improving_patterns(project_root)
    
    return results

# Convenience aliases
AutonomousSDLC = AutonomousExecutor
QualityGates = AutonomousQualityOrchestrator
GlobalFirst = GlobalOrchestrator
SelfImproving = SelfImprovingOrchestrator

__all__ = [
    # Main classes
    'AutonomousExecutor',
    'HypothesisGenerator', 
    'AutomatedExperimentRunner',
    'AutonomousQualityOrchestrator',
    'GlobalOrchestrator',
    'SelfImprovingOrchestrator',
    
    # Execution functions
    'execute_autonomous_sdlc',
    'execute_hypothesis_driven_development', 
    'execute_quality_gates',
    'implement_global_first_features',
    'implement_self_improving_patterns',
    'execute_full_sdlc_pipeline',
    
    # Utilities
    'get_version',
    'get_framework_info',
    
    # Aliases
    'AutonomousSDLC',
    'QualityGates', 
    'GlobalFirst',
    'SelfImproving'
]