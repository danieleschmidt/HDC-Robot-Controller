"""
Generation 7: Swarm Intelligence Module
Emergent coordination for large-scale robot swarms.
"""

from .emergent_coordination import (
    EmergentSwarmCoordinator,
    SwarmAgent,
    SwarmMessage,
    SwarmRole,
    SwarmBehavior,
    demonstrate_swarm_intelligence
)

__all__ = [
    'EmergentSwarmCoordinator',
    'SwarmAgent', 
    'SwarmMessage',
    'SwarmRole',
    'SwarmBehavior',
    'demonstrate_swarm_intelligence'
]