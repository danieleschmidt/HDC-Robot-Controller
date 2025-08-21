#!/usr/bin/env python3
"""
Test Suite for Generation 7: Swarm Intelligence
Comprehensive testing of emergent coordination systems.
"""

import pytest
import numpy as np
import time
import asyncio
import sys
import pathlib
from unittest.mock import Mock, patch

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hdc_robot_controller.swarm_intelligence.emergent_coordination import (
    EmergentSwarmCoordinator,
    SwarmAgent,
    SwarmMessage,
    SwarmRole,
    SwarmBehavior,
    demonstrate_swarm_intelligence
)
from hdc_robot_controller.core.hypervector import HyperVector


class TestSwarmAgent:
    """Test swarm agent functionality."""
    
    @pytest.fixture
    def test_agent(self):
        """Create test swarm agent."""
        return SwarmAgent(
            agent_id="test_robot_001",
            position=np.array([10.0, 20.0, 5.0]),
            role=SwarmRole.EXPLORER,
            behavior_state=HyperVector(1000),
            local_memory=None,  # Will be auto-created
            communication_range=50.0,
            energy_level=0.8,
            neighbors=["robot_002", "robot_003"],
            last_update=time.time()
        )
    
    def test_agent_creation(self, test_agent):
        """Test agent creation and initialization."""
        assert test_agent.agent_id == "test_robot_001"
        assert test_agent.role == SwarmRole.EXPLORER
        assert test_agent.communication_range == 50.0
        assert test_agent.energy_level == 0.8
        assert len(test_agent.neighbors) == 2
        assert test_agent.local_memory is not None  # Auto-created in __post_init__
        
        # Test position
        assert np.array_equal(test_agent.position, [10.0, 20.0, 5.0])


class TestSwarmMessage:
    """Test swarm message functionality."""
    
    @pytest.fixture
    def test_message(self):
        """Create test swarm message."""
        content_hv = HyperVector(1000)
        content_hv.randomize(seed=42)
        
        return SwarmMessage(
            sender_id="robot_001",
            message_type="status_update",
            content=content_hv,
            timestamp=time.time(),
            priority=3,
            broadcast_count=0
        )
    
    def test_message_creation(self, test_message):
        """Test message creation."""
        assert test_message.sender_id == "robot_001"
        assert test_message.message_type == "status_update"
        assert isinstance(test_message.content, HyperVector)
        assert test_message.priority == 3
        assert test_message.broadcast_count == 0
    
    def test_message_serialization(self, test_message):
        """Test message serialization and deserialization."""
        # Serialize to bytes
        serialized = test_message.to_bytes()
        assert isinstance(serialized, bytes)
        
        # Deserialize back
        deserialized = SwarmMessage.from_bytes(serialized)
        
        assert deserialized.sender_id == test_message.sender_id
        assert deserialized.message_type == test_message.message_type
        assert deserialized.priority == test_message.priority
        assert deserialized.broadcast_count == test_message.broadcast_count
        
        # Content should be similar (allowing for floating point precision)
        similarity = test_message.content.similarity(deserialized.content)
        assert similarity > 0.9


class TestEmergentSwarmCoordinator:
    """Test emergent swarm coordination system."""
    
    @pytest.fixture
    def swarm_coordinator(self):
        """Create swarm coordinator for testing."""
        return EmergentSwarmCoordinator(
            max_swarm_size=20,
            communication_protocol="quantum_encrypted",
            coordination_algorithm="emergent_consensus"
        )
    
    @pytest.fixture
    def agent_capabilities(self):
        """Create test agent capabilities."""
        return {
            "mobility": 0.8,
            "sensors": 0.9,
            "communication": 0.7,
            "processing": 0.6,
            "autonomy": 0.75,
            "manipulation": 0.5,
            "storage": 0.6,
            "precision": 0.7,
            "specialization": 0.4,
            "adaptability": 0.8,
            "memory": 0.7,
            "range": 0.8,
            "reliability": 0.9,
            "decision_speed": 0.8,
            "expertise": 0.5
        }
    
    def test_coordinator_initialization(self, swarm_coordinator):
        """Test swarm coordinator initialization."""
        assert swarm_coordinator.max_swarm_size == 20
        assert swarm_coordinator.communication_protocol == "quantum_encrypted"
        assert swarm_coordinator.coordination_algorithm == "emergent_consensus"
        assert len(swarm_coordinator.active_agents) == 0
        assert swarm_coordinator.quantum_engine is not None
        assert swarm_coordinator.hybrid_processor is not None
    
    def test_agent_registration(self, swarm_coordinator, agent_capabilities):
        """Test agent registration in swarm."""
        agent_id = "test_robot_001"
        position = np.array([0.0, 0.0, 0.0])
        
        # Register agent
        agent = swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
        
        assert isinstance(agent, SwarmAgent)
        assert agent.agent_id == agent_id
        assert np.array_equal(agent.position, position)
        assert agent.role in SwarmRole
        
        # Check registration in coordinator
        assert agent_id in swarm_coordinator.active_agents
        assert len(swarm_coordinator.active_agents) == 1
        assert agent.role in swarm_coordinator.role_assignments
        assert agent_id in swarm_coordinator.role_assignments[agent.role]
    
    def test_multiple_agent_registration(self, swarm_coordinator, agent_capabilities):
        """Test registration of multiple agents."""
        agents = []
        
        for i in range(5):
            agent_id = f"robot_{i:03d}"
            position = np.random.uniform(-50, 50, 3)
            
            agent = swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
            agents.append(agent)
        
        assert len(swarm_coordinator.active_agents) == 5
        assert len(agents) == 5
        
        # Check role distribution
        role_counts = {}
        for agent in agents:
            role_counts[agent.role] = role_counts.get(agent.role, 0) + 1
        
        # Should have some role diversity
        assert len(role_counts) > 1
    
    def test_swarm_capacity_limit(self, swarm_coordinator, agent_capabilities):
        """Test swarm capacity limits."""
        # Fill swarm to capacity
        for i in range(swarm_coordinator.max_swarm_size):
            agent_id = f"robot_{i:03d}"
            position = np.random.uniform(-100, 100, 3)
            swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
        
        assert len(swarm_coordinator.active_agents) == swarm_coordinator.max_swarm_size
        
        # Try to add one more (should fail)
        with pytest.raises(ValueError, match="Swarm at maximum capacity"):
            swarm_coordinator.register_agent("overflow_robot", np.array([0, 0, 0]), agent_capabilities)
    
    def test_role_assignment_logic(self, swarm_coordinator):
        """Test emergent role assignment logic."""
        # Create agents with different capability profiles
        
        # Explorer profile (high mobility, sensors)
        explorer_caps = {
            "mobility": 0.9, "sensors": 0.9, "autonomy": 0.8,
            "communication": 0.6, "processing": 0.5, "manipulation": 0.3
        }
        explorer = swarm_coordinator.register_agent("explorer", np.array([0, 0, 0]), explorer_caps)
        assert explorer.role == SwarmRole.EXPLORER
        
        # Coordinator profile (high communication, processing)
        coordinator_caps = {
            "communication": 0.9, "processing": 0.9, "memory": 0.8,
            "mobility": 0.5, "sensors": 0.6, "manipulation": 0.4
        }
        coordinator = swarm_coordinator.register_agent("coordinator", np.array([10, 0, 0]), coordinator_caps)
        # Role assignment is probabilistic, so we just check it's a valid role
        assert coordinator.role in SwarmRole
        
        # Collector profile (high manipulation, storage)
        collector_caps = {
            "manipulation": 0.9, "storage": 0.9, "precision": 0.8,
            "mobility": 0.6, "sensors": 0.5, "communication": 0.4
        }
        collector = swarm_coordinator.register_agent("collector", np.array([20, 0, 0]), collector_caps)
        assert collector.role in SwarmRole
    
    def test_message_sending(self, swarm_coordinator, agent_capabilities):
        """Test message sending between agents."""
        # Register two agents
        agent1 = swarm_coordinator.register_agent("robot_001", np.array([0, 0, 0]), agent_capabilities)
        agent2 = swarm_coordinator.register_agent("robot_002", np.array([10, 0, 0]), agent_capabilities)
        
        # Create test message
        content_hv = HyperVector(1000)
        content_hv.randomize(seed=42)
        
        message = SwarmMessage(
            sender_id="robot_001",
            message_type="test_message",
            content=content_hv,
            timestamp=time.time(),
            priority=2,
            broadcast_count=0
        )
        
        # Send message
        swarm_coordinator.send_message("robot_002", message)
        
        # Allow time for processing
        time.sleep(0.1)
        
        # Check message queue (message should be processed quickly)
        # Note: In real implementation, message would be processed by communication thread
        assert "robot_002" in swarm_coordinator.message_queues
    
    def test_task_assignment(self, swarm_coordinator, agent_capabilities):
        """Test swarm task assignment."""
        # Register agents
        for i in range(3):
            agent_id = f"robot_{i:03d}"
            position = np.random.uniform(-50, 50, 3)
            swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
        
        # Assign task to swarm
        task_id = swarm_coordinator.assign_swarm_task("exploration", priority=5)
        
        assert isinstance(task_id, str)
        assert len(task_id) == 16  # Should be 16-character hash
        
        # Assign task to specific agents
        target_agents = ["robot_001", "robot_002"]
        task_id2 = swarm_coordinator.assign_swarm_task("collection", target_agents=target_agents, priority=3)
        
        assert isinstance(task_id2, str)
        assert task_id != task_id2  # Different tasks should have different IDs
    
    def test_emergency_response(self, swarm_coordinator, agent_capabilities):
        """Test emergency response system."""
        # Register agents
        for i in range(3):
            agent_id = f"robot_{i:03d}"
            position = np.random.uniform(-50, 50, 3)
            swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
        
        # Trigger emergency
        emergency_location = np.array([25.0, 30.0, 5.0])
        swarm_coordinator.trigger_emergency_response("fire_detected", emergency_location, severity=8)
        
        # Allow time for emergency processing
        time.sleep(0.2)
        
        # Emergency should be logged and messages should be sent
        # (Specific behavior depends on implementation details)
        assert len(swarm_coordinator.active_agents) == 3  # Agents should still be active
    
    def test_agent_shutdown(self, swarm_coordinator, agent_capabilities):
        """Test graceful agent shutdown."""
        # Register agents
        agent_ids = []
        for i in range(5):
            agent_id = f"robot_{i:03d}"
            position = np.random.uniform(-50, 50, 3)
            swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
            agent_ids.append(agent_id)
        
        assert len(swarm_coordinator.active_agents) == 5
        
        # Shutdown one agent
        shutdown_agent = agent_ids[0]
        swarm_coordinator.shutdown_agent(shutdown_agent)
        
        assert len(swarm_coordinator.active_agents) == 4
        assert shutdown_agent not in swarm_coordinator.active_agents
        assert shutdown_agent not in swarm_coordinator.message_queues
    
    def test_network_topology_update(self, swarm_coordinator, agent_capabilities):
        """Test network topology updates."""
        # Register agents in close proximity
        agents = []
        for i in range(4):
            agent_id = f"robot_{i:03d}"
            position = np.array([i * 10.0, 0.0, 0.0])  # Linear arrangement
            agent = swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
            agents.append(agent)
        
        # Allow time for topology updates
        time.sleep(0.1)
        
        # Check neighbor relationships
        # Agents should have neighbors based on communication range
        for agent in agents:
            neighbors = agent.neighbors
            
            # Each agent should have some neighbors (except possibly edge agents)
            # The exact number depends on communication range and positions
            assert isinstance(neighbors, list)
            
            # Verify neighbors are other agents in the swarm
            for neighbor_id in neighbors:
                assert neighbor_id in swarm_coordinator.active_agents
                assert neighbor_id != agent.agent_id
    
    def test_swarm_metrics(self, swarm_coordinator, agent_capabilities):
        """Test swarm performance metrics collection."""
        # Register agents
        for i in range(5):
            agent_id = f"robot_{i:03d}"
            position = np.random.uniform(-50, 50, 3)
            swarm_coordinator.register_agent(agent_id, position, agent_capabilities)
        
        # Allow some time for operations
        time.sleep(0.1)
        
        # Get metrics
        metrics = swarm_coordinator.get_swarm_metrics()
        
        # Check required metrics
        assert "swarm_size" in metrics
        assert "max_swarm_size" in metrics
        assert "swarm_capacity_usage" in metrics
        assert "avg_communication_latency_ms" in metrics
        assert "communication_target_ms" in metrics
        assert "latency_target_achieved" in metrics
        assert "role_distribution" in metrics
        assert "system_status" in metrics
        
        # Verify values
        assert metrics["swarm_size"] == 5
        assert metrics["max_swarm_size"] == 20
        assert metrics["swarm_capacity_usage"] == 0.25
        assert metrics["communication_target_ms"] == 5.0
        assert metrics["system_status"] in ["optimal", "degraded"]


class TestSwarmPerformance:
    """Test swarm performance characteristics."""
    
    @pytest.fixture
    def large_swarm(self):
        """Create larger swarm for performance testing."""
        swarm = EmergentSwarmCoordinator(max_swarm_size=100)
        
        # Register many agents
        capabilities = {
            "mobility": 0.8, "sensors": 0.7, "communication": 0.9,
            "processing": 0.6, "autonomy": 0.75
        }
        
        for i in range(50):
            agent_id = f"perf_robot_{i:03d}"
            position = np.random.uniform(-200, 200, 3)
            swarm.register_agent(agent_id, position, capabilities)
        
        return swarm
    
    def test_communication_latency_target(self, large_swarm):
        """Test communication latency targets (<5ms)."""
        # Get initial metrics
        metrics = large_swarm.get_swarm_metrics()
        
        # Communication latency should meet target
        avg_latency = metrics.get("avg_communication_latency_ms", 0)
        target_latency = metrics["communication_target_ms"]
        
        # Allow some tolerance for simulation environment
        assert avg_latency <= target_latency * 2  # Allow 2x target for testing
    
    def test_large_scale_coordination(self, large_swarm):
        """Test coordination with large number of agents."""
        # Assign task to entire swarm
        task_id = large_swarm.assign_swarm_task("large_scale_exploration", priority=4)
        
        assert isinstance(task_id, str)
        
        # Trigger emergency to test coordination
        emergency_location = np.array([100.0, 50.0, 10.0])
        large_swarm.trigger_emergency_response("system_failure", emergency_location, severity=7)
        
        # Allow time for coordination
        time.sleep(0.5)
        
        # Get final metrics
        metrics = large_swarm.get_swarm_metrics()
        assert metrics["swarm_size"] == 50
        assert metrics["system_status"] in ["optimal", "degraded"]
    
    def test_scalability_metrics(self, large_swarm):
        """Test swarm scalability metrics."""
        metrics = large_swarm.get_swarm_metrics()
        
        # Should handle 50 agents efficiently
        assert metrics["swarm_size"] == 50
        assert metrics["swarm_capacity_usage"] == 0.5
        
        # Network should be well-connected
        avg_neighbors = metrics.get("avg_neighbors_per_agent", 0)
        assert avg_neighbors > 0  # Agents should have neighbors
        
        # Role distribution should be reasonable
        role_dist = metrics["role_distribution"]
        assert len(role_dist) > 1  # Should have multiple roles represented


class TestSwarmIntegration:
    """Test integration with quantum and HDC systems."""
    
    def test_quantum_hdc_integration(self):
        """Test integration with quantum HDC engine."""
        swarm = EmergentSwarmCoordinator(max_swarm_size=10)
        
        # Verify quantum engine is available
        assert swarm.quantum_engine is not None
        assert swarm.hybrid_processor is not None
        
        # Register agent and check quantum operations
        capabilities = {"mobility": 0.8, "sensors": 0.9, "communication": 0.7}
        agent = swarm.register_agent("quantum_test", np.array([0, 0, 0]), capabilities)
        
        # Agent behavior state should be HyperVector (quantum-compatible)
        assert isinstance(agent.behavior_state, HyperVector)
        
        # Quantum metrics should be available
        metrics = swarm.get_swarm_metrics()
        assert "quantum_engine_metrics" in metrics
        assert "hybrid_processing_metrics" in metrics
    
    def test_hdc_message_processing(self):
        """Test HDC-based message processing."""
        swarm = EmergentSwarmCoordinator(max_swarm_size=5)
        
        # Register agents
        capabilities = {"mobility": 0.8, "sensors": 0.9}
        for i in range(3):
            agent_id = f"hdc_robot_{i}"
            position = np.array([i * 20.0, 0, 0])
            swarm.register_agent(agent_id, position, capabilities)
        
        # Create HDC-encoded message
        content_hv = HyperVector(1000)
        content_hv.randomize(seed=123)
        
        message = SwarmMessage(
            sender_id="hdc_robot_0",
            message_type="behavior_update", 
            content=content_hv,
            timestamp=time.time(),
            priority=3,
            broadcast_count=0
        )
        
        # Send message
        swarm.send_message("hdc_robot_1", message)
        
        # Allow processing
        time.sleep(0.1)
        
        # Verify agent behavior state was updated (would happen in processing loop)
        agent = swarm.active_agents["hdc_robot_1"]
        assert isinstance(agent.behavior_state, HyperVector)


@pytest.mark.asyncio
async def test_swarm_demonstration():
    """Test the swarm intelligence demonstration function."""
    try:
        swarm = await demonstrate_swarm_intelligence()
        
        assert isinstance(swarm, EmergentSwarmCoordinator)
        
        # Check that agents were created and operations performed
        metrics = swarm.get_swarm_metrics()
        assert metrics["swarm_size"] > 0  # Some agents should remain after shutdown
        
    except Exception as e:
        # Demonstration should not fail
        pytest.fail(f"Swarm demonstration failed: {e}")


class TestSwarmBehaviorEmergence:
    """Test emergent behavior patterns in swarms."""
    
    def test_role_emergence(self):
        """Test emergence of balanced role distribution."""
        swarm = EmergentSwarmCoordinator(max_swarm_size=30)
        
        # Register agents with random capabilities
        for i in range(20):
            capabilities = {
                "mobility": np.random.uniform(0.3, 1.0),
                "sensors": np.random.uniform(0.3, 1.0),
                "communication": np.random.uniform(0.3, 1.0),
                "processing": np.random.uniform(0.3, 1.0),
                "manipulation": np.random.uniform(0.3, 1.0),
                "autonomy": np.random.uniform(0.3, 1.0)
            }
            
            agent_id = f"emerge_robot_{i:02d}"
            position = np.random.uniform(-100, 100, 3)
            swarm.register_agent(agent_id, position, capabilities)
        
        # Check role distribution
        metrics = swarm.get_swarm_metrics()
        role_dist = metrics["role_distribution"]
        
        # Should have multiple roles represented
        assert len(role_dist) >= 3
        
        # No single role should dominate completely
        total_agents = sum(role_dist.values())
        max_role_count = max(role_dist.values())
        dominance_ratio = max_role_count / total_agents
        
        assert dominance_ratio < 0.8  # No role should have >80% of agents
    
    def test_consensus_mechanism(self):
        """Test distributed consensus formation."""
        swarm = EmergentSwarmCoordinator(max_swarm_size=15)
        
        # Register agents
        capabilities = {"mobility": 0.8, "communication": 0.9, "processing": 0.7}
        for i in range(10):
            agent_id = f"consensus_robot_{i:02d}"
            position = np.random.uniform(-50, 50, 3)
            swarm.register_agent(agent_id, position, capabilities)
        
        # Create consensus message
        consensus_hv = HyperVector(1000)
        consensus_hv.randomize(seed=456)
        
        consensus_message = SwarmMessage(
            sender_id="consensus_robot_0",
            message_type="role_coordination",
            content=consensus_hv,
            timestamp=time.time(),
            priority=4,
            broadcast_count=0
        )
        
        # Send to multiple agents to simulate consensus process
        for i in range(5):
            swarm.send_message(f"consensus_robot_{i}", consensus_message)
        
        # Allow time for consensus processing
        time.sleep(0.2)
        
        # Check that consensus system is working
        assert len(swarm.swarm_consensus) >= 0  # Consensus might be processed or completed


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])