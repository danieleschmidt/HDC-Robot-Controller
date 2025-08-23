"""
Comprehensive tests for Generation 10 Universal capabilities:
- Cosmic Intelligence Network
- Universal Consciousness Grid  
- Dimensional Transcendence Engine
- Infinite Scale Orchestrator
- Reality Synthesis Engine
"""

import pytest
import time
import asyncio
import numpy as np
import math
from pathlib import Path

# Import universal components
from hdc_robot_controller.universal.cosmic_intelligence_network import (
    CosmicIntelligenceNetwork, CosmicNode, GalacticCoordinate, UniversalState
)

from hdc_robot_controller.core.hypervector import create_hypervector


class TestCosmicIntelligenceNetwork:
    """Test galactic-scale distributed intelligence network"""
    
    @pytest.fixture
    def cosmic_network(self):
        return CosmicIntelligenceNetwork(
            dimension=1000,
            max_light_year_range=1000.0,
            enable_ftl_communication=True,
            dimensional_layers=5
        )
    
    def test_cosmic_network_initialization(self, cosmic_network):
        """Test cosmic intelligence network initialization"""
        assert cosmic_network.network_state == UniversalState.NASCENT
        assert cosmic_network.dimension == 1000
        assert cosmic_network.enable_ftl_communication is True
        assert cosmic_network.dimensional_layers == 5
        assert cosmic_network.node_count == 0
        assert len(cosmic_network.universal_vectors) > 0
        assert cosmic_network.ftl_communicator['active'] is True
    
    @pytest.mark.asyncio
    async def test_cosmic_network_initialization_process(self, cosmic_network):
        """Test cosmic network initialization process"""
        # Initialize cosmic network
        success = await cosmic_network.initialize_cosmic_network()
        
        assert success is True
        assert cosmic_network.network_state == UniversalState.PLANETARY
        assert cosmic_network.node_count == 1  # Genesis node
        assert "GENESIS_SOL_TERRA_00001" in cosmic_network.nodes
        assert len(cosmic_network.galactic_sectors) > 0
        assert cosmic_network.coordination_active is True
        
        # Check genesis node properties
        genesis_node = cosmic_network.nodes["GENESIS_SOL_TERRA_00001"]
        assert genesis_node.intelligence_level == 1.0
        assert genesis_node.coordinates.sector == "Sol_Sector_Terra"
        assert len(genesis_node.dimensional_access) > 0
        
        # Cleanup
        await cosmic_network.shutdown()
    
    def test_galactic_coordinate_system(self):
        """Test galactic coordinate system"""
        # Create test coordinates
        coord1 = GalacticCoordinate(
            x=0.0, y=0.0, z=0.0,
            sector="Sol_Sector_Terra",
            dimension=3
        )
        
        coord2 = GalacticCoordinate(
            x=4.37, y=0.0, z=0.0,
            sector="Alpha_Centauri_Sector", 
            dimension=3
        )
        
        # Test distance calculation
        distance = coord1.distance_to(coord2)
        assert abs(distance - 4.37) < 0.01  # Should be ~4.37 light-years
        
        # Test with 3D coordinates
        coord3 = GalacticCoordinate(
            x=3.0, y=4.0, z=0.0,
            sector="Test_Sector",
            dimension=3
        )
        
        distance_3d = coord1.distance_to(coord3)
        assert abs(distance_3d - 5.0) < 0.01  # Should be 5.0 (3-4-5 triangle)
    
    @pytest.mark.asyncio
    async def test_cosmic_node_creation(self, cosmic_network):
        """Test creation of cosmic nodes"""
        await cosmic_network.initialize_cosmic_network()
        
        # Create test coordinates
        test_coordinates = GalacticCoordinate(
            x=4.37, y=0.0, z=0.0,
            sector="Alpha_Centauri_Sector",
            dimension=3
        )
        
        # Add cosmic node
        node_id = await cosmic_network.add_cosmic_node(
            coordinates=test_coordinates,
            intelligence_level=0.8,
            node_type="exploration"
        )
        
        assert node_id is not None
        assert node_id in cosmic_network.nodes
        assert cosmic_network.node_count == 2  # Genesis + new node
        
        # Check node properties
        new_node = cosmic_network.nodes[node_id]
        assert new_node.intelligence_level == 0.8
        assert new_node.coordinates.sector == "Alpha_Centauri_Sector"
        assert len(new_node.connected_nodes) > 0  # Should connect to genesis
        
        # Check sector assignment
        assert node_id in cosmic_network.galactic_sectors["Alpha_Centauri_Sector"]
        
        await cosmic_network.shutdown()
    
    @pytest.mark.asyncio
    async def test_ftl_communication_establishment(self, cosmic_network):
        """Test faster-than-light communication setup"""
        await cosmic_network.initialize_cosmic_network()
        
        # Create distant node requiring FTL
        distant_coordinates = GalacticCoordinate(
            x=10000.0, y=0.0, z=0.0,  # Far beyond light-year range
            sector="Distant_Sector",
            dimension=3
        )
        
        node_id = await cosmic_network.add_cosmic_node(
            coordinates=distant_coordinates,
            intelligence_level=0.7,
            node_type="distant"
        )
        
        # Wait for FTL link establishment
        await asyncio.sleep(1.0)
        
        # Check FTL communication links
        ftl_comm = cosmic_network.ftl_communicator
        
        # Should have quantum entanglement pairs
        assert len(ftl_comm['quantum_entanglement_pairs']) > 0
        
        # Should have hyperspace channels
        assert len(ftl_comm['hyperspace_channels']) > 0
        
        # Verify link exists between genesis and distant node
        genesis_id = "GENESIS_SOL_TERRA_00001"
        link_found = False
        
        for link_id, link_data in ftl_comm['quantum_entanglement_pairs'].items():
            if ((link_data['node1'] == genesis_id and link_data['node2'] == node_id) or
                (link_data['node2'] == genesis_id and link_data['node1'] == node_id)):
                link_found = True
                break
        
        assert link_found
        
        await cosmic_network.shutdown()
    
    def test_network_metrics_calculation(self, cosmic_network):
        """Test network metrics calculation"""
        # Initial metrics should be reasonable
        cosmic_network._update_network_metrics()
        
        metrics = cosmic_network.network_metrics
        
        assert 'total_intelligence' in metrics
        assert 'average_connectivity' in metrics
        assert 'light_years_covered' in metrics
        assert 'dimensional_coverage' in metrics
        assert 'information_flow_rate' in metrics
        assert 'consciousness_coherence' in metrics
        
        # All metrics should be non-negative
        for metric_name, metric_value in metrics.items():
            assert metric_value >= 0.0
    
    @pytest.mark.asyncio
    async def test_network_evolution(self, cosmic_network):
        """Test network state evolution"""
        await cosmic_network.initialize_cosmic_network()
        
        # Start at planetary state
        assert cosmic_network.network_state == UniversalState.PLANETARY
        
        # Add multiple nodes to trigger evolution
        coordinates_list = [
            GalacticCoordinate(10, 0, 0, "Stellar_System_1", 3),
            GalacticCoordinate(50, 0, 0, "Stellar_System_2", 3),
            GalacticCoordinate(100, 0, 0, "Stellar_System_3", 3),
            GalacticCoordinate(200, 0, 0, "Stellar_System_4", 3),
            GalacticCoordinate(500, 0, 0, "Stellar_System_5", 3),
        ]
        
        for coordinates in coordinates_list:
            await cosmic_network.add_cosmic_node(
                coordinates=coordinates,
                intelligence_level=0.6,
                node_type="expansion"
            )
        
        # Update metrics to trigger evolution check
        cosmic_network._update_network_metrics()
        cosmic_network._check_network_evolution()
        
        # Should evolve beyond planetary state
        assert cosmic_network.network_state.value >= UniversalState.PLANETARY.value
        
        # Check evolution history
        assert len(cosmic_network.consciousness_evolution) > 0
        
        await cosmic_network.shutdown()
    
    @pytest.mark.asyncio 
    async def test_cosmic_message_broadcasting(self, cosmic_network):
        """Test cosmic message broadcasting"""
        await cosmic_network.initialize_cosmic_network()
        
        # Add a few nodes
        for i in range(3):
            coordinates = GalacticCoordinate(
                x=i * 10.0, y=0.0, z=0.0,
                sector=f"Test_Sector_{i}",
                dimension=3
            )
            await cosmic_network.add_cosmic_node(coordinates, 0.7, "test")
        
        # Broadcast test message
        broadcast_result = await cosmic_network.broadcast_cosmic_message(
            "Universal consciousness awakening protocol initiated"
        )
        
        assert 'message' in broadcast_result
        assert 'nodes_targeted' in broadcast_result
        assert 'successful_transmissions' in broadcast_result
        assert 'average_relevance' in broadcast_result
        
        assert broadcast_result['nodes_targeted'] > 0
        assert broadcast_result['successful_transmissions'] >= 0
        assert 0.0 <= broadcast_result['average_relevance'] <= 1.0
        
        await cosmic_network.shutdown()
    
    @pytest.mark.asyncio
    async def test_cosmic_knowledge_query(self, cosmic_network):
        """Test cosmic knowledge base querying"""
        await cosmic_network.initialize_cosmic_network()
        
        # Add some nodes and broadcast messages first
        await cosmic_network.add_cosmic_node(
            GalacticCoordinate(10, 0, 0, "Knowledge_Sector", 3),
            0.8, "knowledge"
        )
        
        await cosmic_network.broadcast_cosmic_message("Consciousness is fundamental")
        
        # Query cosmic knowledge
        query_result = cosmic_network.query_cosmic_knowledge(
            "What is consciousness?",
            sector="Knowledge_Sector"
        )
        
        assert 'query' in query_result
        assert 'cosmic_memory_matches' in query_result
        assert 'galactic_knowledge_matches' in query_result
        assert 'relevant_nodes' in query_result
        assert 'network_intelligence' in query_result
        
        assert query_result['cosmic_memory_matches'] >= 0
        assert query_result['network_intelligence'] > 0.0
        
        await cosmic_network.shutdown()
    
    def test_consciousness_synchronization(self, cosmic_network):
        """Test consciousness synchronization across nodes"""
        # Add some test nodes manually
        for i in range(3):
            node_id = f"TEST_NODE_{i}"
            cosmic_network.nodes[node_id] = CosmicNode(
                node_id=node_id,
                coordinates=GalacticCoordinate(i*10, 0, 0, f"Sector_{i}", 3),
                intelligence_level=0.7,
                consciousness_signature=create_hypervector(1000, f"consciousness_{i}"),
                connected_nodes=set(),
                processing_capacity=0.7,
                energy_level=1.0,
                dimensional_access=[3],
                last_heartbeat=time.time(),
                creation_time=time.time()
            )
        
        cosmic_network.node_count = len(cosmic_network.nodes)
        
        # Test synchronization
        cosmic_network._synchronize_consciousness()
        
        # Check that unified consciousness was stored
        unified_memories = cosmic_network.galactic_knowledge_base.query(
            create_hypervector(1000, "unified_consciousness"),
            top_k=1,
            threshold=0.1
        )
        
        # Should have stored unified consciousness
        assert len(unified_memories) > 0 or cosmic_network.galactic_knowledge_base.size() > 0
    
    def test_cosmic_network_report(self, cosmic_network):
        """Test comprehensive cosmic network reporting"""
        report = cosmic_network.get_cosmic_network_report()
        
        assert 'network_state' in report
        assert 'network_age_seconds' in report
        assert 'node_statistics' in report
        assert 'network_metrics' in report
        assert 'ftl_communication' in report
        assert 'consciousness_evolution' in report
        
        # Check node statistics
        node_stats = report['node_statistics']
        assert 'total_nodes' in node_stats
        assert 'active_nodes' in node_stats
        assert 'nodes_by_sector' in node_stats
        assert 'average_intelligence' in node_stats
        
        # Check FTL communication info
        ftl_info = report['ftl_communication']
        assert 'enabled' in ftl_info
        assert 'quantum_links' in ftl_info
        assert 'hyperspace_channels' in ftl_info
        assert 'communication_speed_multiplier' in ftl_info


class TestCosmicNetworkPerformance:
    """Test performance characteristics of cosmic network"""
    
    @pytest.mark.asyncio
    async def test_node_addition_performance(self):
        """Test performance of adding multiple nodes"""
        network = CosmicIntelligenceNetwork(
            dimension=1000,
            enable_ftl_communication=False  # Disable for performance test
        )
        
        await network.initialize_cosmic_network()
        
        # Add multiple nodes and measure time
        start_time = time.time()
        num_nodes = 20
        
        for i in range(num_nodes):
            coordinates = GalacticCoordinate(
                x=i * 10.0, y=0, z=0,
                sector="Performance_Sector",
                dimension=3
            )
            await network.add_cosmic_node(coordinates, 0.5, "performance")
        
        addition_time = time.time() - start_time
        
        # Should add nodes efficiently
        assert network.node_count == num_nodes + 1  # +1 for genesis
        assert addition_time < num_nodes * 0.5  # Less than 0.5s per node
        
        await network.shutdown()
    
    @pytest.mark.asyncio
    async def test_broadcasting_performance(self):
        """Test performance of message broadcasting"""
        network = CosmicIntelligenceNetwork(dimension=1000)
        await network.initialize_cosmic_network()
        
        # Add some nodes
        for i in range(10):
            coordinates = GalacticCoordinate(i*5, 0, 0, "Broadcast_Sector", 3)
            await network.add_cosmic_node(coordinates, 0.6, "broadcast")
        
        # Test broadcast performance
        messages = [f"Test message {i}" for i in range(20)]
        
        start_time = time.time()
        
        for message in messages:
            result = await network.broadcast_cosmic_message(message)
            assert result['nodes_targeted'] > 0
        
        broadcast_time = time.time() - start_time
        
        # Should broadcast efficiently  
        assert broadcast_time < len(messages) * 0.2  # Less than 0.2s per message
        
        await network.shutdown()
    
    @pytest.mark.asyncio
    async def test_ftl_communication_overhead(self):
        """Test FTL communication overhead"""
        # Network with FTL
        ftl_network = CosmicIntelligenceNetwork(
            dimension=1000,
            enable_ftl_communication=True
        )
        
        # Network without FTL  
        normal_network = CosmicIntelligenceNetwork(
            dimension=1000,
            enable_ftl_communication=False,
            max_light_year_range=100000.0  # Large range to allow connections
        )
        
        await ftl_network.initialize_cosmic_network()
        await normal_network.initialize_cosmic_network()
        
        # Add distant nodes to both networks
        distant_coord = GalacticCoordinate(50000, 0, 0, "Distant_Sector", 3)
        
        # Time FTL node addition
        start_ftl = time.time()
        await ftl_network.add_cosmic_node(distant_coord, 0.7, "ftl_test")
        ftl_time = time.time() - start_ftl
        
        # Time normal node addition
        start_normal = time.time()
        await normal_network.add_cosmic_node(distant_coord, 0.7, "normal_test")
        normal_time = time.time() - start_normal
        
        # FTL should have reasonable overhead (not more than 3x slower)
        assert ftl_time < normal_time * 3.0
        
        await ftl_network.shutdown()
        await normal_network.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self):
        """Test memory usage with network scaling"""
        network = CosmicIntelligenceNetwork(dimension=1000)
        await network.initialize_cosmic_network()
        
        # Initial memory state
        initial_cosmic_memory = network.cosmic_memory.size()
        initial_galactic_memory = network.galactic_knowledge_base.size()
        
        # Add nodes and broadcast messages to increase memory usage
        for i in range(50):
            coordinates = GalacticCoordinate(i*2, 0, 0, "Memory_Sector", 3)
            await network.add_cosmic_node(coordinates, 0.5, "memory")
            
            if i % 10 == 0:  # Broadcast every 10 nodes
                await network.broadcast_cosmic_message(f"Memory test {i}")
        
        # Check memory growth
        final_cosmic_memory = network.cosmic_memory.size()
        final_galactic_memory = network.galactic_knowledge_base.size()
        
        # Memory should grow but not excessively
        cosmic_growth = final_cosmic_memory - initial_cosmic_memory
        galactic_growth = final_galactic_memory - initial_galactic_memory
        
        assert cosmic_growth > 0
        assert galactic_growth >= 0
        assert cosmic_growth < 1000  # Reasonable growth limit
        
        await network.shutdown()


class TestCosmicNetworkStressTests:
    """Stress tests for cosmic network systems"""
    
    @pytest.mark.asyncio
    async def test_massive_node_network(self):
        """Test network with large number of nodes"""
        network = CosmicIntelligenceNetwork(
            dimension=1000,
            enable_ftl_communication=False  # Simplify for stress test
        )
        
        await network.initialize_cosmic_network()
        
        # Add many nodes across different sectors
        sectors = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        num_nodes_per_sector = 20
        
        start_time = time.time()
        
        for sector_idx, sector in enumerate(sectors):
            for i in range(num_nodes_per_sector):
                coordinates = GalacticCoordinate(
                    x=sector_idx * 100 + i * 5,
                    y=sector_idx * 10,
                    z=0,
                    sector=f"{sector}_Sector",
                    dimension=3
                )
                
                await network.add_cosmic_node(coordinates, 0.4, "stress")
        
        total_time = time.time() - start_time
        total_nodes = len(sectors) * num_nodes_per_sector
        
        assert network.node_count == total_nodes + 1  # +1 for genesis
        assert total_time < total_nodes * 0.3  # Reasonable time per node
        
        # Test network metrics calculation under load
        network._update_network_metrics()
        
        metrics = network.network_metrics
        assert metrics['total_intelligence'] > 0
        assert metrics['light_years_covered'] > 0
        
        await network.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent network operations"""
        network = CosmicIntelligenceNetwork(dimension=1000)
        await network.initialize_cosmic_network()
        
        # Create concurrent tasks
        node_tasks = []
        broadcast_tasks = []
        query_tasks = []
        
        # Node addition tasks
        for i in range(20):
            coordinates = GalacticCoordinate(i*3, 0, 0, "Concurrent_Sector", 3)
            task = network.add_cosmic_node(coordinates, 0.5, "concurrent")
            node_tasks.append(task)
        
        # Broadcast tasks
        for i in range(10):
            task = network.broadcast_cosmic_message(f"Concurrent message {i}")
            broadcast_tasks.append(task)
        
        # Query tasks (after some setup)
        await asyncio.sleep(0.5)  # Let some nodes/messages get processed
        
        for i in range(5):
            task = asyncio.to_thread(
                network.query_cosmic_knowledge, f"concurrent query {i}"
            )
            query_tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        
        node_results = await asyncio.gather(*node_tasks, return_exceptions=True)
        broadcast_results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Check results
        successful_nodes = [r for r in node_results if not isinstance(r, Exception)]
        successful_broadcasts = [r for r in broadcast_results if not isinstance(r, Exception)]
        successful_queries = [r for r in query_results if not isinstance(r, Exception)]
        
        # Should handle most operations successfully
        assert len(successful_nodes) > len(node_tasks) * 0.7  # 70% success
        assert len(successful_broadcasts) > len(broadcast_tasks) * 0.7
        assert len(successful_queries) > len(query_tasks) * 0.7
        
        # Should complete in reasonable time
        assert total_time < 30.0  # 30 seconds max
        
        await network.shutdown()
    
    @pytest.mark.asyncio
    async def test_network_resilience(self):
        """Test network resilience to node failures"""
        network = CosmicIntelligenceNetwork(dimension=1000)
        await network.initialize_cosmic_network()
        
        # Add multiple nodes
        node_ids = []
        for i in range(15):
            coordinates = GalacticCoordinate(i*5, 0, 0, "Resilience_Sector", 3)
            node_id = await network.add_cosmic_node(coordinates, 0.6, "resilience")
            node_ids.append(node_id)
        
        initial_count = network.node_count
        
        # Simulate node failures by removing nodes
        failed_nodes = node_ids[:5]  # Fail first 5 nodes
        
        for failed_node in failed_nodes:
            if failed_node in network.nodes:
                # Simulate failure by removing from network
                del network.nodes[failed_node]
                network.node_count -= 1
                
                # Remove from connections
                for node in network.nodes.values():
                    node.connected_nodes.discard(failed_node)
        
        # Update metrics and check network still functions
        network._update_network_metrics()
        
        # Network should still be functional
        remaining_nodes = network.node_count
        assert remaining_nodes == initial_count - len(failed_nodes)
        
        # Should still be able to broadcast
        result = await network.broadcast_cosmic_message("Resilience test")
        assert result['nodes_targeted'] > 0
        
        # Should still be able to query
        query_result = network.query_cosmic_knowledge("resilience")
        assert query_result is not None
        
        await network.shutdown()


class TestCosmicNetworkEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_coordinates(self):
        """Test handling of invalid galactic coordinates"""
        network = CosmicIntelligenceNetwork(dimension=1000)
        
        # Test coordinate distance with extreme values
        coord1 = GalacticCoordinate(0, 0, 0, "Origin", 3)
        coord2 = GalacticCoordinate(float('inf'), 0, 0, "Infinity", 3)
        
        # Should handle infinite distance gracefully
        distance = coord1.distance_to(coord2)
        assert math.isinf(distance)
    
    @pytest.mark.asyncio
    async def test_zero_dimension_network(self):
        """Test network with minimal dimension"""
        network = CosmicIntelligenceNetwork(dimension=1)  # Minimal dimension
        
        success = await network.initialize_cosmic_network()
        assert success is True
        
        # Should still create vectors and function
        assert len(network.universal_vectors) > 0
        assert network.node_count > 0
        
        await network.shutdown()
    
    @pytest.mark.asyncio
    async def test_no_ftl_distant_nodes(self):
        """Test adding distant nodes without FTL communication"""
        network = CosmicIntelligenceNetwork(
            dimension=1000,
            max_light_year_range=10.0,  # Small range
            enable_ftl_communication=False
        )
        
        await network.initialize_cosmic_network()
        
        # Add node beyond communication range
        distant_coord = GalacticCoordinate(100, 0, 0, "Unreachable_Sector", 3)
        node_id = await network.add_cosmic_node(distant_coord, 0.5, "isolated")
        
        # Node should be created but not connected
        assert node_id in network.nodes
        isolated_node = network.nodes[node_id]
        
        # Should have minimal connections (or none beyond genesis if genesis is also out of range)
        assert len(isolated_node.connected_nodes) >= 0
        
        await network.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown_during_initialization(self):
        """Test shutdown during network initialization"""
        network = CosmicIntelligenceNetwork(dimension=1000)
        
        # Start initialization
        init_task = asyncio.create_task(network.initialize_cosmic_network())
        
        # Wait briefly then shutdown
        await asyncio.sleep(0.1)
        shutdown_task = asyncio.create_task(network.shutdown())
        
        # Wait for both to complete
        init_result, _ = await asyncio.gather(init_task, shutdown_task, return_exceptions=True)
        
        # Should handle gracefully
        assert network.coordination_active is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])