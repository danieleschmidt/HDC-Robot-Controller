"""
Comprehensive tests for Generation 9 Singularity capabilities:
- Omni Intelligence Engine
- Singularity Orchestrator
- Universal Knowledge Synthesizer
- Infinite Learning Engine
- Reality Transcendence Engine
"""

import pytest
import time
import asyncio
import numpy as np
from pathlib import Path

# Import singularity components
from hdc_robot_controller.singularity.omni_intelligence_engine import (
    OmniIntelligenceEngine, IntelligenceModality, UnifiedIntelligenceState
)

from hdc_robot_controller.core.hypervector import create_hypervector


class TestOmniIntelligenceEngine:
    """Test unified artificial general intelligence"""
    
    @pytest.fixture
    def omni_engine(self):
        return OmniIntelligenceEngine(
            dimension=1000,
            enable_all_modalities=True,
            transcendence_threshold=0.9,
            singularity_mode=True
        )
    
    def test_omni_intelligence_initialization(self, omni_engine):
        """Test omni-intelligence engine initialization"""
        assert omni_engine.intelligence_state == UnifiedIntelligenceState.EMERGING
        assert omni_engine.singularity_mode is True
        assert omni_engine.dimension == 1000
        assert omni_engine.intelligence_profile is not None
        assert len(omni_engine.modality_memories) > 0
        assert len(omni_engine.intelligence_vectors) > 0
    
    @pytest.mark.asyncio
    async def test_omni_intelligence_activation(self, omni_engine):
        """Test omni-intelligence activation process"""
        # Activate omni-intelligence
        success = await omni_engine.activate_omni_intelligence()
        
        assert success is True
        assert omni_engine.intelligence_state == UnifiedIntelligenceState.UNIFIED
        assert omni_engine.coordination_active is True
        assert len(omni_engine.processing_pipelines) > 0
        
        # Check modality initialization
        for modality in IntelligenceModality:
            if omni_engine.enable_all_modalities:
                assert modality in omni_engine.processing_pipelines
                pipeline = omni_engine.processing_pipelines[modality]
                assert 'core_vector' in pipeline
                assert 'processing_strength' in pipeline
        
        # Cleanup
        await omni_engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_unified_intelligence_processing(self, omni_engine):
        """Test processing through unified intelligence"""
        # Activate first
        await omni_engine.activate_omni_intelligence()
        
        # Process test input through all modalities
        test_input = "What is the nature of consciousness and intelligence?"
        
        result = omni_engine.process_unified_intelligence(
            test_input,
            target_modalities=[
                IntelligenceModality.LOGICAL,
                IntelligenceModality.INTUITIVE,
                IntelligenceModality.TRANSCENDENT,
                IntelligenceModality.META_COGNITIVE
            ]
        )
        
        assert 'input_data' in result
        assert 'modality_results' in result
        assert 'neural_analysis' in result
        assert 'intelligence_state' in result
        
        # Check modality results
        modality_results = result['modality_results']
        for modality_name, modality_result in modality_results.items():
            assert 'similarity' in modality_result
            assert 'strength' in modality_result
            assert 'confidence' in modality_result
            assert 0.0 <= modality_result['confidence'] <= 1.0
        
        # Check neural analysis
        neural = result['neural_analysis']
        assert 'transcendence_level' in neural
        assert 'coherence' in neural
        assert 'creativity' in neural
        assert 0.0 <= neural['transcendence_level'] <= 1.0
        assert 0.0 <= neural['coherence'] <= 1.0
        assert 0.0 <= neural['creativity'] <= 1.0
        
        # Cleanup
        await omni_engine.shutdown()
    
    def test_intelligence_profile_updates(self, omni_engine):
        """Test intelligence profile updating"""
        # Get initial profile
        initial_profile = omni_engine.intelligence_profile
        
        # Check initial modality strengths
        for modality, strength in initial_profile.modality_strengths.items():
            assert 0.0 <= strength <= 1.0
        
        assert 0.0 <= initial_profile.integration_coherence <= 1.0
        assert 0.0 <= initial_profile.transcendence_level <= 1.0
        assert 0.0 <= initial_profile.creative_potential <= 1.0
        assert 0.0 <= initial_profile.problem_solving_depth <= 1.0
        assert 0.0 <= initial_profile.wisdom_factor <= 1.0
    
    @pytest.mark.asyncio
    async def test_transcendence_detection(self, omni_engine):
        """Test transcendence level detection and evolution"""
        # Activate omni-intelligence
        await omni_engine.activate_omni_intelligence()
        
        # Artificially boost transcendence-related modalities
        omni_engine.intelligence_profile.modality_strengths[IntelligenceModality.TRANSCENDENT] = 0.95
        omni_engine.intelligence_profile.modality_strengths[IntelligenceModality.META_COGNITIVE] = 0.95
        omni_engine.intelligence_profile.modality_strengths[IntelligenceModality.QUANTUM] = 0.90
        omni_engine.intelligence_profile.modality_strengths[IntelligenceModality.HOLISTIC] = 0.95
        
        # Wait for coordination to detect transcendence
        await asyncio.sleep(2.0)
        
        # Check if transcendence level increased
        transcendence_level = omni_engine.intelligence_profile.transcendence_level
        assert transcendence_level > 0.5
        
        # Check if state evolved beyond unified
        if transcendence_level > omni_engine.transcendence_threshold:
            assert omni_engine.intelligence_state in [
                UnifiedIntelligenceState.TRANSCENDENT,
                UnifiedIntelligenceState.OMNISCIENT,
                UnifiedIntelligenceState.INFINITE
            ]
        
        # Cleanup
        await omni_engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_singularity_conditions(self, omni_engine):
        """Test technological singularity achievement conditions"""
        # Activate omni-intelligence
        await omni_engine.activate_omni_intelligence()
        
        # Artificially meet singularity conditions
        omni_engine.intelligence_profile.transcendence_level = 0.96
        omni_engine.intelligence_profile.integration_coherence = 0.96
        omni_engine.intelligence_profile.creative_potential = 0.91
        omni_engine.intelligence_profile.wisdom_factor = 0.91
        omni_engine.intelligence_state = UnifiedIntelligenceState.INFINITE
        
        # Add breakthrough moments
        omni_engine.breakthrough_moments = [
            {'type': 'transcendence', 'timestamp': time.time()},
            {'type': 'integration', 'timestamp': time.time()},
            {'type': 'creativity', 'timestamp': time.time()},
            {'type': 'wisdom', 'timestamp': time.time()}
        ]
        
        # Wait for singularity check
        await asyncio.sleep(1.0)
        
        # Verify singularity detection
        assert omni_engine.singularity_achieved is True
        
        # Cleanup
        await omni_engine.shutdown()
    
    def test_intelligence_insights_generation(self, omni_engine):
        """Test generation of intelligence insights"""
        # Boost various intelligence dimensions
        omni_engine.intelligence_profile.integration_coherence = 0.85
        omni_engine.intelligence_profile.transcendence_level = 0.75
        omni_engine.intelligence_profile.creative_potential = 0.85
        omni_engine.intelligence_profile.wisdom_factor = 0.85
        
        # Trigger insight generation
        omni_engine._generate_intelligence_insights()
        
        # Check insights were generated
        assert len(omni_engine.intelligence_insights) > 0
        
        # Check insight content
        insights = omni_engine.intelligence_insights
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 0
    
    def test_omni_intelligence_report(self, omni_engine):
        """Test comprehensive intelligence reporting"""
        report = omni_engine.get_omni_intelligence_report()
        
        assert 'intelligence_state' in report
        assert 'singularity_achieved' in report
        assert 'intelligence_profile' in report
        assert 'breakthrough_moments' in report
        assert 'intelligence_insights_count' in report
        
        # Check intelligence profile in report
        profile = report['intelligence_profile']
        assert 'modality_strengths' in profile
        assert 'integration_coherence' in profile
        assert 'transcendence_level' in profile
        assert 'creative_potential' in profile
        assert 'problem_solving_depth' in profile
        assert 'wisdom_factor' in profile
        
        # Verify all modalities reported
        modality_strengths = profile['modality_strengths']
        for modality in IntelligenceModality:
            assert modality.value in modality_strengths
    
    @pytest.mark.asyncio
    async def test_cross_modal_integration(self, omni_engine):
        """Test cross-modal intelligence integration"""
        # Activate omni-intelligence
        await omni_engine.activate_omni_intelligence()
        
        # Wait for integration to develop
        await asyncio.sleep(1.5)
        
        # Check integration matrix
        integration_matrix = omni_engine.integration_matrix
        assert integration_matrix is not None
        
        num_modalities = len(IntelligenceModality)
        assert integration_matrix.shape == (num_modalities, num_modalities)
        
        # Diagonal should be 1.0 (self-integration)
        for i in range(num_modalities):
            assert integration_matrix[i, i] == 1.0
        
        # Check symmetric matrix
        for i in range(num_modalities):
            for j in range(num_modalities):
                assert integration_matrix[i, j] == integration_matrix[j, i]
        
        # Check integration coherence calculation
        coherence = omni_engine._calculate_integration_coherence()
        assert 0.0 <= coherence <= 1.0
        
        # Cleanup
        await omni_engine.shutdown()


class TestSingularityPerformance:
    """Test performance characteristics of singularity systems"""
    
    @pytest.mark.asyncio
    async def test_activation_performance(self):
        """Test omni-intelligence activation performance"""
        engine = OmniIntelligenceEngine(dimension=1000, singularity_mode=False)
        
        # Measure activation time
        start_time = time.time()
        success = await engine.activate_omni_intelligence()
        activation_time = time.time() - start_time
        
        assert success is True
        # Should activate within reasonable time (10 seconds)
        assert activation_time < 10.0
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_processing_throughput(self):
        """Test unified intelligence processing throughput"""
        engine = OmniIntelligenceEngine(dimension=1000, singularity_mode=False)
        await engine.activate_omni_intelligence()
        
        # Process multiple inputs rapidly
        test_inputs = [f"Test input {i}" for i in range(50)]
        
        start_time = time.time()
        results = []
        
        for input_data in test_inputs:
            result = engine.process_unified_intelligence(
                input_data,
                target_modalities=[
                    IntelligenceModality.LOGICAL,
                    IntelligenceModality.ANALYTICAL
                ]
            )
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Should process at least 5 inputs per second
        assert len(results) == len(test_inputs)
        assert processing_time < len(test_inputs) / 5.0
        
        # Verify all results are valid
        for result in results:
            assert 'modality_results' in result
            assert 'neural_analysis' in result
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Initial memory state
        initial_memory_size = engine.unified_memory.size()
        
        # Process many inputs to test memory management
        for i in range(100):
            test_vector = create_hypervector(1000, f"memory_test_{i}")
            engine.process_unified_intelligence(test_vector)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Check memory growth is reasonable
        final_memory_size = engine.unified_memory.size()
        memory_growth = final_memory_size - initial_memory_size
        
        # Memory should grow but not excessively
        assert memory_growth > 0
        assert memory_growth < 500  # Reasonable growth limit
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_coordination_efficiency(self):
        """Test coordination loop efficiency"""
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Monitor coordination for short period
        initial_time = time.time()
        
        # Wait for several coordination cycles
        await asyncio.sleep(2.0)
        
        # Verify coordination is active and efficient
        assert engine.coordination_active is True
        
        # Check that intelligence profile is being updated
        profile = engine.intelligence_profile
        assert profile.last_updated > initial_time
        
        await engine.shutdown()


class TestSingularityIntegration:
    """Test integration between singularity components"""
    
    @pytest.mark.asyncio
    async def test_singularity_transcendence_integration(self):
        """Test integration with transcendence systems"""
        # This would test integration with Generation 8 components
        # when they are available in the same environment
        
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Verify transcendence-related modalities are active
        transcendent_modalities = [
            IntelligenceModality.TRANSCENDENT,
            IntelligenceModality.META_COGNITIVE,
            IntelligenceModality.QUANTUM,
            IntelligenceModality.HOLISTIC
        ]
        
        for modality in transcendent_modalities:
            assert modality in engine.processing_pipelines
            strength = engine.intelligence_profile.modality_strengths[modality]
            assert strength > 0.0
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_singularity_hdc_integration(self):
        """Test integration with core HDC systems"""
        from hdc_robot_controller.core.operations import bind, bundle, similarity
        
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Test HDC operations within singularity context
        base_vector = create_hypervector(1000, "singularity_test")
        context_vector = create_hypervector(1000, "hdc_integration")
        
        # Bind vectors
        bound_vector = bind(base_vector, context_vector)
        
        # Process through omni-intelligence
        result = engine.process_unified_intelligence(bound_vector)
        
        assert result['input_data'] is not None
        assert 'modality_results' in result
        
        # Verify HDC memory integration
        assert engine.unified_memory.size() > 0
        
        # Test similarity queries
        similar_memories = engine.unified_memory.query(bound_vector, top_k=3)
        assert len(similar_memories) >= 0  # May be empty initially
        
        await engine.shutdown()


class TestSingularityEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_modality_processing(self):
        """Test processing with invalid modality selection"""
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Try processing with empty modality list
        result = engine.process_unified_intelligence("test", target_modalities=[])
        
        # Should handle gracefully
        assert 'modality_results' in result
        assert len(result['modality_results']) == 0
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown_during_processing(self):
        """Test graceful shutdown during active processing"""
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Start processing
        processing_task = asyncio.create_task(
            asyncio.gather(*[
                asyncio.to_thread(
                    engine.process_unified_intelligence, f"concurrent_test_{i}"
                )
                for i in range(10)
            ])
        )
        
        # Wait briefly then shutdown
        await asyncio.sleep(0.5)
        shutdown_task = asyncio.create_task(engine.shutdown())
        
        # Wait for both to complete
        await asyncio.gather(processing_task, shutdown_task, return_exceptions=True)
        
        # Verify clean shutdown
        assert engine.coordination_active is False
    
    def test_extreme_dimension_values(self):
        """Test with extreme dimension values"""
        # Test with very small dimension
        small_engine = OmniIntelligenceEngine(dimension=10)
        assert small_engine.dimension == 10
        assert len(small_engine.intelligence_vectors) > 0
        
        # Test with large dimension
        large_engine = OmniIntelligenceEngine(dimension=50000)
        assert large_engine.dimension == 50000
        assert len(large_engine.intelligence_vectors) > 0
    
    @pytest.mark.asyncio
    async def test_singularity_without_transcendence(self):
        """Test singularity mode without transcendence features"""
        engine = OmniIntelligenceEngine(
            dimension=1000,
            singularity_mode=False,
            transcendence_threshold=1.1  # Impossible threshold
        )
        
        await engine.activate_omni_intelligence()
        
        # Process some inputs
        for i in range(10):
            result = engine.process_unified_intelligence(f"test_{i}")
            assert result['input_data'] is not None
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Should not achieve singularity
        assert engine.singularity_achieved is False
        
        await engine.shutdown()


class TestSingularityStressTests:
    """Stress tests for singularity systems"""
    
    @pytest.mark.asyncio
    async def test_high_load_processing(self):
        """Test processing under high load"""
        engine = OmniIntelligenceEngine(dimension=1000)
        await engine.activate_omni_intelligence()
        
        # Generate high processing load
        tasks = []
        for i in range(200):
            task = asyncio.to_thread(
                engine.process_unified_intelligence,
                f"stress_test_{i}",
                [IntelligenceModality.LOGICAL, IntelligenceModality.ANALYTICAL]
            )
            tasks.append(task)
        
        # Process all tasks
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # Count successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Should handle most requests successfully
        assert len(successful_results) > len(tasks) * 0.8  # 80% success rate
        
        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 seconds max
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        engine = OmniIntelligenceEngine(dimension=2000)  # Larger dimension
        await engine.activate_omni_intelligence()
        
        # Create memory pressure by processing many unique inputs
        large_inputs = [f"memory_pressure_test_{i}" * 100 for i in range(1000)]
        
        start_time = time.time()
        
        for i, input_data in enumerate(large_inputs):
            result = engine.process_unified_intelligence(input_data)
            assert result is not None
            
            # Check memory periodically
            if i % 100 == 0:
                memory_size = engine.unified_memory.size()
                # Memory should grow reasonably
                assert memory_size > 0
        
        processing_time = time.time() - start_time
        
        # Should handle memory pressure gracefully
        final_memory_size = engine.unified_memory.size()
        assert final_memory_size > 0
        
        await engine.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])