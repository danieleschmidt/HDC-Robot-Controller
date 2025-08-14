"""
Comprehensive test suite for Advanced Intelligence capabilities.

Tests Generation 4 enhancements including multi-modal fusion,
quantum-inspired HDC, neural-HDC hybrids, symbolic reasoning,
and meta-learning systems.
"""

import pytest
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
import time
from pathlib import Path

from hdc_robot_controller.advanced_intelligence import (
    MultiModalFusionEngine, 
    QuantumInspiredHDC,
    NeuralHDCHybrid,
    AdvancedSymbolicReasoner,
    MetaLearningEngine
)
from hdc_robot_controller.advanced_intelligence.multi_modal_fusion import (
    ModalityConfig, TransformerHDCEncoder
)
from hdc_robot_controller.advanced_intelligence.quantum_hdc import (
    QuantumState, QuantumHDCGate
)
from hdc_robot_controller.advanced_intelligence.neural_hdc_hybrid import (
    HybridConfig
)
from hdc_robot_controller.advanced_intelligence.meta_learner import (
    Task, Episode, PrototypicalNetwork
)
from hdc_robot_controller.core.hypervector import HyperVector


class TestMultiModalFusion:
    """Test multi-modal fusion capabilities."""
    
    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine for testing."""
        modality_configs = [
            ModalityConfig(
                name="vision",
                dimension=256,
                encoder_type="transformer",
                attention_heads=8
            ),
            ModalityConfig(
                name="audio", 
                dimension=128,
                encoder_type="transformer",
                attention_heads=4
            ),
            ModalityConfig(
                name="tactile",
                dimension=64,
                encoder_type="transformer",
                attention_heads=2
            )
        ]
        
        return MultiModalFusionEngine(
            modality_configs=modality_configs,
            hdc_dimension=1000,
            fusion_strategy="hierarchical_attention"
        )
    
    def test_modality_encoding(self, fusion_engine):
        """Test individual modality encoding."""
        # Test vision modality
        vision_data = np.random.randn(32, 256)  # Batch of 32, 256 features
        neural_feat, hypervec = fusion_engine.encode_modality("vision", vision_data)
        
        assert neural_feat.shape[0] == 1  # Batch size
        assert neural_feat.shape[1] == 512  # Hidden dimension
        assert isinstance(hypervec, HyperVector)
        assert hypervec.dimension == 1000
        
    def test_multi_modal_fusion(self, fusion_engine):
        """Test fusion of multiple modalities."""
        modality_data = {
            "vision": np.random.randn(256),
            "audio": np.random.randn(128),
            "tactile": np.random.randn(64)
        }
        
        results = fusion_engine.fuse_modalities(modality_data)
        
        assert 'neural_features' in results
        assert 'hypervector' in results
        assert 'modality_weights' in results
        assert 'fusion_time' in results
        assert 'confidence' in results
        
        # Check fusion confidence
        assert 0.0 <= results['confidence'] <= 1.0
        
        # Check weight normalization
        weights = list(results['modality_weights'].values())
        assert abs(sum(weights) - 1.0) < 1e-6
        
    def test_adaptive_weighting(self, fusion_engine):
        """Test adaptive modality weighting."""
        # High quality vision data
        high_quality_vision = np.random.randn(256) * 0.1  # Low noise
        low_quality_audio = np.random.randn(128) * 2.0   # High noise
        
        modality_data = {
            "vision": high_quality_vision,
            "audio": low_quality_audio
        }
        
        results = fusion_engine.fuse_modalities(
            modality_data, 
            adaptive_weighting=True
        )
        
        # Vision should get higher weight due to higher quality
        assert results['modality_weights']['vision'] > results['modality_weights']['audio']
        
    def test_context_application(self, fusion_engine):
        """Test contextual fusion."""
        modality_data = {
            "vision": np.random.randn(256)
        }
        
        context = {
            "environment": "indoor",
            "lighting": "low",
            "modality_importance": {"vision": 0.8}
        }
        
        results = fusion_engine.fuse_modalities(modality_data, context=context)
        
        # Should incorporate context information
        assert 'neural_features' in results
        assert results['confidence'] > 0


class TestQuantumInspiredHDC:
    """Test quantum-inspired HDC operations."""
    
    @pytest.fixture
    def quantum_hdc(self):
        """Create quantum HDC system."""
        return QuantumInspiredHDC(
            dimension=1000,
            enable_superposition=True,
            enable_entanglement=True,
            enable_interference=True
        )
    
    def test_quantum_state_creation(self, quantum_hdc):
        """Test quantum state creation from classical hypervectors."""
        classical_hv = HyperVector.random(1000, seed=42)
        quantum_state = quantum_hdc.create_quantum_hypervector(classical_hv)
        
        assert isinstance(quantum_state, QuantumState)
        assert quantum_state.dimension == 1000
        assert len(quantum_state.amplitudes) == 1000
        
        # Check normalization
        norm = np.linalg.norm(quantum_state.amplitudes)
        assert abs(norm - 1.0) < 1e-6
        
    def test_quantum_operations(self, quantum_hdc):
        """Test quantum bundle and bind operations."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        qs1 = quantum_hdc.create_quantum_hypervector(hv1)
        qs2 = quantum_hdc.create_quantum_hypervector(hv2)
        
        # Test quantum bundling
        bundled = quantum_hdc.quantum_bundle([qs1, qs2])
        assert isinstance(bundled, QuantumState)
        
        # Test quantum binding
        bound, entangle_id = quantum_hdc.quantum_bind(qs1, qs2)
        assert isinstance(bound, QuantumState)
        
        if quantum_hdc.enable_entanglement:
            assert entangle_id is not None
            assert entangle_id in quantum_hdc.entanglement_registry
        
    def test_quantum_similarity(self, quantum_hdc):
        """Test quantum similarity computation."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        qs1 = quantum_hdc.create_quantum_hypervector(hv1)
        qs2 = quantum_hdc.create_quantum_hypervector(hv2)
        
        similarity = quantum_hdc.quantum_similarity(qs1, qs2)
        
        # Should be complex number
        assert isinstance(similarity, complex)
        
        # Magnitude should be reasonable
        assert 0 <= abs(similarity) <= 1.5  # Allow some margin for complex similarities
        
    def test_quantum_measurement(self, quantum_hdc):
        """Test quantum measurement and collapse."""
        hv = HyperVector.random(1000, seed=42)
        qs = quantum_hdc.create_quantum_hypervector(hv)
        
        # Measure in computational basis
        measured_hv = quantum_hdc.measure_quantum_state(qs)
        
        assert isinstance(measured_hv, HyperVector)
        assert measured_hv.dimension == 1000
        assert np.all(np.isin(measured_hv.data, [-1, 1]))  # Bipolar
        
    def test_quantum_learning(self, quantum_hdc):
        """Test quantum-inspired learning."""
        # Create simple training data
        training_data = [
            (np.array([1.0, 0.0]), np.array([1, -1])),
            (np.array([0.0, 1.0]), np.array([-1, 1])),
            (np.array([1.0, 1.0]), np.array([1, 1]))
        ]
        
        results = quantum_hdc.quantum_learning(training_data, num_epochs=10)
        
        assert 'learned_state' in results
        assert 'optimization_history' in results
        assert isinstance(results['learned_state'], QuantumState)
        
    def test_quantum_coherence(self, quantum_hdc):
        """Test quantum coherence tracking."""
        # Create entangled states
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        qs1 = quantum_hdc.create_quantum_hypervector(hv1)
        qs2 = quantum_hdc.create_quantum_hypervector(hv2)
        
        _, entangle_id = quantum_hdc.quantum_bind(qs1, qs2, create_entanglement=True)
        
        if entangle_id:
            coherence_metrics = quantum_hdc.check_quantum_coherence()
            
            assert 'average_coherence' in coherence_metrics
            assert 'entanglement_strength' in coherence_metrics
            assert 'decoherence_rate' in coherence_metrics


class TestNeuralHDCHybrid:
    """Test neural-HDC hybrid architecture."""
    
    @pytest.fixture
    def hybrid_model(self):
        """Create hybrid model for testing."""
        config = HybridConfig(
            neural_hidden_dim=256,
            hdc_dimension=1000,
            fusion_strategy="cross_attention",
            enable_bidirectional=True
        )
        return NeuralHDCHybrid(config)
    
    def test_hybrid_forward_pass(self, hybrid_model):
        """Test hybrid forward pass."""
        input_data = torch.randn(1, 256)  # Batch of 1
        
        # Test different modes
        for mode in ["neural", "hdc", "hybrid"]:
            outputs = hybrid_model.forward(input_data, mode=mode)
            
            if mode in ["neural", "hybrid"]:
                assert 'neural_features' in outputs
                assert outputs['neural_features'].shape == (1, 256)
            
            if mode in ["hdc", "hybrid"]:
                assert 'hdc_tensor' in outputs
                assert 'hdc_vectors' in outputs
                assert len(outputs['hdc_vectors']) == 1
                assert isinstance(outputs['hdc_vectors'][0], HyperVector)
            
            if mode == "hybrid":
                assert 'fused_features' in outputs
                assert 'final_output' in outputs
    
    def test_association_learning(self, hybrid_model):
        """Test association learning in hybrid model."""
        input_data = torch.randn(2, 256)
        target_data = torch.randn(2, 256)
        
        learning_stats = hybrid_model.learn_association(
            input_data, target_data, "test_association"
        )
        
        assert 'neural' in learning_stats
        assert 'hdc' in learning_stats
        
        # Check memory storage
        assert "test_association" in hybrid_model.neural_memory
        assert "test_association" in hybrid_model.hdc_memory
        
    def test_association_retrieval(self, hybrid_model):
        """Test association retrieval."""
        # First learn an association
        input_data = torch.randn(3, 256)
        target_data = torch.randn(3, 256)
        
        hybrid_model.learn_association(input_data, target_data, "retrieval_test")
        
        # Then retrieve with query
        query_data = input_data[:1]  # Use first sample as query
        
        results = hybrid_model.retrieve_association(query_data, "retrieval_test")
        
        assert 'neural' in results
        assert 'hdc' in results
        
        # Check retrieval quality
        if 'similarities' in results['neural']:
            neural_similarities = results['neural']['similarities']
            assert len(neural_similarities) > 0
            assert torch.max(neural_similarities) > 0.5  # Should find good match
    
    def test_online_adaptation(self, hybrid_model):
        """Test online adaptation capability."""
        input_data = torch.randn(1, 256)
        target_data = torch.randn(1, 256)
        
        adaptation_stats = hybrid_model.adapt_online(input_data, target_data)
        
        assert 'neural_loss' in adaptation_stats
        assert 'hdc_sparsity' in adaptation_stats
        assert 'hdc_entropy' in adaptation_stats
        
        # Loss should be finite
        assert np.isfinite(adaptation_stats['neural_loss'])
        
    def test_interpretability_analysis(self, hybrid_model):
        """Test interpretability analysis."""
        # Perform some operations to populate the model
        input_data = torch.randn(1, 256)
        hybrid_model.forward(input_data, mode="hybrid")
        
        analysis = hybrid_model.get_interpretability_analysis()
        
        assert 'neural_analysis' in analysis
        assert 'hdc_analysis' in analysis
        assert 'fusion_analysis' in analysis
        assert 'memory_analysis' in analysis


class TestAdvancedSymbolicReasoner:
    """Test symbolic reasoning capabilities."""
    
    @pytest.fixture
    def reasoner(self):
        """Create symbolic reasoner."""
        return AdvancedSymbolicReasoner(hdc_dimension=1000)
    
    def test_concept_creation(self, reasoner):
        """Test concept creation and encoding."""
        concept = reasoner.add_concept(
            "robot", 
            attributes={"type": "mobile", "sensors": ["lidar", "camera"]},
            confidence=0.9
        )
        
        assert concept.name == "robot"
        assert concept.confidence == 0.9
        assert "robot" in reasoner.concepts
        assert isinstance(concept.hypervector, HyperVector)
        
    def test_rule_creation(self, reasoner):
        """Test rule creation and encoding."""
        rule = reasoner.add_rule(
            "navigation_rule",
            premise="obstacle AND near",
            conclusion="stop OR avoid",
            confidence=0.8
        )
        
        assert rule.premise == "obstacle AND near"
        assert rule.conclusion == "stop OR avoid"
        assert rule.confidence == 0.8
        assert "navigation_rule" in reasoner.rules
        
    def test_fact_assertion(self, reasoner):
        """Test fact assertion."""
        fact = reasoner.add_fact(
            "current_state",
            statement="robot_position AND clear_path",
            truth_value=0.9
        )
        
        assert fact.statement == "robot_position AND clear_path"
        assert fact.truth_value == 0.9
        assert "current_state" in reasoner.facts
        
    def test_forward_reasoning(self, reasoner):
        """Test forward chaining reasoning."""
        # Set up knowledge base
        reasoner.add_concept("obstacle")
        reasoner.add_concept("near")
        reasoner.add_concept("stop")
        
        reasoner.add_rule("safety_rule", "obstacle AND near", "stop", 0.9)
        reasoner.add_fact("sensor_data", "obstacle AND near", 0.8)
        
        # Perform reasoning
        result = reasoner.reason("stop", reasoning_type="forward")
        
        assert 'answer' in result
        assert 'conclusions' in result
        assert 'final_confidence' in result
        assert result['final_confidence'] > 0
        
    def test_backward_reasoning(self, reasoner):
        """Test backward chaining reasoning."""
        # Set up knowledge base
        reasoner.add_concept("goal_reached")
        reasoner.add_concept("path_clear")
        reasoner.add_concept("move_forward")
        
        reasoner.add_rule("movement_rule", "path_clear", "move_forward", 0.9)
        reasoner.add_rule("goal_rule", "move_forward", "goal_reached", 0.8)
        reasoner.add_fact("environment", "path_clear", 0.9)
        
        result = reasoner.reason("goal_reached", reasoning_type="backward")
        
        assert 'answer' in result
        assert 'final_confidence' in result
        
    def test_uncertainty_handling(self, reasoner):
        """Test handling of uncertain facts and rules."""
        reasoner.add_concept("uncertain_sensor")
        reasoner.add_rule("uncertain_rule", "uncertain_sensor", "uncertain_action", 0.3)
        reasoner.add_fact("noisy_data", "uncertain_sensor", 0.4)
        
        result = reasoner.reason("uncertain_action")
        
        # Should handle low confidence appropriately
        assert result['final_confidence'] < 0.5
        
    def test_explanation_generation(self, reasoner):
        """Test reasoning explanation generation."""
        reasoner.add_rule("simple_rule", "A", "B", 1.0)
        reasoner.add_fact("fact_A", "A", 1.0)
        
        result = reasoner.reason("B")
        explanation = reasoner.explain_reasoning(result)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "REASONING STEPS" in explanation or "CONCLUSION" in explanation


class TestMetaLearningEngine:
    """Test meta-learning capabilities."""
    
    @pytest.fixture
    def meta_learner(self):
        """Create meta-learning engine."""
        return MetaLearningEngine(
            input_dim=20,
            output_dim=5,
            hdc_dim=1000,
            meta_learning_algorithm="hybrid",
            inner_steps=3
        )
    
    def test_task_creation(self, meta_learner):
        """Test task creation and encoding."""
        # Create synthetic task data
        input_data = np.random.randn(100, 20)
        output_data = np.random.randint(0, 5, 100)
        
        task = Task(
            task_id="test_task",
            task_type="classification",
            input_data=input_data,
            output_data=output_data,
            metadata={"difficulty": 0.5}
        )
        
        # Encode task
        task_hv = meta_learner.hdc_encoder.encode_task(task)
        
        assert isinstance(task_hv, HyperVector)
        assert task_hv.dimension == 1000
        assert "test_task" in meta_learner.hdc_encoder.task_registry
        
    def test_episode_creation(self, meta_learner):
        """Test few-shot episode creation."""
        input_data = np.random.randn(50, 20)
        output_data = np.random.randint(0, 3, 50)  # 3 classes
        
        task = Task(
            task_id="episode_task",
            task_type="few_shot",
            input_data=input_data,
            output_data=output_data,
            metadata={}
        )
        
        episode = meta_learner.create_episode(task, n_way=3, k_shot=2, n_query=5)
        
        assert isinstance(episode, Episode)
        assert len(episode.support_set) == 2  # (X, y)
        assert len(episode.query_set) == 2    # (X, y)
        
        support_X, support_y = episode.support_set
        query_X, query_y = episode.query_set
        
        # Check episode structure
        assert len(support_X) == 6  # 3 classes × 2 shots
        assert len(support_y) == 6
        assert len(np.unique(support_y)) <= 3  # At most 3 classes
        
    def test_fast_adaptation(self, meta_learner):
        """Test fast adaptation to new tasks."""
        # Create adaptation task
        input_data = np.random.randn(30, 20)
        output_data = np.random.randint(0, 2, 30)  # Binary task
        
        task = Task(
            task_id="adaptation_task",
            task_type="adaptation",
            input_data=input_data,
            output_data=output_data,
            metadata={"difficulty": 0.3}
        )
        
        results = meta_learner.fast_adapt(task, n_way=2, k_shot=1)
        
        assert 'accuracy' in results
        assert 'adaptation_time' in results
        assert 'adaptation_strategy' in results
        assert results['adaptation_time'] > 0
        
    def test_similar_task_finding(self, meta_learner):
        """Test finding similar tasks."""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            input_data = np.random.randn(20, 20)
            output_data = np.random.randint(0, 2, 20)
            
            task = Task(
                task_id=f"task_{i}",
                task_type="classification",
                input_data=input_data,
                output_data=output_data,
                metadata={"difficulty": 0.5 + i * 0.1}
            )
            
            # Register task
            meta_learner.hdc_encoder.encode_task(task)
            tasks.append(task)
        
        # Find similar tasks to first task
        similar = meta_learner.hdc_encoder.find_similar_tasks(tasks[0], top_k=2)
        
        assert len(similar) <= 2
        for task_id, similarity in similar:
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1
    
    def test_performance_tracking(self, meta_learner):
        """Test performance metrics tracking."""
        summary = meta_learner.get_performance_summary()
        
        assert 'meta_metrics' in summary
        assert 'memory_stats' in summary
        assert 'algorithm_config' in summary
        
        # Check metric structure
        metrics = summary['meta_metrics']
        expected_metrics = [
            'episodes_trained', 'tasks_learned', 'adaptations_performed',
            'average_adaptation_time', 'best_few_shot_accuracy'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics


class TestIntegrationScenarios:
    """Integration tests for combined capabilities."""
    
    def test_multi_modal_symbolic_reasoning(self):
        """Test integration of multi-modal fusion with symbolic reasoning."""
        # Create multi-modal fusion engine
        modality_configs = [
            ModalityConfig(name="vision", dimension=128, encoder_type="transformer")
        ]
        fusion_engine = MultiModalFusionEngine(modality_configs, hdc_dimension=1000)
        
        # Create symbolic reasoner
        reasoner = AdvancedSymbolicReasoner(hdc_dimension=1000)
        
        # Process multi-modal data
        vision_data = np.random.randn(128)
        fusion_results = fusion_engine.fuse_modalities({"vision": vision_data})
        
        # Convert to symbolic representation
        reasoner.add_concept("visual_input", confidence=fusion_results['confidence'])
        reasoner.add_rule("perception_rule", "visual_input", "object_detected", 0.8)
        
        # Perform reasoning
        reasoning_result = reasoner.reason("object_detected")
        
        assert reasoning_result['final_confidence'] > 0
        
    def test_quantum_neural_hybrid(self):
        """Test integration of quantum HDC with neural networks."""
        # Create quantum HDC
        quantum_hdc = QuantumInspiredHDC(dimension=500)
        
        # Create neural-HDC hybrid
        config = HybridConfig(neural_hidden_dim=128, hdc_dimension=500)
        hybrid_model = NeuralHDCHybrid(config)
        
        # Create classical hypervector
        classical_hv = HyperVector.random(500, seed=42)
        
        # Convert to quantum state
        quantum_state = quantum_hdc.create_quantum_hypervector(classical_hv)
        
        # Measure back to classical
        measured_hv = quantum_hdc.measure_quantum_state(quantum_state)
        
        # Process with hybrid model
        input_tensor = torch.randn(1, 128)
        hybrid_outputs = hybrid_model.forward(input_tensor, mode="hybrid")
        
        # Verify integration works
        assert isinstance(measured_hv, HyperVector)
        assert 'final_output' in hybrid_outputs
        
    def test_meta_learning_with_quantum_hdc(self):
        """Test meta-learning with quantum-enhanced HDC."""
        # Create meta-learner
        meta_learner = MetaLearningEngine(input_dim=10, output_dim=3, hdc_dim=500)
        
        # Create quantum HDC
        quantum_hdc = QuantumInspiredHDC(dimension=500)
        
        # Create tasks with quantum enhancement
        input_data = np.random.randn(20, 10)
        output_data = np.random.randint(0, 3, 20)
        
        task = Task(
            task_id="quantum_enhanced_task",
            task_type="quantum_classification",
            input_data=input_data,
            output_data=output_data,
            metadata={"quantum_enhanced": True}
        )
        
        # Encode task with both systems
        classical_task_hv = meta_learner.hdc_encoder.encode_task(task)
        quantum_task_state = quantum_hdc.create_quantum_hypervector(classical_task_hv)
        
        # Verify quantum enhancement
        quantum_similarity = quantum_hdc.quantum_similarity(quantum_task_state, quantum_task_state)
        assert abs(quantum_similarity) > 0.9  # High self-similarity


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for advanced intelligence."""
    
    def test_fusion_latency(self):
        """Test multi-modal fusion latency."""
        modality_configs = [
            ModalityConfig(name="test", dimension=256, encoder_type="transformer")
        ]
        fusion_engine = MultiModalFusionEngine(modality_configs, hdc_dimension=1000)
        
        # Benchmark fusion
        data = {"test": np.random.randn(256)}
        
        start_time = time.time()
        for _ in range(10):
            fusion_engine.fuse_modalities(data)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / 10
        assert avg_latency < 0.1  # Should be under 100ms per fusion
        
    def test_quantum_operation_speed(self):
        """Test quantum HDC operation speed."""
        quantum_hdc = QuantumInspiredHDC(dimension=1000)
        
        # Create test states
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        qs1 = quantum_hdc.create_quantum_hypervector(hv1)
        qs2 = quantum_hdc.create_quantum_hypervector(hv2)
        
        # Benchmark quantum operations
        start_time = time.time()
        for _ in range(100):
            quantum_hdc.quantum_bundle([qs1, qs2])
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be under 10ms per operation
        
    def test_reasoning_speed(self):
        """Test symbolic reasoning speed."""
        reasoner = AdvancedSymbolicReasoner(hdc_dimension=1000)
        
        # Set up knowledge base
        for i in range(10):
            reasoner.add_concept(f"concept_{i}")
            reasoner.add_rule(f"rule_{i}", f"concept_{i}", f"concept_{i+1}", 0.9)
            reasoner.add_fact(f"fact_{i}", f"concept_{i}", 0.8)
        
        # Benchmark reasoning
        start_time = time.time()
        for _ in range(10):
            reasoner.reason("concept_5", max_steps=5)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.05  # Should be under 50ms per reasoning session


if __name__ == "__main__":
    # Run tests with performance benchmarks
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
    print("\\nRunning performance benchmarks...")
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])