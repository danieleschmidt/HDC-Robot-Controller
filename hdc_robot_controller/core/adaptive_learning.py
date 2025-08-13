"""
Adaptive Learning Engine for HDC Robot Controller
Enhanced one-shot and few-shot learning capabilities with meta-learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
import pickle
from pathlib import Path
import logging

from .hypervector import HyperVector, weighted_bundle
from .memory import AssociativeMemory

logger = logging.getLogger(__name__)


@dataclass
class LearningContext:
    """Context information for adaptive learning."""
    environment_id: str
    task_type: str
    sensor_modalities: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class LearningExample:
    """Single learning example with context."""
    state_vector: HyperVector
    action_vector: HyperVector
    reward: float
    context: LearningContext
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveLearningEngine:
    """
    Adaptive Learning Engine for HDC-based robotic control.
    
    Implements one-shot, few-shot, and continual learning capabilities
    with meta-learning for rapid adaptation to new environments.
    """
    
    def __init__(self, 
                 dimension: int = 10000,
                 memory_capacity: int = 10000,
                 adaptation_rate: float = 0.1,
                 meta_learning_enabled: bool = True):
        """
        Initialize the adaptive learning engine.
        
        Args:
            dimension: Hypervector dimension
            memory_capacity: Maximum number of examples to store
            adaptation_rate: Learning rate for adaptation
            meta_learning_enabled: Enable meta-learning capabilities
        """
        self.dimension = dimension
        self.memory_capacity = memory_capacity
        self.adaptation_rate = adaptation_rate
        self.meta_learning_enabled = meta_learning_enabled
        
        # Core components
        self.episodic_memory = AssociativeMemory(dimension, similarity_threshold=0.7)
        self.behavior_library = {}  # Learned behaviors
        self.meta_parameters = {}  # Meta-learning parameters
        
        # Learning statistics
        self.learning_stats = {
            'one_shot_successes': 0,
            'few_shot_adaptations': 0,
            'total_examples': 0,
            'adaptation_times': [],
            'success_rates': []
        }
        
        # Initialize meta-learning components
        if meta_learning_enabled:
            self._initialize_meta_learning()
            
    def _initialize_meta_learning(self):
        """Initialize meta-learning parameters and structures."""
        # Meta-learning hyperparameters
        self.meta_parameters = {
            'adaptation_strategies': self._create_adaptation_strategies(),
            'task_embeddings': {},
            'context_patterns': {},
            'learning_rates': np.ones(self.dimension) * self.adaptation_rate
        }
        
    def _create_adaptation_strategies(self) -> Dict[str, HyperVector]:
        """Create adaptation strategy hypervectors."""
        strategies = {}
        
        # Different adaptation strategies
        strategy_types = [
            'imitation', 'reinforcement', 'curiosity_driven',
            'safety_first', 'efficiency_focused', 'exploration'
        ]
        
        for strategy in strategy_types:
            strategies[strategy] = HyperVector.random(self.dimension, seed=abs(hash(strategy)) % (2**31))
            
        return strategies
        
    def learn_from_demonstration(self, 
                                state_action_pairs: List[Tuple[HyperVector, HyperVector]],
                                task_name: str,
                                context: LearningContext,
                                reward_signal: Optional[float] = None) -> bool:
        """
        Learn a new behavior from demonstration (one-shot learning).
        
        Args:
            state_action_pairs: List of (state, action) hypervector pairs
            task_name: Name identifier for the task
            context: Learning context information
            reward_signal: Optional reward/success signal
            
        Returns:
            Success status of learning
        """
        start_time = time.time()
        
        try:
            # Create behavior hypervector by bundling state-action pairs
            behavior_components = []
            
            for i, (state_hv, action_hv) in enumerate(state_action_pairs):
                # Bind state with action
                state_action = state_hv.bind(action_hv)
                
                # Add temporal position information
                position_hv = HyperVector.random(self.dimension, seed=i + 1000)
                temporal_component = state_action.bind(position_hv)
                
                behavior_components.append(temporal_component)
                
            # Bundle all components into behavior representation
            if behavior_components:
                behavior_hv = HyperVector.bundle_vectors(behavior_components)
                
                # Apply meta-learning if enabled
                if self.meta_learning_enabled:
                    behavior_hv = self._apply_meta_learning(
                        behavior_hv, context, task_name
                    )
                
                # Store in behavior library
                self.behavior_library[task_name] = {
                    'behavior_vector': behavior_hv,
                    'context': context,
                    'examples': state_action_pairs,
                    'reward': reward_signal,
                    'creation_time': time.time(),
                    'usage_count': 0
                }
                
                # Store examples in episodic memory
                for state_hv, action_hv in state_action_pairs:
                    example = LearningExample(
                        state_vector=state_hv,
                        action_vector=action_hv,
                        reward=reward_signal or 0.0,
                        context=context
                    )
                    self._store_example(example)
                
                # Update statistics
                self.learning_stats['one_shot_successes'] += 1
                self.learning_stats['total_examples'] += len(state_action_pairs)
                
                adaptation_time = time.time() - start_time
                self.learning_stats['adaptation_times'].append(adaptation_time)
                
                logger.info(f"Learned behavior '{task_name}' from {len(state_action_pairs)} examples in {adaptation_time:.3f}s")
                
                return True
            else:
                logger.warning("No valid state-action pairs provided for learning")
                return False
                
        except Exception as e:
            logger.error(f"Failed to learn from demonstration: {e}")
            return False
            
    def adapt_behavior(self, 
                      base_behavior: str,
                      adaptation_examples: List[LearningExample],
                      new_behavior_name: str,
                      adaptation_strength: float = 0.3) -> bool:
        """
        Adapt existing behavior to new context (few-shot learning).
        
        Args:
            base_behavior: Name of base behavior to adapt
            adaptation_examples: New examples for adaptation
            new_behavior_name: Name for adapted behavior
            adaptation_strength: Strength of adaptation (0.0 to 1.0)
            
        Returns:
            Success status of adaptation
        """
        if base_behavior not in self.behavior_library:
            logger.error(f"Base behavior '{base_behavior}' not found")
            return False
            
        start_time = time.time()
        
        try:
            base_behavior_data = self.behavior_library[base_behavior]
            base_hv = base_behavior_data['behavior_vector']
            
            # Create adaptation vector from new examples
            adaptation_components = []
            
            for example in adaptation_examples:
                state_action = example.state_vector.bind(example.action_vector)
                adaptation_components.append(state_action)
                
            if adaptation_components:
                adaptation_hv = HyperVector.bundle_vectors(adaptation_components)
                
                # Blend base behavior with adaptation
                adapted_hv = weighted_bundle([
                    (base_hv, 1.0 - adaptation_strength),
                    (adaptation_hv, adaptation_strength)
                ])
                
                # Store adapted behavior
                self.behavior_library[new_behavior_name] = {
                    'behavior_vector': adapted_hv,
                    'context': adaptation_examples[0].context,
                    'base_behavior': base_behavior,
                    'adaptation_examples': adaptation_examples,
                    'adaptation_strength': adaptation_strength,
                    'creation_time': time.time(),
                    'usage_count': 0
                }
                
                # Update statistics
                self.learning_stats['few_shot_adaptations'] += 1
                adaptation_time = time.time() - start_time
                self.learning_stats['adaptation_times'].append(adaptation_time)
                
                logger.info(f"Adapted behavior '{base_behavior}' to '{new_behavior_name}' with {len(adaptation_examples)} examples")
                
                return True
            else:
                logger.warning("No adaptation examples provided")
                return False
                
        except Exception as e:
            logger.error(f"Failed to adapt behavior: {e}")
            return False
            
    def execute_behavior(self, 
                        behavior_name: str,
                        current_state: HyperVector,
                        context: Optional[LearningContext] = None) -> Optional[HyperVector]:
        """
        Execute learned behavior given current state.
        
        Args:
            behavior_name: Name of behavior to execute
            current_state: Current state hypervector
            context: Optional execution context
            
        Returns:
            Action hypervector or None if behavior not found
        """
        if behavior_name not in self.behavior_library:
            logger.warning(f"Behavior '{behavior_name}' not found")
            return None
            
        try:
            behavior_data = self.behavior_library[behavior_name]
            behavior_hv = behavior_data['behavior_vector']
            
            # Update usage count
            behavior_data['usage_count'] += 1
            
            # Query episodic memory for similar states
            similar_examples = self.episodic_memory.query(
                current_state, 
                top_k=5,
                threshold=0.7
            )
            
            if similar_examples:
                # Use most similar example's action as base
                best_match = similar_examples[0]
                action_hv = best_match['action_vector']
                
                # Apply behavior-specific modifications
                behavior_influence = current_state.bind(behavior_hv)
                
                # Blend retrieved action with behavior influence
                final_action = weighted_bundle([
                    (action_hv, 0.7),
                    (behavior_influence, 0.3)
                ])
                
                return final_action
            else:
                # No similar examples, use behavior vector directly
                logger.info(f"No similar states found, using behavior vector directly")
                return behavior_hv
                
        except Exception as e:
            logger.error(f"Failed to execute behavior '{behavior_name}': {e}")
            return None
            
    def _apply_meta_learning(self, 
                           behavior_hv: HyperVector,
                           context: LearningContext,
                           task_name: str) -> HyperVector:
        """Apply meta-learning to enhance behavior representation."""
        try:
            # Create task embedding
            task_embedding = self._create_task_embedding(context, task_name)
            
            # Apply task-specific adaptation strategy
            adaptation_strategy = self._select_adaptation_strategy(context)
            
            # Enhance behavior with meta-learned components
            enhanced_hv = weighted_bundle([
                (behavior_hv, 0.8),
                (task_embedding, 0.1),
                (adaptation_strategy, 0.1)
            ])
            
            return enhanced_hv
            
        except Exception as e:
            logger.error(f"Meta-learning application failed: {e}")
            return behavior_hv
            
    def _create_task_embedding(self, 
                             context: LearningContext,
                             task_name: str) -> HyperVector:
        """Create task embedding hypervector."""
        # Combine context elements
        context_elements = []
        
        # Environment embedding
        env_hv = HyperVector.random(self.dimension, seed=abs(hash(context.environment_id)) % (2**31))
        context_elements.append(env_hv)
        
        # Task type embedding
        task_type_hv = HyperVector.random(self.dimension, seed=abs(hash(context.task_type)) % (2**31))
        context_elements.append(task_type_hv)
        
        # Sensor modalities
        for modality in context.sensor_modalities:
            modality_hv = HyperVector.random(self.dimension, seed=abs(hash(modality)) % (2**31))
            context_elements.append(modality_hv)
            
        # Bundle all context elements
        if context_elements:
            task_embedding = HyperVector.bundle_vectors(context_elements)
        else:
            task_embedding = HyperVector.random(self.dimension, seed=abs(hash(task_name)) % (2**31))
            
        return task_embedding
        
    def _select_adaptation_strategy(self, context: LearningContext) -> HyperVector:
        """Select appropriate adaptation strategy based on context."""
        # Simple heuristic-based strategy selection
        if 'safety' in context.constraints:
            return self.meta_parameters['adaptation_strategies']['safety_first']
        elif 'efficiency' in context.constraints:
            return self.meta_parameters['adaptation_strategies']['efficiency_focused']
        elif context.task_type == 'exploration':
            return self.meta_parameters['adaptation_strategies']['exploration']
        else:
            return self.meta_parameters['adaptation_strategies']['imitation']
            
    def _store_example(self, example: LearningExample):
        """Store learning example in episodic memory."""
        try:
            # Create composite key from state and context
            composite_key = example.state_vector.bind(
                HyperVector.random(self.dimension, seed=abs(hash(example.context.environment_id)) % (2**31))
            )
            
            example_data = {
                'state_vector': example.state_vector,
                'action_vector': example.action_vector,
                'reward': example.reward,
                'context': example.context,
                'metadata': example.metadata
            }
            
            self.episodic_memory.store(composite_key, example_data)
            
        except Exception as e:
            logger.error(f"Failed to store example: {e}")
            
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        stats = self.learning_stats.copy()
        
        # Add computed metrics
        stats['behavior_count'] = len(self.behavior_library)
        stats['memory_utilization'] = len(self.episodic_memory.memory) / self.memory_capacity
        
        if stats['adaptation_times']:
            stats['avg_adaptation_time'] = np.mean(stats['adaptation_times'])
            stats['max_adaptation_time'] = np.max(stats['adaptation_times'])
            stats['min_adaptation_time'] = np.min(stats['adaptation_times'])
            
        return stats
        
    def save_learned_behaviors(self, filepath: Union[str, Path]):
        """Save learned behaviors to file."""
        try:
            data = {
                'behavior_library': self.behavior_library,
                'meta_parameters': self.meta_parameters,
                'learning_stats': self.learning_stats,
                'dimension': self.dimension
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Saved {len(self.behavior_library)} behaviors to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save behaviors: {e}")
            
    def load_learned_behaviors(self, filepath: Union[str, Path]):
        """Load learned behaviors from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.behavior_library = data.get('behavior_library', {})
            self.meta_parameters = data.get('meta_parameters', {})
            self.learning_stats = data.get('learning_stats', {})
            
            logger.info(f"Loaded {len(self.behavior_library)} behaviors from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load behaviors: {e}")
            
    def continual_learning_update(self, 
                                new_examples: List[LearningExample],
                                forgetting_factor: float = 0.01):
        """
        Update behaviors with new examples using continual learning.
        
        Args:
            new_examples: New learning examples
            forgetting_factor: Rate of forgetting old knowledge
        """
        for example in new_examples:
            # Store new example
            self._store_example(example)
            
            # Update relevant behaviors
            self._update_behaviors_with_example(example, forgetting_factor)
            
    def _update_behaviors_with_example(self, 
                                     example: LearningExample,
                                     forgetting_factor: float):
        """Update relevant behaviors with new example."""
        state_hv = example.state_vector
        action_hv = example.action_vector
        
        # Find behaviors that might be relevant
        for behavior_name, behavior_data in self.behavior_library.items():
            # Check if contexts are compatible
            if self._contexts_compatible(behavior_data['context'], example.context):
                # Create state-action binding
                new_component = state_hv.bind(action_hv)
                
                # Update behavior with new component
                old_behavior = behavior_data['behavior_vector']
                updated_behavior = weighted_bundle([
                    (old_behavior, 1.0 - forgetting_factor),
                    (new_component, forgetting_factor)
                ])
                
                behavior_data['behavior_vector'] = updated_behavior
                behavior_data['usage_count'] += 1
                
    def _contexts_compatible(self, 
                           context1: LearningContext,
                           context2: LearningContext) -> bool:
        """Check if two learning contexts are compatible."""
        # Simple compatibility check based on environment and task type
        return (context1.environment_id == context2.environment_id and
                context1.task_type == context2.task_type)
                
    def prune_behaviors(self, usage_threshold: int = 5, age_threshold: float = 86400):
        """
        Prune unused or old behaviors to maintain memory efficiency.
        
        Args:
            usage_threshold: Minimum usage count to keep behavior
            age_threshold: Maximum age in seconds to keep behavior
        """
        current_time = time.time()
        behaviors_to_remove = []
        
        for behavior_name, behavior_data in self.behavior_library.items():
            usage_count = behavior_data.get('usage_count', 0)
            creation_time = behavior_data.get('creation_time', current_time)
            age = current_time - creation_time
            
            if usage_count < usage_threshold and age > age_threshold:
                behaviors_to_remove.append(behavior_name)
                
        for behavior_name in behaviors_to_remove:
            del self.behavior_library[behavior_name]
            logger.info(f"Pruned unused behavior: {behavior_name}")
            
        logger.info(f"Pruned {len(behaviors_to_remove)} behaviors")


class MetaLearningOptimizer:
    """Meta-learning optimizer for adaptive learning parameters."""
    
    def __init__(self, learning_engine: AdaptiveLearningEngine):
        self.learning_engine = learning_engine
        self.optimization_history = []
        
    def optimize_adaptation_rates(self, 
                                performance_feedback: List[float],
                                optimization_steps: int = 50) -> bool:
        """
        Optimize adaptation rates based on performance feedback.
        
        Args:
            performance_feedback: List of performance scores
            optimization_steps: Number of optimization steps
            
        Returns:
            Success status of optimization
        """
        try:
            # Current adaptation rate
            current_rate = self.learning_engine.adaptation_rate
            
            # Performance-based optimization
            best_rate = current_rate
            best_performance = np.mean(performance_feedback) if performance_feedback else 0.0
            
            for step in range(optimization_steps):
                # Generate candidate rates
                candidate_rate = current_rate * (1.0 + 0.1 * np.random.normal())
                candidate_rate = np.clip(candidate_rate, 0.01, 0.5)
                
                # Simulate performance with candidate rate
                simulated_performance = self._simulate_performance(candidate_rate)
                
                if simulated_performance > best_performance:
                    best_performance = simulated_performance
                    best_rate = candidate_rate
                    
            # Update learning engine
            self.learning_engine.adaptation_rate = best_rate
            
            self.optimization_history.append({
                'step': len(self.optimization_history),
                'old_rate': current_rate,
                'new_rate': best_rate,
                'performance_improvement': best_performance - np.mean(performance_feedback)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Meta-learning optimization failed: {e}")
            return False
            
    def _simulate_performance(self, adaptation_rate: float) -> float:
        """Simulate performance with given adaptation rate."""
        # Simple performance model based on adaptation rate
        # In practice, this would involve actual testing
        
        # Optimal rate around 0.1-0.2
        optimal_rate = 0.15
        performance = 1.0 - abs(adaptation_rate - optimal_rate) / optimal_rate
        
        # Add some noise
        performance += 0.1 * np.random.normal()
        
        return np.clip(performance, 0.0, 1.0)