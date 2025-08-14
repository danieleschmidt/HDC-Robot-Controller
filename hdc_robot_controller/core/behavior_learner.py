"""
Behavior Learning Engine for HDC Robot Controller.

Implements one-shot learning, few-shot adaptation, and continual learning
using hyperdimensional computing principles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import json
from collections import defaultdict, deque
from .hypervector import HyperVector
from .memory import HierarchicalMemory, AssociativeMemory
from .sensor_encoder import SensorEncoder


class BehaviorLearner:
    """One-shot and few-shot behavior learning using HDC."""
    
    def __init__(self, dimension: int = 10000, similarity_threshold: float = 0.8,
                 adaptation_rate: float = 0.1):
        """Initialize behavior learner.
        
        Args:
            dimension: Hypervector dimension
            similarity_threshold: Minimum similarity for behavior matching
            adaptation_rate: Learning rate for behavior adaptation
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0 and 1")
        if not 0.0 <= adaptation_rate <= 1.0:
            raise ValueError("Adaptation rate must be between 0 and 1")
            
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.adaptation_rate = adaptation_rate
        
        # Memory systems
        self.behavior_memory = AssociativeMemory(dimension, similarity_threshold)
        self.episode_memory = HierarchicalMemory(dimension)
        self.sensor_encoder = SensorEncoder(dimension)
        
        # Learning statistics
        self.learning_stats = {
            'behaviors_learned': 0,
            'demonstrations_processed': 0,
            'adaptations_made': 0,
            'last_learning_time': None
        }
        
        # Behavior execution history
        self.execution_history = deque(maxlen=1000)
        
        # Context tracking
        self.current_context = HyperVector.zero(dimension)
        self.context_history = deque(maxlen=50)
    
    def learn_from_demonstration(self, demonstration: Dict[str, Any], 
                                behavior_name: str,
                                context: Optional[HyperVector] = None) -> float:
        """Learn behavior from single demonstration (one-shot learning).
        
        Args:
            demonstration: Dictionary containing sensor states and actions
                         e.g., {'states': [state1, state2, ...], 'actions': [action1, action2, ...]}
            behavior_name: Name/label for the learned behavior
            context: Optional context hypervector
            
        Returns:
            Confidence score of learned behavior (0.0 to 1.0)
        """
        if not demonstration.get('states') or not demonstration.get('actions'):
            raise ValueError("Demonstration must contain 'states' and 'actions'")
        
        states = demonstration['states']
        actions = demonstration['actions']
        
        if len(states) != len(actions):
            raise ValueError("Number of states must equal number of actions")
        
        # Encode demonstration as sequence of state-action pairs
        state_action_pairs = []
        for state, action in zip(states, actions):
            # Encode state (assuming it's sensor data)
            if isinstance(state, dict):
                state_hv = self.sensor_encoder.encode_multimodal_state(state)
            elif isinstance(state, HyperVector):
                state_hv = state
            else:
                # Generic encoding
                state_hv = self.sensor_encoder.encode_image_features(np.array(state).flatten())
            
            # Encode action
            if isinstance(action, (list, np.ndarray)):
                action_data = np.array(action).flatten()
                action_hv = self.sensor_encoder.encode_image_features(action_data)
            elif isinstance(action, HyperVector):
                action_hv = action
            else:
                action_hv = self.sensor_encoder.basis_vectors.encode_category(str(action))
            
            # Bind state and action
            state_action_hv = state_hv.bind(action_hv)
            state_action_pairs.append(state_action_hv)
        
        # Create behavior hypervector from sequence
        behavior_hv = HyperVector.bundle_vectors(state_action_pairs)
        
        # Include context if provided
        if context is not None:
            behavior_hv = behavior_hv.bind(context)
        
        # Store in behavior memory
        confidence = 1.0 / (1.0 + len(states))  # Longer demos get lower initial confidence
        confidence = max(0.3, confidence)  # Minimum confidence
        
        self.behavior_memory.store(behavior_name, behavior_hv, confidence)
        
        # Store in episodic memory
        states_as_hvs = [self.sensor_encoder.encode_multimodal_state(s) if isinstance(s, dict) 
                        else s for s in states]
        self.episode_memory.get_episodic_memory().store_episode(
            states_as_hvs, [behavior_name] * len(states), confidence
        )
        
        # Update statistics
        self.learning_stats['behaviors_learned'] += 1
        self.learning_stats['demonstrations_processed'] += 1
        self.learning_stats['last_learning_time'] = time.time()
        
        return confidence
    
    def adapt_behavior(self, base_behavior: str, adaptation_examples: List[Dict[str, Any]],
                      new_behavior_name: Optional[str] = None) -> float:
        """Adapt existing behavior with few-shot examples.
        
        Args:
            base_behavior: Name of existing behavior to adapt
            adaptation_examples: List of adaptation demonstrations
            new_behavior_name: Optional name for adapted behavior (defaults to base + '_adapted')
            
        Returns:
            Confidence score of adapted behavior
        """
        if not self.behavior_memory.contains(base_behavior):
            raise ValueError(f"Base behavior '{base_behavior}' not found")
        
        if not adaptation_examples:
            raise ValueError("At least one adaptation example required")
        
        # Get base behavior
        base_behavior_hv = self.behavior_memory.retrieve(base_behavior)
        base_confidence = self.behavior_memory.get_confidence(base_behavior)
        
        # Learn from adaptation examples
        adaptation_hvs = []
        for example in adaptation_examples:
            # Create behavior from example (similar to learn_from_demonstration)
            states = example['states']
            actions = example['actions']
            
            state_action_pairs = []
            for state, action in zip(states, actions):
                if isinstance(state, dict):
                    state_hv = self.sensor_encoder.encode_multimodal_state(state)
                else:
                    state_hv = self.sensor_encoder.encode_image_features(np.array(state).flatten())
                
                if isinstance(action, (list, np.ndarray)):
                    action_hv = self.sensor_encoder.encode_image_features(np.array(action).flatten())
                else:
                    action_hv = self.sensor_encoder.basis_vectors.encode_category(str(action))
                
                state_action_pairs.append(state_hv.bind(action_hv))
            
            adaptation_hvs.append(HyperVector.bundle_vectors(state_action_pairs))
        
        # Combine adaptations
        adaptation_behavior_hv = HyperVector.bundle_vectors(adaptation_hvs)
        
        # Weighted combination of base behavior and adaptations
        adaptation_weight = min(0.7, len(adaptation_examples) * 0.2)  # Cap at 70%
        base_weight = 1.0 - adaptation_weight
        
        adapted_behavior_hv = self.weighted_bundle([
            (base_behavior_hv, base_weight),
            (adaptation_behavior_hv, adaptation_weight)
        ])
        
        # Calculate adapted confidence
        adapted_confidence = base_confidence * 0.8 + len(adaptation_examples) * 0.05
        adapted_confidence = min(1.0, adapted_confidence)
        
        # Store adapted behavior
        if new_behavior_name is None:
            new_behavior_name = f"{base_behavior}_adapted"
        
        self.behavior_memory.store(new_behavior_name, adapted_behavior_hv, adapted_confidence)
        
        # Update statistics
        self.learning_stats['adaptations_made'] += 1
        
        return adapted_confidence
    
    def query_behavior(self, current_state: Dict[str, Any], 
                      context: Optional[HyperVector] = None,
                      top_k: int = 3) -> List[Dict[str, Any]]:
        """Query for relevant behaviors given current state.
        
        Args:
            current_state: Current sensor/state information
            context: Optional context hypervector
            top_k: Number of top behaviors to return
            
        Returns:
            List of behavior matches with similarities and actions
        """
        # Encode current state
        if isinstance(current_state, dict):
            state_hv = self.sensor_encoder.encode_multimodal_state(current_state)
        elif isinstance(current_state, HyperVector):
            state_hv = current_state
        else:
            state_hv = self.sensor_encoder.encode_image_features(np.array(current_state).flatten())
        
        # Include context
        if context is not None:
            query_hv = state_hv.bind(context)
        else:
            query_hv = state_hv
        
        # Query behavior memory
        results = self.behavior_memory.query(query_hv, max_results=top_k)
        
        # Filter by similarity threshold
        filtered_results = [r for r in results if r['similarity'] >= self.similarity_threshold]
        
        # Add execution recommendations
        for result in filtered_results:
            result['recommended_action'] = self._extract_action_from_behavior(
                result['vector'], state_hv)
            result['execution_confidence'] = result['similarity'] * result['confidence']
        
        return filtered_results
    
    def execute_behavior(self, behavior_name: str, current_state: Dict[str, Any],
                        context: Optional[HyperVector] = None) -> Dict[str, Any]:
        """Execute learned behavior in current context.
        
        Args:
            behavior_name: Name of behavior to execute
            current_state: Current sensor/state information  
            context: Optional execution context
            
        Returns:
            Dictionary with recommended action and execution metadata
        """
        if not self.behavior_memory.contains(behavior_name):
            raise ValueError(f"Behavior '{behavior_name}' not found")
        
        # Get behavior
        behavior_hv = self.behavior_memory.retrieve(behavior_name)
        behavior_confidence = self.behavior_memory.get_confidence(behavior_name)
        
        # Encode current state
        if isinstance(current_state, dict):
            state_hv = self.sensor_encoder.encode_multimodal_state(current_state)
        else:
            state_hv = self.sensor_encoder.encode_image_features(np.array(current_state).flatten())
        
        # Extract action from behavior
        recommended_action = self._extract_action_from_behavior(behavior_hv, state_hv)
        
        # Calculate execution confidence
        state_similarity = behavior_hv.similarity(state_hv)
        execution_confidence = behavior_confidence * max(0.1, state_similarity)
        
        # Record execution
        execution_record = {
            'behavior_name': behavior_name,
            'timestamp': time.time(),
            'state': current_state,
            'recommended_action': recommended_action,
            'confidence': execution_confidence,
            'state_similarity': state_similarity
        }
        
        self.execution_history.append(execution_record)
        
        return {
            'action': recommended_action,
            'confidence': execution_confidence,
            'behavior': behavior_name,
            'metadata': {
                'state_similarity': state_similarity,
                'behavior_confidence': behavior_confidence,
                'execution_time': time.time()
            }
        }
    
    def continual_learning_update(self, feedback: Dict[str, Any]) -> None:
        """Update behaviors based on execution feedback.
        
        Args:
            feedback: Dictionary containing execution results and performance metrics
                     e.g., {'behavior': name, 'success': True/False, 'reward': float, ...}
        """
        behavior_name = feedback.get('behavior')
        success = feedback.get('success', False)
        reward = feedback.get('reward', 0.0)
        
        if not behavior_name or not self.behavior_memory.contains(behavior_name):
            return
        
        # Get current behavior
        current_confidence = self.behavior_memory.get_confidence(behavior_name)
        
        # Update confidence based on feedback
        if success:
            # Positive feedback increases confidence
            confidence_delta = self.adaptation_rate * (1.0 - current_confidence)
            new_confidence = min(1.0, current_confidence + confidence_delta)
        else:
            # Negative feedback decreases confidence
            confidence_delta = self.adaptation_rate * current_confidence * 0.5
            new_confidence = max(0.1, current_confidence - confidence_delta)
        
        # Update behavior confidence
        self.behavior_memory.update_confidence(behavior_name, new_confidence)
        
        # If very poor performance, consider removing behavior
        if new_confidence < 0.2 and not success:
            print(f"Warning: Behavior '{behavior_name}' has low confidence ({new_confidence:.2f})")
    
    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """Analyze learning patterns and behavior effectiveness.
        
        Returns:
            Dictionary with learning analytics and insights
        """
        # Behavior statistics
        behavior_stats = {}
        for behavior_name in self.behavior_memory.get_labels():
            confidence = self.behavior_memory.get_confidence(behavior_name)
            
            # Count executions
            executions = [e for e in self.execution_history if e['behavior_name'] == behavior_name]
            
            behavior_stats[behavior_name] = {
                'confidence': confidence,
                'execution_count': len(executions),
                'avg_execution_confidence': np.mean([e['confidence'] for e in executions]) if executions else 0.0,
                'last_execution': max([e['timestamp'] for e in executions]) if executions else None
            }
        
        # Overall learning metrics
        total_behaviors = len(behavior_stats)
        avg_confidence = np.mean([stats['confidence'] for stats in behavior_stats.values()]) if behavior_stats else 0.0
        
        return {
            'total_behaviors_learned': total_behaviors,
            'average_behavior_confidence': avg_confidence,
            'total_executions': len(self.execution_history),
            'behavior_statistics': behavior_stats,
            'learning_stats': self.learning_stats.copy(),
            'memory_usage': {
                'behavior_memory_size': self.behavior_memory.size(),
                'episode_memory_size': self.episode_memory.get_episodic_memory().size(),
                'execution_history_size': len(self.execution_history)
            }
        }
    
    def save_learned_behaviors(self, filepath: str) -> None:
        """Save learned behaviors to file.
        
        Args:
            filepath: Path to save behaviors
        """
        self.behavior_memory.save_to_file(filepath)
    
    def load_learned_behaviors(self, filepath: str) -> None:
        """Load learned behaviors from file.
        
        Args:
            filepath: Path to load behaviors from
        """
        self.behavior_memory.load_from_file(filepath)
        
        # Update statistics
        self.learning_stats['behaviors_learned'] = self.behavior_memory.size()
    
    def weighted_bundle(self, weighted_vectors: List[Tuple[HyperVector, float]]) -> HyperVector:
        """Create weighted bundle of hypervectors."""
        if not weighted_vectors:
            raise ValueError("Cannot bundle empty vector list")
        
        dimension = weighted_vectors[0][0].dimension
        weighted_sum = np.zeros(dimension, dtype=np.float64)
        
        for vector, weight in weighted_vectors:
            if vector.dimension != dimension:
                raise ValueError("All vectors must have same dimension")
            weighted_sum += vector.data.astype(np.float64) * weight
        
        result_data = np.where(weighted_sum > 0, 1, -1).astype(np.int8)
        return HyperVector(dimension, result_data)
    
    def _extract_action_from_behavior(self, behavior_hv: HyperVector, 
                                    state_hv: HyperVector) -> Dict[str, Any]:
        """Extract recommended action from behavior given current state."""
        # This is a simplified action extraction
        # In practice, this would involve more sophisticated decoding
        
        # For now, return a generic action recommendation
        similarity = behavior_hv.similarity(state_hv)
        
        # Simple heuristic: higher similarity suggests more confident action
        if similarity > 0.8:
            action_type = "precise_action"
            confidence = similarity
        elif similarity > 0.5:
            action_type = "moderate_action"
            confidence = similarity * 0.8
        else:
            action_type = "exploratory_action"
            confidence = similarity * 0.5
        
        return {
            'type': action_type,
            'confidence': confidence,
            'parameters': {
                'similarity': similarity,
                'behavior_vector_sample': behavior_hv.data[:10].tolist()  # First 10 dimensions
            }
        }