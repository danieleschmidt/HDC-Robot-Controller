"""
Multidimensional Intelligence System
===================================

Implements intelligence that operates across multiple dimensions of reality,
including spatial, temporal, conceptual, and abstract dimensions.
Enables reasoning and learning in high-dimensional spaces beyond physical reality.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import json
from pathlib import Path

from ..core.hypervector import HyperVector
from ..core.memory import AssociativeMemory


class DimensionType(Enum):
    """Types of intelligence dimensions."""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CONCEPTUAL = "conceptual"
    EMOTIONAL = "emotional"
    CAUSAL = "causal"
    ABSTRACT = "abstract"
    QUANTUM = "quantum"
    LINGUISTIC = "linguistic"
    SOCIAL = "social"
    CREATIVE = "creative"


@dataclass
class DimensionalAxis:
    """Represents an axis within a dimensional space."""
    name: str
    dimension_type: DimensionType
    min_value: float = -1.0
    max_value: float = 1.0
    resolution: int = 1000
    semantic_labels: Dict[float, str] = field(default_factory=dict)
    
    def quantize_value(self, value: float) -> int:
        """Quantize continuous value to discrete axis position."""
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        normalized = np.clip(normalized, 0.0, 1.0)
        return int(normalized * (self.resolution - 1))
    
    def get_semantic_label(self, value: float) -> Optional[str]:
        """Get semantic label for a value if available."""
        quantized = self.quantize_value(value)
        return self.semantic_labels.get(quantized)


@dataclass
class DimensionalContext:
    """Context within multidimensional space."""
    dimensions: Dict[str, DimensionalAxis]
    current_position: Dict[str, float]
    uncertainty: Dict[str, float] = field(default_factory=dict)
    context_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def get_hypervector_encoding(self, hdc_dimension: int) -> HyperVector:
        """Encode dimensional context as hypervector."""
        context_hv = HyperVector.zero(hdc_dimension)
        
        for dim_name, position in self.current_position.items():
            if dim_name in self.dimensions:
                axis = self.dimensions[dim_name]
                quantized = axis.quantize_value(position)
                
                # Create position hypervector
                pos_hv = HyperVector.random(hdc_dimension, seed=hash(dim_name + str(quantized)))
                context_hv = context_hv.bundle(pos_hv)
        
        return context_hv
    
    def distance_to(self, other: 'DimensionalContext') -> float:
        """Calculate distance to another dimensional context."""
        total_distance = 0.0
        shared_dimensions = 0
        
        for dim_name in self.current_position:
            if dim_name in other.current_position:
                self_pos = self.current_position[dim_name]
                other_pos = other.current_position[dim_name]
                
                # Normalize by dimension range
                if dim_name in self.dimensions:
                    axis = self.dimensions[dim_name]
                    dim_range = axis.max_value - axis.min_value
                    normalized_distance = abs(self_pos - other_pos) / dim_range
                    total_distance += normalized_distance ** 2
                    shared_dimensions += 1
        
        if shared_dimensions == 0:
            return float('inf')
        
        return np.sqrt(total_distance / shared_dimensions)


class DimensionalSpace:
    """Represents a multidimensional intelligence space."""
    
    def __init__(self, name: str, dimensions: List[DimensionalAxis]):
        self.name = name
        self.dimensions = {axis.name: axis for axis in dimensions}
        self.context_history = []
        self.dimension_interactions = {}
        
    def create_context(self, positions: Dict[str, float], context_id: str = "") -> DimensionalContext:
        """Create dimensional context at specified positions."""
        # Validate positions
        validated_positions = {}
        uncertainties = {}
        
        for dim_name, position in positions.items():
            if dim_name in self.dimensions:
                axis = self.dimensions[dim_name]
                # Clamp to valid range
                clamped_pos = np.clip(position, axis.min_value, axis.max_value)
                validated_positions[dim_name] = clamped_pos
                
                # Calculate uncertainty based on clamping
                uncertainties[dim_name] = abs(position - clamped_pos)
            else:
                logging.warning(f"Unknown dimension: {dim_name}")
        
        context = DimensionalContext(
            dimensions=self.dimensions,
            current_position=validated_positions,
            uncertainty=uncertainties,
            context_id=context_id or f"ctx_{len(self.context_history)}"
        )
        
        self.context_history.append(context)
        return context
    
    def navigate_to(self, target_context: DimensionalContext, 
                   current_context: DimensionalContext,
                   step_size: float = 0.1) -> DimensionalContext:
        """Navigate from current context toward target context."""
        
        new_positions = {}
        
        for dim_name in current_context.current_position:
            current_pos = current_context.current_position[dim_name]
            
            if dim_name in target_context.current_position:
                target_pos = target_context.current_position[dim_name]
                direction = target_pos - current_pos
                
                # Take step toward target
                new_pos = current_pos + direction * step_size
                new_positions[dim_name] = new_pos
            else:
                # Maintain current position if no target
                new_positions[dim_name] = current_pos
        
        return self.create_context(new_positions, f"nav_{time.time()}")
    
    def find_similar_contexts(self, query_context: DimensionalContext, 
                             max_distance: float = 0.3,
                             max_results: int = 10) -> List[Tuple[DimensionalContext, float]]:
        """Find similar contexts within the dimensional space."""
        
        similar_contexts = []
        
        for historical_context in self.context_history:
            distance = query_context.distance_to(historical_context)
            
            if distance <= max_distance:
                similar_contexts.append((historical_context, distance))
        
        # Sort by distance and return top results
        similar_contexts.sort(key=lambda x: x[1])
        return similar_contexts[:max_results]
    
    def analyze_dimension_correlations(self) -> Dict[Tuple[str, str], float]:
        """Analyze correlations between different dimensions."""
        
        correlations = {}
        
        if len(self.context_history) < 10:
            return correlations
        
        # Get dimension names
        dim_names = list(self.dimensions.keys())
        
        for i, dim1 in enumerate(dim_names):
            for j, dim2 in enumerate(dim_names):
                if i < j:  # Avoid duplicate pairs
                    # Extract time series for each dimension
                    dim1_values = []
                    dim2_values = []
                    
                    for context in self.context_history:
                        if dim1 in context.current_position and dim2 in context.current_position:
                            dim1_values.append(context.current_position[dim1])
                            dim2_values.append(context.current_position[dim2])
                    
                    if len(dim1_values) > 5:
                        correlation = np.corrcoef(dim1_values, dim2_values)[0, 1]
                        if not np.isnan(correlation):
                            correlations[(dim1, dim2)] = correlation
        
        return correlations


class ConceptualMapper:
    """Maps concepts to multidimensional space."""
    
    def __init__(self, hdc_dimension: int = 10000):
        self.hdc_dimension = hdc_dimension
        self.concept_embeddings = {}
        self.concept_relationships = {}
        
    def embed_concept(self, concept: str, 
                     dimensional_context: DimensionalContext) -> HyperVector:
        """Embed concept in multidimensional space."""
        
        # Create base concept hypervector
        concept_hv = HyperVector.random(self.hdc_dimension, seed=hash(concept))
        
        # Bind with dimensional context
        context_hv = dimensional_context.get_hypervector_encoding(self.hdc_dimension)
        embedded_concept = concept_hv.bind(context_hv)
        
        self.concept_embeddings[concept] = embedded_concept
        return embedded_concept
    
    def find_conceptual_neighbors(self, concept: str, 
                                similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find conceptually similar concepts."""
        
        if concept not in self.concept_embeddings:
            return []
        
        query_hv = self.concept_embeddings[concept]
        neighbors = []
        
        for other_concept, other_hv in self.concept_embeddings.items():
            if other_concept != concept:
                similarity = query_hv.similarity(other_hv)
                if similarity >= similarity_threshold:
                    neighbors.append((other_concept, similarity))
        
        # Sort by similarity
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors
    
    def create_conceptual_analogy(self, concept_a: str, concept_b: str, 
                                concept_c: str) -> Optional[str]:
        """Create conceptual analogy: A is to B as C is to ?"""
        
        required_concepts = [concept_a, concept_b, concept_c]
        if not all(c in self.concept_embeddings for c in required_concepts):
            return None
        
        # Calculate analogy vector: B - A + C = D
        hv_a = self.concept_embeddings[concept_a]
        hv_b = self.concept_embeddings[concept_b]
        hv_c = self.concept_embeddings[concept_c]
        
        analogy_hv = hv_b.unbind(hv_a).bind(hv_c)
        
        # Find most similar concept to analogy result
        best_match = None
        best_similarity = 0.0
        
        for concept, concept_hv in self.concept_embeddings.items():
            if concept not in required_concepts:
                similarity = analogy_hv.similarity(concept_hv)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = concept
        
        return best_match if best_similarity > 0.6 else None


class TemporalIntelligence:
    """Handles temporal reasoning and prediction."""
    
    def __init__(self, hdc_dimension: int = 10000):
        self.hdc_dimension = hdc_dimension
        self.temporal_sequences = {}
        self.temporal_patterns = {}
        
    def encode_temporal_sequence(self, sequence_id: str, 
                                events: List[DimensionalContext],
                                time_deltas: List[float]) -> HyperVector:
        """Encode temporal sequence of dimensional contexts."""
        
        if len(events) != len(time_deltas) + 1:
            raise ValueError("time_deltas should be one less than events")
        
        sequence_hv = HyperVector.zero(self.hdc_dimension)
        
        for i, (event, time_delta) in enumerate(zip(events[:-1], time_deltas)):
            # Encode event
            event_hv = event.get_hypervector_encoding(self.hdc_dimension)
            
            # Encode temporal position
            time_hv = self._encode_time(time_delta, i)
            
            # Bind event with temporal position
            temporal_event = event_hv.bind(time_hv)
            sequence_hv = sequence_hv.bundle(temporal_event)
        
        # Add final event
        final_event_hv = events[-1].get_hypervector_encoding(self.hdc_dimension)
        final_time_hv = self._encode_time(0.0, len(events) - 1)
        final_temporal_event = final_event_hv.bind(final_time_hv)
        sequence_hv = sequence_hv.bundle(final_temporal_event)
        
        self.temporal_sequences[sequence_id] = sequence_hv
        return sequence_hv
    
    def _encode_time(self, time_delta: float, position: int) -> HyperVector:
        """Encode temporal information as hypervector."""
        # Create time hypervector based on delta and position
        time_seed = hash(f"time_{time_delta:.3f}_{position}")
        return HyperVector.random(self.hdc_dimension, seed=time_seed)
    
    def predict_next_context(self, sequence_prefix: List[DimensionalContext],
                           prediction_horizon: float = 1.0) -> Optional[DimensionalContext]:
        """Predict next context in temporal sequence."""
        
        if len(sequence_prefix) < 2:
            return None
        
        # Find similar temporal patterns
        similar_sequences = self._find_similar_temporal_patterns(sequence_prefix)
        
        if not similar_sequences:
            return None
        
        # Weighted prediction based on similarity
        predicted_positions = {}
        total_weight = 0.0
        
        for sequence_id, similarity in similar_sequences[:5]:  # Top 5 matches
            if sequence_id in self.temporal_patterns:
                pattern = self.temporal_patterns[sequence_id]
                if len(pattern) > len(sequence_prefix):
                    next_context = pattern[len(sequence_prefix)]
                    weight = similarity
                    
                    for dim_name, position in next_context.current_position.items():
                        if dim_name not in predicted_positions:
                            predicted_positions[dim_name] = 0.0
                        predicted_positions[dim_name] += position * weight
                    
                    total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Normalize predictions
        for dim_name in predicted_positions:
            predicted_positions[dim_name] /= total_weight
        
        # Create predicted context
        # This is a simplified implementation - would need the dimensional space
        return DimensionalContext(
            dimensions={},  # Would need proper dimensions
            current_position=predicted_positions,
            context_id=f"predicted_{time.time()}"
        )
    
    def _find_similar_temporal_patterns(self, prefix: List[DimensionalContext]) -> List[Tuple[str, float]]:
        """Find temporal patterns similar to given prefix."""
        
        if len(prefix) < 2:
            return []
        
        # Encode prefix
        time_deltas = []
        for i in range(len(prefix) - 1):
            delta = prefix[i + 1].timestamp - prefix[i].timestamp
            time_deltas.append(delta)
        
        prefix_hv = self.encode_temporal_sequence("temp_prefix", prefix, time_deltas)
        
        # Find similar sequences
        similar_sequences = []
        for sequence_id, sequence_hv in self.temporal_sequences.items():
            similarity = prefix_hv.similarity(sequence_hv)
            if similarity > 0.5:
                similar_sequences.append((sequence_id, similarity))
        
        similar_sequences.sort(key=lambda x: x[1], reverse=True)
        return similar_sequences


class AbstractReasoner:
    """Handles abstract reasoning and pattern recognition."""
    
    def __init__(self, hdc_dimension: int = 10000):
        self.hdc_dimension = hdc_dimension
        self.abstract_patterns = {}
        self.reasoning_rules = {}
        
    def identify_abstract_pattern(self, contexts: List[DimensionalContext],
                                pattern_name: str) -> HyperVector:
        """Identify abstract patterns across dimensional contexts."""
        
        if len(contexts) < 3:
            return HyperVector.zero(self.hdc_dimension)
        
        # Look for invariant properties across contexts
        invariant_dimensions = set()
        
        # Find dimensions that vary consistently
        varying_dimensions = {}
        
        for dim_name in contexts[0].current_position:
            values = [ctx.current_position.get(dim_name, 0.0) for ctx in contexts]
            
            # Check for invariance
            if max(values) - min(values) < 0.1:
                invariant_dimensions.add(dim_name)
            else:
                # Check for linear relationship
                x = np.arange(len(values))
                correlation = np.corrcoef(x, values)[0, 1]
                if abs(correlation) > 0.7:
                    varying_dimensions[dim_name] = correlation
        
        # Create pattern hypervector
        pattern_hv = HyperVector.zero(self.hdc_dimension)
        
        # Encode invariant dimensions
        for dim_name in invariant_dimensions:
            dim_hv = HyperVector.random(self.hdc_dimension, seed=hash(f"invariant_{dim_name}"))
            pattern_hv = pattern_hv.bundle(dim_hv)
        
        # Encode varying dimensions with their correlation
        for dim_name, correlation in varying_dimensions.items():
            dim_hv = HyperVector.random(self.hdc_dimension, seed=hash(f"varying_{dim_name}"))
            corr_hv = HyperVector.random(self.hdc_dimension, seed=hash(f"corr_{correlation:.2f}"))
            varying_hv = dim_hv.bind(corr_hv)
            pattern_hv = pattern_hv.bundle(varying_hv)
        
        self.abstract_patterns[pattern_name] = pattern_hv
        return pattern_hv
    
    def apply_abstract_reasoning(self, rule_name: str, 
                               input_contexts: List[DimensionalContext]) -> List[DimensionalContext]:
        """Apply abstract reasoning rule to contexts."""
        
        if rule_name not in self.reasoning_rules:
            return input_contexts
        
        rule = self.reasoning_rules[rule_name]
        
        # Apply rule (simplified implementation)
        output_contexts = []
        
        for context in input_contexts:
            # Transform context according to rule
            transformed_positions = {}
            
            for dim_name, position in context.current_position.items():
                if dim_name in rule.get('transformations', {}):
                    transform = rule['transformations'][dim_name]
                    if transform['type'] == 'linear':
                        transformed_pos = position * transform['scale'] + transform['offset']
                    elif transform['type'] == 'nonlinear':
                        transformed_pos = transform['function'](position)
                    else:
                        transformed_pos = position
                else:
                    transformed_pos = position
                
                transformed_positions[dim_name] = transformed_pos
            
            transformed_context = DimensionalContext(
                dimensions=context.dimensions,
                current_position=transformed_positions,
                context_id=f"transformed_{context.context_id}"
            )
            
            output_contexts.append(transformed_context)
        
        return output_contexts
    
    def create_reasoning_rule(self, rule_name: str, 
                            transformations: Dict[str, Dict[str, Any]]):
        """Create new abstract reasoning rule."""
        
        self.reasoning_rules[rule_name] = {
            'transformations': transformations,
            'created_at': time.time()
        }


class MultidimensionalIntelligence:
    """Main multidimensional intelligence system."""
    
    def __init__(self, hdc_dimension: int = 10000, logger: logging.Logger = None):
        self.hdc_dimension = hdc_dimension
        self.logger = logger or logging.getLogger(__name__)
        
        # Core components
        self.dimensional_spaces = {}
        self.conceptual_mapper = ConceptualMapper(hdc_dimension)
        self.temporal_intelligence = TemporalIntelligence(hdc_dimension)
        self.abstract_reasoner = AbstractReasoner(hdc_dimension)
        
        # Intelligence metrics
        self.reasoning_history = []
        self.intelligence_metrics = {
            'concepts_learned': 0,
            'patterns_discovered': 0,
            'predictions_made': 0,
            'reasoning_operations': 0
        }
        
        # Threading for continuous intelligence
        self.intelligence_thread = None
        self.is_intelligence_active = False
        
        self.logger.info(f"🌌 Multidimensional Intelligence initialized (dimension: {hdc_dimension})")
    
    def create_dimensional_space(self, space_name: str, 
                                dimensions: List[DimensionalAxis]) -> DimensionalSpace:
        """Create new dimensional space."""
        
        space = DimensionalSpace(space_name, dimensions)
        self.dimensional_spaces[space_name] = space
        
        self.logger.info(f"📐 Created dimensional space '{space_name}' with {len(dimensions)} dimensions")
        return space
    
    def activate_intelligence(self):
        """Activate continuous multidimensional intelligence processes."""
        
        if self.is_intelligence_active:
            return
        
        self.is_intelligence_active = True
        
        # Start intelligence processing thread
        self.intelligence_thread = threading.Thread(
            target=self._intelligence_loop,
            daemon=True
        )
        self.intelligence_thread.start()
        
        self.logger.info("🧠 Multidimensional intelligence activated")
    
    def deactivate_intelligence(self):
        """Deactivate intelligence processes."""
        
        if not self.is_intelligence_active:
            return
        
        self.is_intelligence_active = False
        
        if self.intelligence_thread:
            self.intelligence_thread.join(timeout=2.0)
        
        self.logger.info("💤 Multidimensional intelligence deactivated")
    
    def _intelligence_loop(self):
        """Main intelligence processing loop."""
        
        while self.is_intelligence_active:
            try:
                # Analyze dimensional spaces for patterns
                self._analyze_dimensional_patterns()
                
                # Update conceptual relationships
                self._update_conceptual_mappings()
                
                # Perform temporal predictions
                self._perform_temporal_predictions()
                
                # Abstract reasoning
                self._perform_abstract_reasoning()
                
                # Sleep
                time.sleep(5.0)  # Process every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in intelligence loop: {e}")
                time.sleep(5.0)
    
    def _analyze_dimensional_patterns(self):
        """Analyze patterns in dimensional spaces."""
        
        for space_name, space in self.dimensional_spaces.items():
            if len(space.context_history) > 10:
                # Analyze dimension correlations
                correlations = space.analyze_dimension_correlations()
                
                # Look for strong correlations (abstract patterns)
                for (dim1, dim2), correlation in correlations.items():
                    if abs(correlation) > 0.8:
                        pattern_name = f"{space_name}_{dim1}_{dim2}_correlation"
                        
                        # Get contexts that show this correlation
                        relevant_contexts = []
                        for context in space.context_history[-20:]:  # Recent contexts
                            if dim1 in context.current_position and dim2 in context.current_position:
                                relevant_contexts.append(context)
                        
                        if len(relevant_contexts) >= 3:
                            self.abstract_reasoner.identify_abstract_pattern(
                                relevant_contexts, pattern_name
                            )
                            self.intelligence_metrics['patterns_discovered'] += 1
    
    def _update_conceptual_mappings(self):
        """Update conceptual mappings based on recent activity."""
        
        # This would integrate with external concept sources in a real system
        # For now, simulate concept learning
        
        if len(self.conceptual_mapper.concept_embeddings) < 100:
            # Add some example concepts
            example_concepts = [
                "movement", "stability", "precision", "speed", "efficiency",
                "learning", "adaptation", "memory", "attention", "goal"
            ]
            
            for concept in example_concepts:
                if concept not in self.conceptual_mapper.concept_embeddings:
                    # Create synthetic dimensional context for concept
                    if self.dimensional_spaces:
                        space = list(self.dimensional_spaces.values())[0]
                        positions = {
                            dim_name: np.random.uniform(axis.min_value, axis.max_value)
                            for dim_name, axis in space.dimensions.items()
                        }
                        context = space.create_context(positions, f"concept_{concept}")
                        
                        self.conceptual_mapper.embed_concept(concept, context)
                        self.intelligence_metrics['concepts_learned'] += 1
    
    def _perform_temporal_predictions(self):
        """Perform temporal predictions on dimensional contexts."""
        
        for space_name, space in self.dimensional_spaces.items():
            if len(space.context_history) > 5:
                # Take recent context sequence
                recent_contexts = space.context_history[-5:]
                
                # Try to predict next context
                prediction = self.temporal_intelligence.predict_next_context(recent_contexts)
                
                if prediction:
                    self.intelligence_metrics['predictions_made'] += 1
                    
                    # Store temporal pattern
                    pattern_id = f"{space_name}_pattern_{len(self.temporal_intelligence.temporal_patterns)}"
                    self.temporal_intelligence.temporal_patterns[pattern_id] = recent_contexts
    
    def _perform_abstract_reasoning(self):
        """Perform abstract reasoning operations."""
        
        # Create some basic reasoning rules if none exist
        if not self.abstract_reasoner.reasoning_rules:
            self.abstract_reasoner.create_reasoning_rule(
                "scale_transform",
                {
                    "spatial_x": {"type": "linear", "scale": 2.0, "offset": 0.0},
                    "spatial_y": {"type": "linear", "scale": 2.0, "offset": 0.0}
                }
            )
            
            self.abstract_reasoner.create_reasoning_rule(
                "inverse_transform", 
                {
                    "efficiency": {"type": "linear", "scale": -1.0, "offset": 1.0}
                }
            )
        
        self.intelligence_metrics['reasoning_operations'] += 1
    
    def reason_about_context(self, context: DimensionalContext,
                           question: str) -> Dict[str, Any]:
        """Reason about a dimensional context to answer a question."""
        
        reasoning_result = {
            'question': question,
            'context_id': context.context_id,
            'reasoning_type': 'multidimensional',
            'timestamp': time.time()
        }
        
        if "similar" in question.lower():
            # Find similar contexts
            for space_name, space in self.dimensional_spaces.items():
                if context.dimensions == space.dimensions:
                    similar = space.find_similar_contexts(context)
                    reasoning_result['similar_contexts'] = [
                        {'context_id': ctx.context_id, 'distance': dist}
                        for ctx, dist in similar[:5]
                    ]
                    break
        
        elif "predict" in question.lower():
            # Temporal prediction
            prediction = self.temporal_intelligence.predict_next_context([context])
            if prediction:
                reasoning_result['prediction'] = {
                    'predicted_context_id': prediction.context_id,
                    'predicted_positions': prediction.current_position
                }
        
        elif "pattern" in question.lower():
            # Pattern analysis
            patterns_found = []
            for pattern_name, pattern_hv in self.abstract_reasoner.abstract_patterns.items():
                # Check if context matches pattern
                context_hv = context.get_hypervector_encoding(self.hdc_dimension)
                similarity = context_hv.similarity(pattern_hv)
                if similarity > 0.6:
                    patterns_found.append({
                        'pattern_name': pattern_name,
                        'similarity': similarity
                    })
            
            reasoning_result['matching_patterns'] = patterns_found
        
        elif "concept" in question.lower():
            # Conceptual reasoning
            related_concepts = []
            context_hv = context.get_hypervector_encoding(self.hdc_dimension)
            
            for concept, concept_hv in self.conceptual_mapper.concept_embeddings.items():
                similarity = context_hv.similarity(concept_hv)
                if similarity > 0.5:
                    related_concepts.append({
                        'concept': concept,
                        'similarity': similarity
                    })
            
            reasoning_result['related_concepts'] = sorted(
                related_concepts, key=lambda x: x['similarity'], reverse=True
            )[:10]
        
        self.reasoning_history.append(reasoning_result)
        return reasoning_result
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive intelligence metrics."""
        
        metrics = self.intelligence_metrics.copy()
        
        metrics.update({
            'dimensional_spaces': len(self.dimensional_spaces),
            'total_contexts': sum(len(space.context_history) for space in self.dimensional_spaces.values()),
            'reasoning_history_length': len(self.reasoning_history),
            'abstract_patterns': len(self.abstract_reasoner.abstract_patterns),
            'temporal_patterns': len(self.temporal_intelligence.temporal_patterns),
            'conceptual_embeddings': len(self.conceptual_mapper.concept_embeddings),
            'reasoning_rules': len(self.abstract_reasoner.reasoning_rules)
        })
        
        return metrics
    
    def save_intelligence_state(self, filepath: str):
        """Save complete intelligence state."""
        
        state_data = {
            'dimensional_spaces': {
                name: {
                    'dimensions': {
                        dim_name: {
                            'name': axis.name,
                            'dimension_type': axis.dimension_type.value,
                            'min_value': axis.min_value,
                            'max_value': axis.max_value,
                            'resolution': axis.resolution
                        }
                        for dim_name, axis in space.dimensions.items()
                    },
                    'context_count': len(space.context_history)
                }
                for name, space in self.dimensional_spaces.items()
            },
            'intelligence_metrics': self.get_intelligence_metrics(),
            'reasoning_history': self.reasoning_history[-50:],  # Last 50 reasoning operations
            'conceptual_embeddings_count': len(self.conceptual_mapper.concept_embeddings),
            'abstract_patterns_count': len(self.abstract_reasoner.abstract_patterns)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Intelligence state saved to {filepath}")


# Example usage and demonstration
if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create multidimensional intelligence
    intelligence = MultidimensionalIntelligence(hdc_dimension=10000)
    
    print("🌌 Multidimensional Intelligence Demo")
    
    # Create dimensional spaces
    spatial_dims = [
        DimensionalAxis("x", DimensionType.SPATIAL, -10.0, 10.0),
        DimensionalAxis("y", DimensionType.SPATIAL, -10.0, 10.0),
        DimensionalAxis("z", DimensionType.SPATIAL, -5.0, 5.0)
    ]
    
    conceptual_dims = [
        DimensionalAxis("efficiency", DimensionType.CONCEPTUAL, 0.0, 1.0),
        DimensionalAxis("complexity", DimensionType.CONCEPTUAL, 0.0, 1.0),
        DimensionalAxis("novelty", DimensionType.CONCEPTUAL, 0.0, 1.0)
    ]
    
    temporal_dims = [
        DimensionalAxis("duration", DimensionType.TEMPORAL, 0.0, 100.0),
        DimensionalAxis("frequency", DimensionType.TEMPORAL, 0.1, 10.0)
    ]
    
    # Create spaces
    spatial_space = intelligence.create_dimensional_space("spatial", spatial_dims)
    conceptual_space = intelligence.create_dimensional_space("conceptual", conceptual_dims)
    temporal_space = intelligence.create_dimensional_space("temporal", temporal_dims)
    
    # Activate intelligence
    intelligence.activate_intelligence()
    
    # Generate some contexts
    print("\n📐 Creating dimensional contexts...")
    for i in range(20):
        # Spatial context
        spatial_positions = {
            "x": np.random.uniform(-5, 5),
            "y": np.random.uniform(-5, 5), 
            "z": np.random.uniform(-2, 2)
        }
        spatial_context = spatial_space.create_context(spatial_positions, f"spatial_{i}")
        
        # Conceptual context
        conceptual_positions = {
            "efficiency": np.random.uniform(0.3, 0.9),
            "complexity": np.random.uniform(0.2, 0.8),
            "novelty": np.random.uniform(0.1, 0.7)
        }
        conceptual_context = conceptual_space.create_context(conceptual_positions, f"conceptual_{i}")
        
        time.sleep(0.1)  # Small delay to create temporal sequence
    
    # Wait for intelligence processing
    time.sleep(2)
    
    # Demonstrate reasoning
    print("\n🧠 Demonstrating multidimensional reasoning...")
    
    # Create test context
    test_context = spatial_space.create_context({
        "x": 1.0, "y": 2.0, "z": 0.5
    }, "test_context")
    
    # Ask different types of questions
    questions = [
        "What are similar contexts to this one?",
        "What patterns match this context?", 
        "What concepts are related to this context?",
        "Can you predict the next context?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = intelligence.reason_about_context(test_context, question)
        
        # Print relevant parts of the result
        if 'similar_contexts' in result:
            print(f"   Similar contexts: {len(result['similar_contexts'])}")
        if 'matching_patterns' in result:
            print(f"   Matching patterns: {len(result['matching_patterns'])}")
        if 'related_concepts' in result:
            print(f"   Related concepts: {len(result['related_concepts'])}")
        if 'prediction' in result:
            print("   Temporal prediction available")
    
    # Get intelligence metrics
    metrics = intelligence.get_intelligence_metrics()
    print("\n📊 Intelligence Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Test conceptual analogies
    print("\n🔗 Testing conceptual analogies...")
    
    # Create some concepts
    concepts = ["fast", "slow", "efficient", "inefficient", "simple", "complex"]
    for concept in concepts:
        positions = {dim: np.random.uniform(0, 1) for dim in ["efficiency", "complexity", "novelty"]}
        context = conceptual_space.create_context(positions, f"concept_{concept}")
        intelligence.conceptual_mapper.embed_concept(concept, context)
    
    # Test analogy: fast is to slow as efficient is to ?
    analogy_result = intelligence.conceptual_mapper.create_conceptual_analogy(
        "fast", "slow", "efficient"
    )
    print(f"   fast : slow :: efficient : {analogy_result}")
    
    # Save intelligence state
    intelligence.save_intelligence_state("multidimensional_intelligence_state.json")
    
    # Deactivate intelligence
    intelligence.deactivate_intelligence()
    print("\n💤 Intelligence deactivated")