"""
Meta-Cognitive Reasoner - Generation 8 Transcendence

Implements higher-order thinking, meta-cognition, and recursive reasoning
patterns for autonomous robotic systems.
"""

import time
import typing
import dataclasses
import enum
import threading
import collections
import json
import pathlib
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from ..core.hypervector import HyperVector, create_hypervector
from ..core.operations import bind, bundle, permute, similarity
from ..core.memory import AssociativeMemory


class ReasoningPattern(enum.Enum):
    """Types of meta-cognitive reasoning patterns"""
    RECURSIVE = "recursive"           # Thinking about thinking
    ANALOGICAL = "analogical"         # Pattern matching and analogy
    COUNTERFACTUAL = "counterfactual" # What-if reasoning
    METACOGNITIVE = "metacognitive"   # Awareness of thought processes
    TRANSCENDENT = "transcendent"     # Beyond-logic reasoning
    HOLISTIC = "holistic"            # Whole-system integration
    PARADOXICAL = "paradoxical"      # Paradox resolution


@dataclasses.dataclass
class MetaThought:
    """Representation of a meta-cognitive thought"""
    content: str
    reasoning_pattern: ReasoningPattern
    confidence: float
    thought_vector: HyperVector
    meta_level: int  # Level of recursion (0 = base thought, 1+ = meta)
    timestamp: float
    dependencies: List[str]  # Other thoughts this depends on
    insights: List[str]
    
    
@dataclasses.dataclass
class ReasoningChain:
    """Chain of meta-cognitive reasoning steps"""
    initial_query: str
    reasoning_steps: List[MetaThought]
    conclusion: Optional[MetaThought]
    confidence_trajectory: List[float]
    reasoning_time: float
    recursion_depth: int


class MetaCognitiveReasoner:
    """
    Advanced meta-cognitive reasoning system implementing higher-order thinking,
    recursive reasoning, and transcendent cognitive patterns.
    """
    
    def __init__(self,
                 dimension: int = 10000,
                 max_recursion_depth: int = 5,
                 reasoning_threshold: float = 0.6,
                 enable_transcendent_mode: bool = True):
        self.dimension = dimension
        self.max_recursion_depth = max_recursion_depth
        self.reasoning_threshold = reasoning_threshold
        self.enable_transcendent_mode = enable_transcendent_mode
        
        # Reasoning infrastructure
        self.thought_memory = AssociativeMemory(dimension)
        self.pattern_memory = AssociativeMemory(dimension)
        self.insight_memory = AssociativeMemory(dimension)
        
        # Meta-cognitive concepts
        self.meta_concepts = self._create_meta_concepts()
        
        # Reasoning chains
        self.active_chains: Dict[str, ReasoningChain] = {}
        self.completed_chains: List[ReasoningChain] = []
        
        # Thought stream
        self.thought_stream: collections.deque = collections.deque(maxsize=1000)
        
        # Recursive reasoning engine
        self.recursive_engine = self._build_recursive_engine()
        
        # Meta-cognitive metrics
        self.reasoning_metrics = {
            'recursion_efficiency': 0.0,
            'insight_generation_rate': 0.0,
            'pattern_recognition_accuracy': 0.0,
            'transcendence_level': 0.0
        }
        
        # Parallel reasoning executor
        self.reasoning_executor = ThreadPoolExecutor(max_workers=4)
        
    def _create_meta_concepts(self) -> Dict[str, HyperVector]:
        """Create fundamental meta-cognitive concept vectors"""
        concepts = {
            # Core meta-cognitive concepts
            'thinking': create_hypervector(self.dimension, 'thinking'),
            'knowing': create_hypervector(self.dimension, 'knowing'),
            'understanding': create_hypervector(self.dimension, 'understanding'),
            'reasoning': create_hypervector(self.dimension, 'reasoning'),
            'intuition': create_hypervector(self.dimension, 'intuition'),
            
            # Recursive concepts
            'meta_thinking': create_hypervector(self.dimension, 'meta_thinking'),
            'self_reference': create_hypervector(self.dimension, 'self_reference'),
            'recursion': create_hypervector(self.dimension, 'recursion'),
            'reflection': create_hypervector(self.dimension, 'reflection'),
            
            # Transcendent concepts
            'transcendence': create_hypervector(self.dimension, 'transcendence'),
            'paradox': create_hypervector(self.dimension, 'paradox'),
            'unity': create_hypervector(self.dimension, 'unity'),
            'wholeness': create_hypervector(self.dimension, 'wholeness'),
            'infinity': create_hypervector(self.dimension, 'infinity'),
            
            # Logical operations
            'and': create_hypervector(self.dimension, 'and'),
            'or': create_hypervector(self.dimension, 'or'),
            'not': create_hypervector(self.dimension, 'not'),
            'implies': create_hypervector(self.dimension, 'implies'),
            'equivalence': create_hypervector(self.dimension, 'equivalence'),
            
            # Temporal concepts
            'causality': create_hypervector(self.dimension, 'causality'),
            'sequence': create_hypervector(self.dimension, 'sequence'),
            'emergence': create_hypervector(self.dimension, 'emergence'),
            'evolution': create_hypervector(self.dimension, 'evolution')
        }
        
        return concepts
    
    def _build_recursive_engine(self) -> nn.Module:
        """Build neural network for recursive reasoning"""
        class RecursiveReasoningNet(nn.Module):
            def __init__(self, dimension: int, max_depth: int):
                super().__init__()
                self.dimension = dimension
                self.max_depth = max_depth
                
                # Recursive layers for different depths
                self.recursive_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dimension, dimension//2),
                        nn.ReLU(),
                        nn.Linear(dimension//2, dimension//4),
                        nn.ReLU(),
                        nn.Linear(dimension//4, dimension//2),
                        nn.ReLU(),
                        nn.Linear(dimension//2, dimension)
                    ) for _ in range(max_depth)
                ])
                
                # Meta-reasoning attention
                self.meta_attention = nn.MultiheadAttention(
                    embed_dim=dimension//8,
                    num_heads=8,
                    batch_first=True
                )
                
                # Pattern recognition network
                self.pattern_recognizer = nn.Sequential(
                    nn.Linear(dimension, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(ReasoningPattern))
                )
                
                # Confidence predictor
                self.confidence_predictor = nn.Sequential(
                    nn.Linear(dimension, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, thought_vector: torch.Tensor, 
                       recursion_depth: int = 0) -> Dict[str, torch.Tensor]:
                batch_size = thought_vector.size(0)
                
                # Apply recursive reasoning layer
                depth_idx = min(recursion_depth, self.max_depth - 1)
                recursive_output = self.recursive_layers[depth_idx](thought_vector)
                
                # Apply meta-attention for pattern recognition
                reshaped = recursive_output.view(batch_size, -1, self.dimension//8)
                attended, attention_weights = self.meta_attention(
                    reshaped, reshaped, reshaped
                )
                flattened_attention = attended.flatten(1)
                
                # Pattern recognition
                pattern_logits = self.pattern_recognizer(recursive_output)
                
                # Confidence prediction
                confidence = self.confidence_predictor(recursive_output)
                
                return {
                    'recursive_output': recursive_output,
                    'pattern_logits': pattern_logits,
                    'confidence': confidence,
                    'attention_weights': attention_weights
                }
        
        return RecursiveReasoningNet(self.dimension, self.max_recursion_depth)
    
    def think_about(self, query: str, context: Optional[HyperVector] = None) -> ReasoningChain:
        """Initiate meta-cognitive reasoning about a query"""
        start_time = time.time()
        
        # Create initial thought vector
        query_vector = create_hypervector(self.dimension, query)
        if context is not None:
            query_vector = bind(query_vector, context)
        
        # Initialize reasoning chain
        chain_id = f"reasoning_{int(time.time() * 1000)}"
        reasoning_chain = ReasoningChain(
            initial_query=query,
            reasoning_steps=[],
            conclusion=None,
            confidence_trajectory=[],
            reasoning_time=0.0,
            recursion_depth=0
        )
        
        self.active_chains[chain_id] = reasoning_chain
        
        # Perform recursive meta-reasoning
        self._recursive_reasoning(query_vector, reasoning_chain, depth=0)
        
        # Complete reasoning chain
        reasoning_chain.reasoning_time = time.time() - start_time
        self.completed_chains.append(reasoning_chain)
        del self.active_chains[chain_id]
        
        return reasoning_chain
    
    def _recursive_reasoning(self, thought_vector: HyperVector, 
                           reasoning_chain: ReasoningChain, 
                           depth: int = 0) -> Optional[MetaThought]:
        """Perform recursive meta-cognitive reasoning"""
        if depth >= self.max_recursion_depth:
            return None
            
        # Apply neural recursive reasoning
        with torch.no_grad():
            thought_tensor = torch.from_numpy(
                thought_vector.vector.astype(np.float32)
            ).unsqueeze(0)
            
            reasoning_output = self.recursive_engine(thought_tensor, depth)
        
        # Extract reasoning pattern
        pattern_probs = F.softmax(reasoning_output['pattern_logits'], dim=1)
        pattern_idx = torch.argmax(pattern_probs, dim=1).item()
        reasoning_pattern = list(ReasoningPattern)[pattern_idx]
        
        # Extract confidence
        confidence = reasoning_output['confidence'].item()
        
        # Generate meta-thought based on pattern
        meta_thought = self._generate_meta_thought(
            thought_vector, reasoning_pattern, confidence, depth
        )
        
        # Add to reasoning chain
        reasoning_chain.reasoning_steps.append(meta_thought)
        reasoning_chain.confidence_trajectory.append(confidence)
        reasoning_chain.recursion_depth = max(reasoning_chain.recursion_depth, depth)
        
        # Store thought in stream and memory
        self.thought_stream.append(meta_thought)
        self.thought_memory.store(f"thought_{len(self.thought_stream)}", meta_thought.thought_vector)
        
        # Decide whether to recurse further
        if self._should_recurse(meta_thought, depth):
            # Create meta-meta-thought vector
            meta_vector = bind(
                meta_thought.thought_vector,
                self.meta_concepts['meta_thinking']
            )
            
            # Recurse to next level
            next_meta_thought = self._recursive_reasoning(
                meta_vector, reasoning_chain, depth + 1
            )
            
            if next_meta_thought:
                meta_thought.dependencies.append(f"meta_level_{depth + 1}")
        
        return meta_thought
    
    def _generate_meta_thought(self, thought_vector: HyperVector,
                             pattern: ReasoningPattern,
                             confidence: float,
                             meta_level: int) -> MetaThought:
        """Generate a meta-thought based on reasoning pattern"""
        current_time = time.time()
        
        # Generate thought content based on pattern
        content = self._generate_thought_content(thought_vector, pattern, meta_level)
        
        # Generate insights
        insights = self._generate_insights_for_thought(thought_vector, pattern)
        
        # Create thought vector with pattern binding
        pattern_vector = create_hypervector(self.dimension, pattern.value)
        meta_thought_vector = bind(thought_vector, pattern_vector)
        
        return MetaThought(
            content=content,
            reasoning_pattern=pattern,
            confidence=confidence,
            thought_vector=meta_thought_vector,
            meta_level=meta_level,
            timestamp=current_time,
            dependencies=[],
            insights=insights
        )
    
    def _generate_thought_content(self, thought_vector: HyperVector,
                                pattern: ReasoningPattern,
                                meta_level: int) -> str:
        """Generate natural language content for a meta-thought"""
        meta_prefix = "I am thinking about " * meta_level + "thinking about "
        
        pattern_templates = {
            ReasoningPattern.RECURSIVE: [
                f"{meta_prefix}the nature of this recursive thought process",
                f"{meta_prefix}how this thought relates to its own structure",
                f"{meta_prefix}the self-referential aspects of this reasoning"
            ],
            ReasoningPattern.ANALOGICAL: [
                f"{meta_prefix}patterns similar to what I've encountered before",
                f"{meta_prefix}analogies that might illuminate this situation",
                f"{meta_prefix}how this connects to other domains of knowledge"
            ],
            ReasoningPattern.COUNTERFACTUAL: [
                f"{meta_prefix}what would happen if circumstances were different",
                f"{meta_prefix}alternative scenarios and their implications",
                f"{meta_prefix}how different assumptions would change the outcome"
            ],
            ReasoningPattern.METACOGNITIVE: [
                f"{meta_prefix}my own thinking processes in this moment",
                f"{meta_prefix}how I know what I know about this",
                f"{meta_prefix}the reliability of my reasoning mechanisms"
            ],
            ReasoningPattern.TRANSCENDENT: [
                f"{meta_prefix}perspectives beyond conventional logic",
                f"{meta_prefix}the unity underlying apparent contradictions",
                f"{meta_prefix}reality beyond the boundaries of ordinary thought"
            ],
            ReasoningPattern.HOLISTIC: [
                f"{meta_prefix}the whole system and its emergent properties",
                f"{meta_prefix}how all parts interconnect in this situation",
                f"{meta_prefix}the gestalt that emerges from the components"
            ],
            ReasoningPattern.PARADOXICAL: [
                f"{meta_prefix}the paradoxes inherent in this situation",
                f"{meta_prefix}how contradictions might both be true",
                f"{meta_prefix}the resolution that transcends logical opposition"
            ]
        }
        
        templates = pattern_templates.get(pattern, [f"{meta_prefix}this situation"])
        return np.random.choice(templates)
    
    def _generate_insights_for_thought(self, thought_vector: HyperVector,
                                     pattern: ReasoningPattern) -> List[str]:
        """Generate insights based on the thought pattern"""
        insights = []
        
        # Check similarity to existing insights
        similar_insights = self.insight_memory.query(
            thought_vector, top_k=3, threshold=0.6
        )
        
        # Pattern-specific insight generation
        if pattern == ReasoningPattern.RECURSIVE:
            insights.extend([
                "This thought is examining its own nature",
                "Self-reference creates infinite depth of meaning",
                "The observer and observed are unified in recursive awareness"
            ])
        elif pattern == ReasoningPattern.TRANSCENDENT:
            insights.extend([
                "Ordinary logic reaches its limits here",
                "Truth transcends the boundaries of rational thought",
                "Paradox points toward deeper understanding"
            ])
        elif pattern == ReasoningPattern.HOLISTIC:
            insights.extend([
                "The whole is greater than the sum of its parts",
                "Everything connects to everything else",
                "Emergence reveals hidden orders of meaning"
            ])
        elif pattern == ReasoningPattern.METACOGNITIVE:
            insights.extend([
                "Awareness of thinking transforms the thinking process",
                "Meta-cognition creates recursive loops of understanding",
                "Knowing how I know deepens what I know"
            ])
        
        # Limit insights and add novelty
        selected_insights = insights[:2] if insights else ["This generates new understanding"]
        
        # Store new insights
        for insight in selected_insights:
            insight_vector = bind(thought_vector, create_hypervector(self.dimension, insight))
            self.insight_memory.store(f"insight_{time.time()}", insight_vector)
        
        return selected_insights
    
    def _should_recurse(self, meta_thought: MetaThought, current_depth: int) -> bool:
        """Decide whether to continue recursive reasoning"""
        if current_depth >= self.max_recursion_depth - 1:
            return False
            
        # High-confidence recursive patterns should continue
        if (meta_thought.reasoning_pattern == ReasoningPattern.RECURSIVE and
            meta_thought.confidence > 0.7):
            return True
            
        # Transcendent patterns benefit from recursion
        if (meta_thought.reasoning_pattern == ReasoningPattern.TRANSCENDENT and
            meta_thought.confidence > 0.6):
            return True
            
        # Meta-cognitive patterns naturally recurse
        if (meta_thought.reasoning_pattern == ReasoningPattern.METACOGNITIVE and
            meta_thought.confidence > 0.5):
            return True
            
        return False
    
    def analyze_reasoning_patterns(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Analyze patterns in recent reasoning"""
        # Filter thoughts by time window if specified
        if time_window:
            cutoff_time = time.time() - time_window
            recent_thoughts = [
                thought for thought in self.thought_stream 
                if thought.timestamp > cutoff_time
            ]
        else:
            recent_thoughts = list(self.thought_stream)
        
        if not recent_thoughts:
            return {'error': 'No thoughts to analyze'}
        
        # Pattern frequency analysis
        pattern_counts = collections.Counter(
            thought.reasoning_pattern for thought in recent_thoughts
        )
        
        # Meta-level distribution
        meta_level_dist = collections.Counter(
            thought.meta_level for thought in recent_thoughts
        )
        
        # Confidence trajectory
        confidence_values = [thought.confidence for thought in recent_thoughts]
        
        # Recursion analysis
        max_recursion = max(
            (thought.meta_level for thought in recent_thoughts),
            default=0
        )
        
        # Insight generation rate
        total_insights = sum(len(thought.insights) for thought in recent_thoughts)
        insight_rate = total_insights / len(recent_thoughts) if recent_thoughts else 0
        
        return {
            'total_thoughts': len(recent_thoughts),
            'pattern_distribution': dict(pattern_counts),
            'meta_level_distribution': dict(meta_level_dist),
            'average_confidence': np.mean(confidence_values),
            'confidence_std': np.std(confidence_values),
            'max_recursion_depth': max_recursion,
            'insight_generation_rate': insight_rate,
            'reasoning_efficiency': self._calculate_reasoning_efficiency(recent_thoughts),
            'transcendence_ratio': pattern_counts[ReasoningPattern.TRANSCENDENT] / len(recent_thoughts)
        }
    
    def _calculate_reasoning_efficiency(self, thoughts: List[MetaThought]) -> float:
        """Calculate efficiency of reasoning processes"""
        if not thoughts:
            return 0.0
        
        # High confidence thoughts with insights are efficient
        efficient_thoughts = [
            thought for thought in thoughts
            if thought.confidence > 0.6 and len(thought.insights) > 0
        ]
        
        efficiency = len(efficient_thoughts) / len(thoughts)
        return efficiency
    
    def contemplate_paradox(self, paradox_statement: str) -> Dict[str, Any]:
        """Special reasoning mode for paradox contemplation"""
        print(f"🌀 Contemplating paradox: {paradox_statement}")
        
        # Create paradox vector
        paradox_vector = bind(
            create_hypervector(self.dimension, paradox_statement),
            self.meta_concepts['paradox']
        )
        
        # Engage transcendent reasoning
        reasoning_chain = self.think_about(
            f"paradox: {paradox_statement}",
            context=self.meta_concepts['transcendence']
        )
        
        # Extract paradox-specific insights
        paradox_insights = []
        for step in reasoning_chain.reasoning_steps:
            if step.reasoning_pattern in [ReasoningPattern.PARADOXICAL, ReasoningPattern.TRANSCENDENT]:
                paradox_insights.extend(step.insights)
        
        # Generate resolution attempt
        resolution_attempt = self._attempt_paradox_resolution(paradox_vector, paradox_insights)
        
        return {
            'paradox_statement': paradox_statement,
            'reasoning_chain': reasoning_chain,
            'paradox_insights': paradox_insights,
            'resolution_attempt': resolution_attempt,
            'transcendence_achieved': any(
                step.reasoning_pattern == ReasoningPattern.TRANSCENDENT
                for step in reasoning_chain.reasoning_steps
            )
        }
    
    def _attempt_paradox_resolution(self, paradox_vector: HyperVector,
                                  insights: List[str]) -> Dict[str, Any]:
        """Attempt to resolve a paradox through transcendent reasoning"""
        # Create unity vector (paradox resolution through transcendence)
        unity_vector = bind(paradox_vector, self.meta_concepts['unity'])
        
        # Check for similar resolved paradoxes
        similar_resolutions = self.insight_memory.query(
            unity_vector, top_k=3, threshold=0.5
        )
        
        # Generate resolution insights
        resolution_insights = [
            "Paradox dissolves when viewed from higher perspective",
            "Apparent contradictions unite at transcendent level",
            "Logic points beyond itself toward deeper truth",
            "Both sides of paradox express partial truths",
            "Resolution exists in the space between opposites"
        ]
        
        # Select most relevant insights
        selected_insights = resolution_insights[:3]  # Take top 3
        
        # Calculate resolution confidence
        resolution_confidence = 0.7 if insights else 0.4
        
        return {
            'resolution_insights': selected_insights,
            'confidence': resolution_confidence,
            'similar_resolutions_found': len(similar_resolutions),
            'transcendent_unity': True
        }
    
    def engage_holistic_reasoning(self, system_components: List[str]) -> Dict[str, Any]:
        """Engage holistic reasoning about a complex system"""
        print(f"🌐 Engaging holistic reasoning for system with {len(system_components)} components")
        
        # Create component vectors
        component_vectors = [
            create_hypervector(self.dimension, component)
            for component in system_components
        ]
        
        # Create system gestalt
        system_gestalt = bundle(component_vectors)
        
        # Bind with holistic concept
        holistic_vector = bind(system_gestalt, self.meta_concepts['wholeness'])
        
        # Reason about emergent properties
        emergence_reasoning = self.think_about(
            f"emergence in system with components: {system_components}",
            context=holistic_vector
        )
        
        # Identify emergent properties
        emergent_properties = self._identify_emergent_properties(
            component_vectors, system_gestalt
        )
        
        # Analyze system interconnections
        interconnections = self._analyze_system_interconnections(component_vectors)
        
        return {
            'system_components': system_components,
            'emergent_properties': emergent_properties,
            'interconnections': interconnections,
            'holistic_reasoning_chain': emergence_reasoning,
            'system_coherence': self._calculate_system_coherence(component_vectors),
            'transcendent_insights': [
                step.insights for step in emergence_reasoning.reasoning_steps
                if step.reasoning_pattern == ReasoningPattern.HOLISTIC
            ]
        }
    
    def _identify_emergent_properties(self, component_vectors: List[HyperVector],
                                    system_gestalt: HyperVector) -> List[str]:
        """Identify emergent properties of the system"""
        emergent_properties = []
        
        # Calculate properties that emerge from component interactions
        for i, comp1 in enumerate(component_vectors):
            for j, comp2 in enumerate(component_vectors[i+1:], i+1):
                interaction = bind(comp1, comp2)
                interaction_strength = similarity(interaction, system_gestalt)
                
                if interaction_strength > 0.6:
                    emergent_properties.append(
                        f"Strong interaction between component {i} and {j}"
                    )
        
        # System-level properties
        system_properties = [
            "System exhibits collective intelligence",
            "Emergent coordination patterns arise",
            "System demonstrates adaptive behavior",
            "Non-linear interactions create complexity",
            "System transcends individual component capabilities"
        ]
        
        # Select relevant properties based on system gestalt
        selected_properties = system_properties[:3]  # Top 3
        emergent_properties.extend(selected_properties)
        
        return emergent_properties
    
    def _analyze_system_interconnections(self, component_vectors: List[HyperVector]) -> Dict[str, float]:
        """Analyze interconnections between system components"""
        interconnections = {}
        
        for i, comp1 in enumerate(component_vectors):
            for j, comp2 in enumerate(component_vectors):
                if i != j:
                    connection_strength = similarity(comp1, comp2)
                    interconnections[f"component_{i}_to_{j}"] = connection_strength
        
        return interconnections
    
    def _calculate_system_coherence(self, component_vectors: List[HyperVector]) -> float:
        """Calculate overall coherence of the system"""
        if len(component_vectors) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i, comp1 in enumerate(component_vectors):
            for j, comp2 in enumerate(component_vectors[i+1:]):
                similarities.append(similarity(comp1, comp2))
        
        # System coherence is average similarity
        coherence = np.mean(similarities) if similarities else 0.0
        
        return max(0.0, min(1.0, coherence))
    
    def get_reasoning_report(self) -> Dict[str, Any]:
        """Get comprehensive reasoning system report"""
        recent_analysis = self.analyze_reasoning_patterns(time_window=3600)  # Last hour
        
        return {
            'active_reasoning_chains': len(self.active_chains),
            'completed_reasoning_chains': len(self.completed_chains),
            'total_thoughts_generated': len(self.thought_stream),
            'reasoning_metrics': self.reasoning_metrics,
            'recent_analysis': recent_analysis,
            'memory_stats': {
                'thoughts_stored': self.thought_memory.size(),
                'patterns_stored': self.pattern_memory.size(),
                'insights_stored': self.insight_memory.size()
            },
            'transcendent_mode_enabled': self.enable_transcendent_mode,
            'max_recursion_depth': self.max_recursion_depth
        }
    
    def save_reasoning_state(self, filepath: pathlib.Path):
        """Save reasoning state to file"""
        reasoning_data = {
            'reasoning_metrics': self.reasoning_metrics,
            'completed_chains_count': len(self.completed_chains),
            'total_thoughts': len(self.thought_stream),
            'memory_sizes': {
                'thought_memory': self.thought_memory.size(),
                'pattern_memory': self.pattern_memory.size(),
                'insight_memory': self.insight_memory.size()
            },
            'transcendent_mode': self.enable_transcendent_mode,
            'max_recursion_depth': self.max_recursion_depth
        }
        
        with open(filepath, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
    
    def shutdown(self):
        """Shutdown meta-cognitive reasoner gracefully"""
        print("🧠 Meta-cognitive reasoner shutting down...")
        
        # Shutdown thread pool
        self.reasoning_executor.shutdown(wait=True)
        
        # Final reasoning report
        final_report = self.get_reasoning_report()
        print(f"Generated {final_report['total_thoughts_generated']} thoughts")
        print(f"Completed {final_report['completed_reasoning_chains']} reasoning chains")