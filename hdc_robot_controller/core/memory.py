"""
Memory systems for HDC operations including associative, episodic and working memory.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import time
from collections import deque
from .hypervector import HyperVector


class AssociativeMemory:
    """Associative memory for storing and retrieving hypervectors."""
    
    def __init__(self, dimension: int = 10000, similarity_threshold: float = 0.7):
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between -1 and 1")
            
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.memory: Dict[str, Dict[str, Any]] = {}
    
    def store(self, label: str, vector: HyperVector, confidence: float = 1.0) -> None:
        """Store a hypervector with label and confidence."""
        if not label:
            raise ValueError("Label cannot be empty")
        if vector.dimension != self.dimension:
            raise ValueError("Vector dimension mismatch")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        
        self.memory[label] = {
            'vector': vector,
            'confidence': confidence,
            'access_count': 0,
            'last_access': time.time()
        }
    
    def store_with_update(self, label: str, vector: HyperVector, learning_rate: float = 0.1) -> None:
        """Store or update existing entry with exponential moving average."""
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("Learning rate must be between 0 and 1")
        
        if label in self.memory:
            # Update existing entry
            existing = self.memory[label]
            old_vector = existing['vector']
            
            # Weighted average
            new_data = (1.0 - learning_rate) * old_vector.data + learning_rate * vector.data
            new_vector = HyperVector(self.dimension, np.where(new_data > 0, 1, -1).astype(np.int8))
            
            existing['vector'] = new_vector
            existing['confidence'] = min(1.0, existing['confidence'] + learning_rate * 0.1)
            existing['access_count'] += 1
            existing['last_access'] = time.time()
        else:
            self.store(label, vector, 0.5)
    
    def query(self, query_vector: HyperVector, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query memory for similar vectors."""
        if query_vector.dimension != self.dimension:
            raise ValueError("Query vector dimension mismatch")
        if max_results <= 0:
            raise ValueError("Max results must be positive")
        
        results = []
        for label, entry in self.memory.items():
            similarity = query_vector.similarity(entry['vector'])
            results.append({
                'label': label,
                'vector': entry['vector'],
                'similarity': similarity,
                'confidence': entry['confidence']
            })
            
            # Update access statistics
            entry['access_count'] += 1
            entry['last_access'] = time.time()
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:max_results]
    
    def query_best(self, query_vector: HyperVector) -> Dict[str, Any]:
        """Get best matching entry."""
        results = self.query(query_vector, 1)
        if not results:
            raise RuntimeError("No entries in memory")
        return results[0]
    
    def query_threshold(self, query_vector: HyperVector, threshold: float) -> List[Dict[str, Any]]:
        """Query with similarity threshold."""
        all_results = self.query(query_vector, len(self.memory))
        return [r for r in all_results if r['similarity'] >= threshold]
    
    def contains(self, label: str) -> bool:
        """Check if label exists in memory."""
        return label in self.memory
    
    def remove(self, label: str) -> None:
        """Remove entry from memory."""
        if label not in self.memory:
            raise ValueError(f"Label '{label}' not found")
        del self.memory[label]
    
    def clear(self) -> None:
        """Clear all memory."""
        self.memory.clear()
    
    def size(self) -> int:
        """Get number of entries."""
        return len(self.memory)
    
    def get_labels(self) -> List[str]:
        """Get all labels."""
        return list(self.memory.keys())
    
    def get_confidence(self, label: str) -> float:
        """Get confidence for label."""
        if label not in self.memory:
            raise ValueError(f"Label '{label}' not found")
        return self.memory[label]['confidence']
    
    def update_confidence(self, label: str, confidence: float) -> None:
        """Update confidence for label."""
        if label not in self.memory:
            raise ValueError(f"Label '{label}' not found")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        
        self.memory[label]['confidence'] = confidence
    
    def decay_confidence(self, decay_rate: float = 0.01) -> None:
        """Decay confidence for all entries."""
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("Decay rate must be between 0 and 1")
        
        for entry in self.memory.values():
            entry['confidence'] *= (1.0 - decay_rate)
    
    def remove_low_confidence(self, min_confidence: float = 0.1) -> None:
        """Remove entries with low confidence."""
        to_remove = [label for label, entry in self.memory.items() 
                    if entry['confidence'] < min_confidence]
        for label in to_remove:
            del self.memory[label]
    
    def save_to_file(self, filename: str) -> None:
        """Save memory to file."""
        data = {
            'dimension': self.dimension,
            'similarity_threshold': self.similarity_threshold,
            'entries': {}
        }
        
        for label, entry in self.memory.items():
            data['entries'][label] = {
                'vector_data': entry['vector'].data.tolist(),
                'confidence': entry['confidence'],
                'access_count': entry['access_count'],
                'last_access': entry['last_access']
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str) -> None:
        """Load memory from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if data['dimension'] != self.dimension:
            raise ValueError("Dimension mismatch in loaded file")
        
        self.similarity_threshold = data['similarity_threshold']
        self.memory.clear()
        
        for label, entry_data in data['entries'].items():
            vector_data = np.array(entry_data['vector_data'], dtype=np.int8)
            vector = HyperVector(self.dimension, vector_data)
            
            self.memory[label] = {
                'vector': vector,
                'confidence': entry_data['confidence'],
                'access_count': entry_data['access_count'],
                'last_access': entry_data['last_access']
            }


class EpisodicMemory:
    """Episodic memory for storing temporal sequences."""
    
    def __init__(self, dimension: int = 10000, max_episodes: int = 1000):
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        if max_episodes <= 0:
            raise ValueError("Max episodes must be positive")
            
        self.dimension = dimension
        self.max_episodes = max_episodes
        self.episodes: List[Dict[str, Any]] = []
    
    def store_episode(self, sequence: List[HyperVector], labels: Optional[List[str]] = None,
                     importance: float = 1.0) -> None:
        """Store an episode (sequence of hypervectors)."""
        if not sequence:
            raise ValueError("Episode sequence cannot be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("Importance must be between 0 and 1")
        
        # Validate dimensions
        for vector in sequence:
            if vector.dimension != self.dimension:
                raise ValueError("Vector dimension mismatch in sequence")
        
        episode = {
            'sequence': sequence,
            'labels': labels or [],
            'timestamp': time.time(),
            'importance': importance
        }
        
        self.episodes.append(episode)
        
        # Maintain size limit
        if len(self.episodes) > self.max_episodes:
            # Remove oldest with lowest importance
            self.episodes.sort(key=lambda e: (e['importance'], e['timestamp']))
            self.episodes = self.episodes[1:]  # Remove oldest
    
    def query_similar_episodes(self, query_sequence: List[HyperVector], 
                              similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Query for similar episodes."""
        if not query_sequence:
            raise ValueError("Query sequence cannot be empty")
        
        from . import create_sequence
        query_hv = create_sequence(query_sequence)
        
        similar_episodes = []
        for episode in self.episodes:
            episode_hv = create_sequence(episode['sequence'])
            similarity = query_hv.similarity(episode_hv)
            
            if similarity >= similarity_threshold:
                episode_with_sim = episode.copy()
                episode_with_sim['similarity'] = similarity
                similar_episodes.append(episode_with_sim)
        
        # Sort by similarity
        similar_episodes.sort(key=lambda e: e['similarity'], reverse=True)
        return similar_episodes
    
    def query_by_pattern(self, pattern: HyperVector, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query episodes containing similar patterns."""
        matches = []
        
        for episode in self.episodes:
            max_similarity = 0.0
            for vector in episode['sequence']:
                similarity = pattern.similarity(vector)
                max_similarity = max(max_similarity, similarity)
            
            episode_with_sim = episode.copy()
            episode_with_sim['similarity'] = max_similarity
            matches.append(episode_with_sim)
        
        # Sort and limit
        matches.sort(key=lambda e: e['similarity'], reverse=True)
        return matches[:max_results]
    
    def get_recent_episodes(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent episodes."""
        sorted_episodes = sorted(self.episodes, key=lambda e: e['timestamp'], reverse=True)
        return sorted_episodes[:count]
    
    def size(self) -> int:
        """Get number of episodes."""
        return len(self.episodes)
    
    def decay_importance(self, decay_rate: float = 0.02) -> None:
        """Decay importance of all episodes."""
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("Decay rate must be between 0 and 1")
        
        for episode in self.episodes:
            episode['importance'] *= (1.0 - decay_rate)
    
    def prune_old_episodes(self, max_age_seconds: float = 7 * 24 * 3600) -> None:
        """Remove episodes older than max_age."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        self.episodes = [e for e in self.episodes if e['timestamp'] >= cutoff_time]


class WorkingMemory:
    """Working memory for short-term storage and pattern detection."""
    
    def __init__(self, dimension: int = 10000, capacity: int = 100):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
            
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.context = HyperVector.zero(dimension)
    
    def push(self, vector: HyperVector, context: str = "") -> None:
        """Add vector to working memory."""
        self.buffer.append({'vector': vector, 'context': context, 'timestamp': time.time()})
        
        # Update running context
        if len(self.buffer) == 1:
            self.context = vector
        else:
            self.context = self.context.bundle(vector)
    
    def pop(self) -> HyperVector:
        """Remove and return most recent vector."""
        if not self.buffer:
            raise RuntimeError("Cannot pop from empty working memory")
        
        item = self.buffer.pop()
        
        # Recalculate context
        if self.buffer:
            vectors = [item['vector'] for item in self.buffer]
            self.context = HyperVector.bundle_vectors(vectors)
        else:
            self.context = HyperVector.zero(self.context.dimension)
        
        return item['vector']
    
    def peek(self) -> HyperVector:
        """Return most recent vector without removing."""
        if not self.buffer:
            raise RuntimeError("Cannot peek empty working memory")
        return self.buffer[-1]['vector']
    
    def get_context(self) -> HyperVector:
        """Get current context vector."""
        return self.context
    
    def detect_patterns(self, pattern_length: int = 3) -> List[HyperVector]:
        """Detect patterns in the buffer."""
        if pattern_length <= 0:
            raise ValueError("Pattern length must be positive")
        if len(self.buffer) < pattern_length:
            return []
        
        patterns = []
        for i in range(len(self.buffer) - pattern_length + 1):
            pattern_vectors = [self.buffer[i + j]['vector'] for j in range(pattern_length)]
            from . import create_sequence
            patterns.append(create_sequence(pattern_vectors))
        
        return patterns
    
    def has_pattern(self, pattern: List[HyperVector]) -> bool:
        """Check if pattern exists in buffer."""
        if not pattern or len(self.buffer) < len(pattern):
            return False
        
        from . import create_sequence
        target_pattern = create_sequence(pattern)
        detected_patterns = self.detect_patterns(len(pattern))
        
        threshold = 0.8
        return any(target_pattern.similarity(p) >= threshold for p in detected_patterns)
    
    def clear(self) -> None:
        """Clear working memory."""
        self.buffer.clear()
        self.context = HyperVector.zero(self.context.dimension)
    
    def empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.buffer) == 0
    
    def full(self) -> bool:
        """Check if memory is at capacity."""
        return len(self.buffer) >= self.capacity
    
    def size(self) -> int:
        """Get current size."""
        return len(self.buffer)


class HierarchicalMemory:
    """Hierarchical memory combining associative, episodic, and working memory."""
    
    def __init__(self, dimension: int = 10000):
        self.associative_memory = AssociativeMemory(dimension)
        self.episodic_memory = EpisodicMemory(dimension)
        self.working_memory = WorkingMemory(dimension)
    
    def get_associative_memory(self) -> AssociativeMemory:
        """Get associative memory."""
        return self.associative_memory
    
    def get_episodic_memory(self) -> EpisodicMemory:
        """Get episodic memory."""
        return self.episodic_memory
    
    def get_working_memory(self) -> WorkingMemory:
        """Get working memory."""
        return self.working_memory
    
    def store_experience(self, perception: HyperVector, action: HyperVector,
                        label: str, confidence: float = 1.0) -> None:
        """Store experience across all memory systems."""
        # Combine perception and action
        experience = perception.bind(action)
        
        # Store in associative memory
        self.associative_memory.store(label, experience, confidence)
        
        # Store in episodic memory
        self.episodic_memory.store_episode([perception, action], [label])
        
        # Add to working memory
        self.working_memory.push(experience, label)
    
    def query_experience(self, query: HyperVector, max_results: int = 10) -> List[Dict[str, Any]]:
        """Query experience from associative memory."""
        return self.associative_memory.query(query, max_results)
    
    def total_size(self) -> int:
        """Get total memory size."""
        return (self.associative_memory.size() + 
                self.episodic_memory.size() + 
                self.working_memory.size())
    
    def print_stats(self) -> None:
        """Print memory statistics."""
        print("Hierarchical Memory Statistics:")
        print(f"  Associative Memory: {self.associative_memory.size()} entries")
        print(f"  Episodic Memory: {self.episodic_memory.size()} episodes")
        print(f"  Working Memory: {self.working_memory.size()} items")
        print(f"  Total Size: {self.total_size()}")
    
    def consolidate_all(self, similarity_threshold: float = 0.9) -> None:
        """Consolidate all memory systems."""
        # Decay old memories
        self.associative_memory.decay_confidence()
        self.associative_memory.remove_low_confidence()
        self.episodic_memory.decay_importance()
    
    def cleanup_all(self) -> None:
        """Cleanup all memory systems."""
        self.consolidate_all()
        self.episodic_memory.prune_old_episodes()


# Add create_sequence function for compatibility
def create_sequence(vectors: List[HyperVector]) -> HyperVector:
    """Create sequence hypervector by binding vectors with positions."""
    if not vectors:
        raise ValueError("Cannot create sequence from empty vector list")
    
    dimension = vectors[0].dimension
    result = HyperVector.zero(dimension)
    
    for i, vector in enumerate(vectors):
        # Create position vector
        position_vector = HyperVector.random(dimension, seed=i + 1000)
        
        # Bind element with position and bundle into result
        bound = vector.bind(position_vector)
        result = result.bundle(bound)
    
    return result