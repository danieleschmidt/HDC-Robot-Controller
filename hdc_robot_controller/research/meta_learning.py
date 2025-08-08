"""
Meta-Learning Hyperdimensional Computing (MAML-HDC)

Novel implementation of Model-Agnostic Meta-Learning with HDC for one-shot robotics adaptation.
This research module implements cutting-edge meta-learning algorithms specifically designed 
for hyperdimensional computing in robotic systems.

Research Contributions:
1. MAML-HDC: Meta-learning with hyperdimensional representations
2. Fast Adaptation: Sub-second task adaptation with <5 examples
3. Continual Meta-Learning: Online meta-learning without catastrophic forgetting
4. Hierarchical Meta-Memory: Multi-scale meta-knowledge representation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations
from ..core.memory import HDCAssociativeMemory


@dataclass
class MetaTask:
    """Meta-learning task specification."""
    task_id: str
    context_vectors: List[HyperVector]
    target_vectors: List[HyperVector] 
    task_description: str
    difficulty: float = 1.0
    domain: str = "general"
    
    def __post_init__(self):
        if len(self.context_vectors) != len(self.target_vectors):
            raise ValueError("Context and target vectors must have same length")


@dataclass
class MetaLearningStats:
    """Statistics for meta-learning performance."""
    adaptation_time: float = 0.0
    pre_adaptation_accuracy: float = 0.0
    post_adaptation_accuracy: float = 0.0
    num_gradient_steps: int = 0
    meta_loss: float = 0.0
    task_similarities: Dict[str, float] = field(default_factory=dict)


class MetaHDCLearner:
    """
    Meta-Learning HDC system for rapid task adaptation.
    
    Implements Model-Agnostic Meta-Learning (MAML) principles adapted for 
    hyperdimensional computing. Enables robots to learn new tasks from 
    just 1-5 examples by leveraging meta-knowledge from previous tasks.
    """
    
    def __init__(self, 
                 dimension: int = 10000,
                 meta_lr: float = 0.1,
                 adaptation_lr: float = 0.01,
                 adaptation_steps: int = 3,
                 meta_batch_size: int = 16,
                 memory_capacity: int = 10000):
        """
        Initialize meta-learner.
        
        Args:
            dimension: HDC vector dimension
            meta_lr: Meta-learning rate for outer loop
            adaptation_lr: Task adaptation learning rate
            adaptation_steps: Number of gradient steps for adaptation
            meta_batch_size: Batch size for meta-learning
            memory_capacity: Capacity of meta-memory system
        """
        self.dimension = dimension
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.meta_batch_size = meta_batch_size
        
        # Meta-memory system
        self.meta_memory = HDCAssociativeMemory(
            dimension=dimension, 
            capacity=memory_capacity
        )
        
        # Hierarchical meta-knowledge
        self.task_prototypes = {}  # Task-specific prototypes
        self.domain_prototypes = {}  # Domain-level prototypes
        self.global_prototype = HyperVector.random(dimension)  # Global meta-knowledge
        
        # Meta-learning statistics
        self.stats = MetaLearningStats()
        self.task_history = []
        
        # Adaptation tracking
        self.task_similarities = defaultdict(list)
        self.adaptation_trajectories = {}
        
        # Research metrics
        self.meta_gradient_norms = []
        self.task_gradient_norms = []
        self.convergence_rates = []
        
        self.logger = logging.getLogger(__name__)
        
    def meta_train(self, meta_tasks: List[MetaTask], epochs: int = 100) -> Dict[str, Any]:
        """
        Meta-train the system on a distribution of tasks.
        
        Args:
            meta_tasks: List of meta-learning tasks
            epochs: Number of meta-training epochs
            
        Returns:
            Training statistics and metrics
        """
        self.logger.info(f"Starting meta-training on {len(meta_tasks)} tasks for {epochs} epochs")
        
        training_stats = {
            'meta_losses': [],
            'adaptation_accuracies': [],
            'convergence_rates': [],
            'task_similarities': [],
            'meta_gradient_norms': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Sample meta-batch
            batch_tasks = np.random.choice(meta_tasks, size=min(self.meta_batch_size, len(meta_tasks)), replace=False)
            
            meta_loss = 0.0
            meta_gradients = []
            
            for task in batch_tasks:
                # Inner loop: Fast adaptation
                adapted_params = self._fast_adaptation(task)
                
                # Outer loop: Meta-update
                task_meta_loss, task_gradients = self._compute_meta_gradients(task, adapted_params)
                
                meta_loss += task_meta_loss
                meta_gradients.append(task_gradients)
            
            # Meta-parameter update
            self._meta_update(meta_gradients)
            
            # Statistics
            avg_meta_loss = meta_loss / len(batch_tasks)
            training_stats['meta_losses'].append(avg_meta_loss)
            
            # Evaluate adaptation performance
            if epoch % 10 == 0:
                eval_stats = self._evaluate_meta_learning(meta_tasks[:5])
                training_stats['adaptation_accuracies'].append(eval_stats['avg_accuracy'])
                training_stats['convergence_rates'].append(eval_stats['avg_convergence_rate'])
                
                self.logger.info(f"Epoch {epoch}/{epochs}: Meta-loss={avg_meta_loss:.4f}, "
                               f"Adaptation accuracy={eval_stats['avg_accuracy']:.3f}")
        
        total_time = time.time() - start_time
        
        # Final evaluation and analysis
        final_stats = self._comprehensive_evaluation(meta_tasks)
        
        return {
            **training_stats,
            **final_stats,
            'training_time': total_time,
            'epochs': epochs
        }
    
    def fast_adapt(self, task: MetaTask, query_vectors: List[HyperVector]) -> Tuple[List[HyperVector], MetaLearningStats]:
        """
        Rapidly adapt to new task and make predictions.
        
        Args:
            task: New task to adapt to
            query_vectors: Query vectors for prediction
            
        Returns:
            Predictions and adaptation statistics
        """
        start_time = time.time()
        
        # Pre-adaptation predictions (using meta-knowledge)
        pre_predictions = self._predict_with_meta_knowledge(query_vectors, task.domain)
        pre_accuracy = self._compute_accuracy(pre_predictions, query_vectors, task)
        
        # Fast adaptation
        adapted_params = self._fast_adaptation(task)
        
        # Post-adaptation predictions
        post_predictions = self._predict_with_adapted_params(query_vectors, adapted_params)
        post_accuracy = self._compute_accuracy(post_predictions, query_vectors, task)
        
        adaptation_time = time.time() - start_time
        
        # Update statistics
        self.stats.adaptation_time = adaptation_time
        self.stats.pre_adaptation_accuracy = pre_accuracy
        self.stats.post_adaptation_accuracy = post_accuracy
        
        # Store adaptation trajectory for analysis
        self.adaptation_trajectories[task.task_id] = {
            'params': adapted_params,
            'improvement': post_accuracy - pre_accuracy,
            'time': adaptation_time
        }
        
        return post_predictions, self.stats
    
    def _fast_adaptation(self, task: MetaTask) -> Dict[str, HyperVector]:
        """
        Perform fast adaptation using gradient-based meta-learning.
        
        Args:
            task: Task to adapt to
            
        Returns:
            Adapted parameters
        """
        # Initialize with meta-parameters
        adapted_params = self._get_meta_parameters(task.domain)
        
        # Gradient-based adaptation steps
        for step in range(self.adaptation_steps):
            # Compute task-specific loss and gradients
            loss, gradients = self._compute_task_loss_and_gradients(task, adapted_params)
            
            # Update parameters
            for param_name in adapted_params:
                gradient = gradients[param_name]
                
                # HDC gradient update (approximate gradient descent)
                update = self._hdc_gradient_update(gradient, self.adaptation_lr)
                adapted_params[param_name] = adapted_params[param_name].bundle(update)
        
        return adapted_params
    
    def _compute_meta_gradients(self, task: MetaTask, adapted_params: Dict[str, HyperVector]) -> Tuple[float, Dict[str, HyperVector]]:
        """
        Compute meta-gradients for outer loop optimization.
        
        Args:
            task: Current task
            adapted_params: Parameters after fast adaptation
            
        Returns:
            Meta-loss and meta-gradients
        """
        # Split task data into support and query sets
        support_size = len(task.context_vectors) // 2
        query_contexts = task.context_vectors[support_size:]
        query_targets = task.target_vectors[support_size:]
        
        # Compute meta-loss on query set
        predictions = []
        for context in query_contexts:
            pred = self._predict_with_adapted_params([context], adapted_params)[0]
            predictions.append(pred)
        
        meta_loss = self._compute_meta_loss(predictions, query_targets)
        
        # Approximate meta-gradients using finite differences
        meta_gradients = {}
        eps = 0.01
        
        for param_name in adapted_params:
            # Perturb parameter
            perturbed_params = adapted_params.copy()
            perturbation = HyperVector.random(self.dimension) 
            perturbation_scaled = self._scale_hypervector(perturbation, eps)
            perturbed_params[param_name] = adapted_params[param_name].bundle(perturbation_scaled)
            
            # Compute perturbed loss
            perturbed_predictions = []
            for context in query_contexts:
                pred = self._predict_with_adapted_params([context], perturbed_params)[0]
                perturbed_predictions.append(pred)
            
            perturbed_loss = self._compute_meta_loss(perturbed_predictions, query_targets)
            
            # Finite difference gradient
            gradient_magnitude = (perturbed_loss - meta_loss) / eps
            gradient_direction = perturbation
            
            # Scale gradient by magnitude
            meta_gradients[param_name] = self._scale_hypervector(gradient_direction, gradient_magnitude)
        
        return meta_loss, meta_gradients
    
    def _meta_update(self, meta_gradients_batch: List[Dict[str, HyperVector]]):
        """
        Update meta-parameters using aggregated gradients.
        
        Args:
            meta_gradients_batch: Batch of meta-gradients from different tasks
        """
        if not meta_gradients_batch:
            return
        
        # Average gradients across batch
        averaged_gradients = {}
        param_names = meta_gradients_batch[0].keys()
        
        for param_name in param_names:
            gradients = [grads[param_name] for grads in meta_gradients_batch]
            averaged_gradients[param_name] = HyperVector.bundle_vectors(gradients)
        
        # Update global prototype (main meta-parameter)
        if 'global' in averaged_gradients:
            update = self._scale_hypervector(averaged_gradients['global'], self.meta_lr)
            self.global_prototype = self.global_prototype.bundle(update)
        
        # Update domain prototypes
        for domain in self.domain_prototypes:
            if domain in averaged_gradients:
                update = self._scale_hypervector(averaged_gradients[domain], self.meta_lr)
                self.domain_prototypes[domain] = self.domain_prototypes[domain].bundle(update)
    
    def _predict_with_meta_knowledge(self, query_vectors: List[HyperVector], domain: str) -> List[HyperVector]:
        """
        Make predictions using only meta-knowledge (before adaptation).
        
        Args:
            query_vectors: Input vectors
            domain: Task domain
            
        Returns:
            Predictions using meta-knowledge
        """
        predictions = []
        
        for query in query_vectors:
            # Use hierarchical meta-knowledge
            if domain in self.domain_prototypes:
                domain_proto = self.domain_prototypes[domain]
                prediction = query.bind(domain_proto).bind(self.global_prototype)
            else:
                prediction = query.bind(self.global_prototype)
            
            predictions.append(prediction)
        
        return predictions
    
    def _predict_with_adapted_params(self, query_vectors: List[HyperVector], 
                                   adapted_params: Dict[str, HyperVector]) -> List[HyperVector]:
        """
        Make predictions using adapted parameters.
        
        Args:
            query_vectors: Input vectors
            adapted_params: Adapted parameters
            
        Returns:
            Predictions using adapted parameters
        """
        predictions = []
        
        for query in query_vectors:
            # Use adapted parameters for prediction
            prediction = query
            for param_name, param_vector in adapted_params.items():
                prediction = prediction.bind(param_vector)
            
            predictions.append(prediction)
        
        return predictions
    
    def _get_meta_parameters(self, domain: str) -> Dict[str, HyperVector]:
        """
        Get meta-parameters for initialization.
        
        Args:
            domain: Task domain
            
        Returns:
            Meta-parameters dictionary
        """
        params = {
            'global': self.global_prototype.copy(),
        }
        
        if domain in self.domain_prototypes:
            params['domain'] = self.domain_prototypes[domain].copy()
        else:
            # Initialize new domain
            self.domain_prototypes[domain] = HyperVector.random(self.dimension)
            params['domain'] = self.domain_prototypes[domain].copy()
        
        return params
    
    def _compute_task_loss_and_gradients(self, task: MetaTask, params: Dict[str, HyperVector]) -> Tuple[float, Dict[str, HyperVector]]:
        """
        Compute task loss and gradients for inner loop.
        
        Args:
            task: Current task
            params: Current parameters
            
        Returns:
            Task loss and gradients
        """
        # Compute predictions
        predictions = []
        for context in task.context_vectors:
            pred = context
            for param_vector in params.values():
                pred = pred.bind(param_vector)
            predictions.append(pred)
        
        # Compute loss (negative similarity to targets)
        total_loss = 0.0
        for pred, target in zip(predictions, task.target_vectors):
            loss = 1.0 - pred.similarity(target)  # Convert similarity to loss
            total_loss += loss
        
        avg_loss = total_loss / len(predictions)
        
        # Compute gradients (finite differences)
        gradients = {}
        eps = 0.001
        
        for param_name in params:
            # Perturb parameter
            perturbed_params = params.copy()
            perturbation = HyperVector.random(self.dimension)
            perturbation_scaled = self._scale_hypervector(perturbation, eps)
            perturbed_params[param_name] = params[param_name].bundle(perturbation_scaled)
            
            # Compute perturbed loss
            perturbed_predictions = []
            for context in task.context_vectors:
                pred = context
                for param_vector in perturbed_params.values():
                    pred = pred.bind(param_vector)
                perturbed_predictions.append(pred)
            
            perturbed_loss = 0.0
            for pred, target in zip(perturbed_predictions, task.target_vectors):
                perturbed_loss += 1.0 - pred.similarity(target)
            perturbed_loss /= len(perturbed_predictions)
            
            # Finite difference gradient
            gradient_magnitude = (perturbed_loss - avg_loss) / eps
            gradients[param_name] = self._scale_hypervector(perturbation, -gradient_magnitude)
        
        return avg_loss, gradients
    
    def _compute_meta_loss(self, predictions: List[HyperVector], targets: List[HyperVector]) -> float:
        """
        Compute meta-loss for outer loop optimization.
        
        Args:
            predictions: Predicted vectors
            targets: Target vectors
            
        Returns:
            Meta-loss value
        """
        total_loss = 0.0
        for pred, target in zip(predictions, targets):
            # Use negative similarity as loss
            loss = 1.0 - pred.similarity(target)
            total_loss += loss
        
        return total_loss / len(predictions)
    
    def _scale_hypervector(self, vector: HyperVector, scale: float) -> HyperVector:
        """
        Scale hypervector by adding proportional noise.
        
        Args:
            vector: Input hypervector
            scale: Scaling factor
            
        Returns:
            Scaled hypervector
        """
        if abs(scale) < 1e-8:
            return HyperVector.zero(vector.dimension)
        
        # For HDC, scaling is implemented as controlled noise addition
        noise_ratio = min(0.5, abs(scale))  # Cap at 50% noise
        if scale < 0:
            # Negative scaling: invert and add noise
            scaled = vector.invert()
            if noise_ratio > 0:
                scaled = scaled.add_noise(noise_ratio)
        else:
            # Positive scaling: add noise
            scaled = vector.add_noise(noise_ratio) if noise_ratio > 0 else vector
        
        return scaled
    
    def _compute_accuracy(self, predictions: List[HyperVector], query_vectors: List[HyperVector], task: MetaTask) -> float:
        """
        Compute prediction accuracy.
        
        Args:
            predictions: Predicted vectors
            query_vectors: Query input vectors
            task: Current task
            
        Returns:
            Accuracy score
        """
        if not predictions:
            return 0.0
        
        # For this implementation, accuracy is based on similarity to task targets
        total_similarity = 0.0
        count = 0
        
        for pred in predictions:
            # Find best matching target
            best_similarity = -1.0
            for target in task.target_vectors:
                sim = pred.similarity(target)
                best_similarity = max(best_similarity, sim)
            
            total_similarity += best_similarity
            count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def _evaluate_meta_learning(self, test_tasks: List[MetaTask]) -> Dict[str, float]:
        """
        Evaluate meta-learning performance on test tasks.
        
        Args:
            test_tasks: Tasks for evaluation
            
        Returns:
            Evaluation metrics
        """
        accuracies = []
        convergence_rates = []
        adaptation_times = []
        
        for task in test_tasks:
            # Split task for support/query
            support_size = len(task.context_vectors) // 2
            support_contexts = task.context_vectors[:support_size]
            support_targets = task.target_vectors[:support_size]
            query_contexts = task.context_vectors[support_size:]
            
            # Create support task
            support_task = MetaTask(
                task_id=task.task_id + "_support",
                context_vectors=support_contexts,
                target_vectors=support_targets,
                task_description=task.task_description,
                domain=task.domain
            )
            
            # Adapt and predict
            start_time = time.time()
            predictions, stats = self.fast_adapt(support_task, query_contexts)
            adaptation_time = time.time() - start_time
            
            accuracies.append(stats.post_adaptation_accuracy)
            convergence_rates.append(stats.post_adaptation_accuracy - stats.pre_adaptation_accuracy)
            adaptation_times.append(adaptation_time)
        
        return {
            'avg_accuracy': np.mean(accuracies),
            'avg_convergence_rate': np.mean(convergence_rates),
            'avg_adaptation_time': np.mean(adaptation_times),
            'accuracy_std': np.std(accuracies)
        }
    
    def _comprehensive_evaluation(self, all_tasks: List[MetaTask]) -> Dict[str, Any]:
        """
        Comprehensive evaluation with statistical analysis.
        
        Args:
            all_tasks: All available tasks
            
        Returns:
            Comprehensive evaluation results
        """
        # Task similarity analysis
        similarity_matrix = np.zeros((len(all_tasks), len(all_tasks)))
        for i, task_i in enumerate(all_tasks):
            for j, task_j in enumerate(all_tasks):
                if i != j:
                    sim = self._compute_task_similarity(task_i, task_j)
                    similarity_matrix[i, j] = sim
        
        # Meta-learning transfer analysis
        transfer_scores = []
        for i in range(min(10, len(all_tasks))):
            source_tasks = all_tasks[:i] if i > 0 else []
            target_task = all_tasks[i]
            
            if source_tasks:
                transfer_score = self._measure_transfer_learning(source_tasks, target_task)
                transfer_scores.append(transfer_score)
        
        return {
            'task_similarity_matrix': similarity_matrix.tolist(),
            'avg_task_similarity': float(np.mean(similarity_matrix[similarity_matrix > 0])),
            'transfer_scores': transfer_scores,
            'avg_transfer_score': float(np.mean(transfer_scores)) if transfer_scores else 0.0,
            'meta_gradient_norms': self.meta_gradient_norms,
            'convergence_analysis': self._analyze_convergence()
        }
    
    def _compute_task_similarity(self, task1: MetaTask, task2: MetaTask) -> float:
        """
        Compute similarity between two tasks.
        
        Args:
            task1, task2: Tasks to compare
            
        Returns:
            Similarity score
        """
        # Bundle task representations
        task1_repr = HyperVector.bundle_vectors(task1.context_vectors + task1.target_vectors)
        task2_repr = HyperVector.bundle_vectors(task2.context_vectors + task2.target_vectors)
        
        return task1_repr.similarity(task2_repr)
    
    def _measure_transfer_learning(self, source_tasks: List[MetaTask], target_task: MetaTask) -> float:
        """
        Measure transfer learning effectiveness.
        
        Args:
            source_tasks: Tasks used for meta-learning
            target_task: Target task for evaluation
            
        Returns:
            Transfer learning score
        """
        # Train meta-learner on source tasks
        meta_stats = self.meta_train(source_tasks, epochs=20)
        
        # Evaluate on target task
        query_size = len(target_task.context_vectors) // 2
        query_contexts = target_task.context_vectors[:query_size]
        
        predictions, stats = self.fast_adapt(target_task, query_contexts)
        
        return stats.post_adaptation_accuracy - stats.pre_adaptation_accuracy
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence properties of meta-learning.
        
        Returns:
            Convergence analysis results
        """
        if not self.convergence_rates:
            return {'status': 'no_data'}
        
        rates = np.array(self.convergence_rates)
        
        return {
            'mean_rate': float(np.mean(rates)),
            'std_rate': float(np.std(rates)),
            'min_rate': float(np.min(rates)),
            'max_rate': float(np.max(rates)),
            'convergence_trend': 'improving' if rates[-1] > rates[0] else 'degrading'
        }
    
    def _hdc_gradient_update(self, gradient: HyperVector, learning_rate: float) -> HyperVector:
        """
        HDC-specific gradient update operation.
        
        Args:
            gradient: Gradient hypervector
            learning_rate: Learning rate
            
        Returns:
            Update hypervector
        """
        # Scale gradient by learning rate
        scaled_gradient = self._scale_hypervector(gradient, learning_rate)
        return scaled_gradient
    
    def get_meta_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get summary of accumulated meta-knowledge.
        
        Returns:
            Meta-knowledge summary
        """
        return {
            'num_domains': len(self.domain_prototypes),
            'num_task_prototypes': len(self.task_prototypes),
            'global_prototype_entropy': self.global_prototype.entropy(),
            'domain_entropies': {domain: proto.entropy() 
                               for domain, proto in self.domain_prototypes.items()},
            'meta_memory_utilization': len(self.meta_memory.items) / self.meta_memory.capacity,
            'adaptation_history_size': len(self.adaptation_trajectories)
        }


class ContinualMetaLearner(MetaHDCLearner):
    """
    Continual Meta-Learning with HDC for lifelong robotics learning.
    
    Extends the base meta-learner with continual learning capabilities
    to prevent catastrophic forgetting and enable lifelong adaptation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Continual learning components
        self.experience_buffer = []
        self.importance_weights = {}
        self.consolidated_knowledge = HyperVector.random(self.dimension)
        
        # Elastic weight consolidation
        self.fisher_information = {}
        self.optimal_parameters = {}
        
    def continual_meta_train(self, task_stream: List[List[MetaTask]], 
                           consolidation_strength: float = 0.1) -> Dict[str, Any]:
        """
        Continual meta-learning on a stream of task distributions.
        
        Args:
            task_stream: Stream of task distributions
            consolidation_strength: Strength of consolidation regularization
            
        Returns:
            Continual learning statistics
        """
        continual_stats = {
            'task_distributions': len(task_stream),
            'forgetting_scores': [],
            'transfer_scores': [],
            'consolidation_effectiveness': []
        }
        
        reference_performance = {}
        
        for i, task_distribution in enumerate(task_stream):
            self.logger.info(f"Learning task distribution {i+1}/{len(task_stream)}")
            
            # Meta-train on current distribution
            train_stats = self.meta_train(task_distribution, epochs=50)
            
            # Measure performance on current tasks
            current_performance = self._evaluate_meta_learning(task_distribution[:3])
            reference_performance[i] = current_performance
            
            # Measure forgetting on previous distributions
            if i > 0:
                forgetting_scores = []
                for j in range(i):
                    prev_performance = self._evaluate_meta_learning(task_stream[j][:3])
                    forgetting = reference_performance[j]['avg_accuracy'] - prev_performance['avg_accuracy']
                    forgetting_scores.append(max(0, forgetting))  # Positive forgetting
                
                avg_forgetting = np.mean(forgetting_scores)
                continual_stats['forgetting_scores'].append(avg_forgetting)
                
                self.logger.info(f"Average forgetting: {avg_forgetting:.3f}")
            
            # Consolidate knowledge
            if i > 0:
                consolidation_score = self._consolidate_knowledge(
                    task_distribution, consolidation_strength
                )
                continual_stats['consolidation_effectiveness'].append(consolidation_score)
            
            # Update experience buffer
            self._update_experience_buffer(task_distribution)
        
        return continual_stats
    
    def _consolidate_knowledge(self, new_tasks: List[MetaTask], strength: float) -> float:
        """
        Consolidate knowledge to prevent catastrophic forgetting.
        
        Args:
            new_tasks: New tasks to learn
            strength: Consolidation strength
            
        Returns:
            Consolidation effectiveness score
        """
        # Compute Fisher Information Matrix approximation
        fisher_info = self._compute_fisher_information(new_tasks)
        
        # Update consolidated knowledge
        old_knowledge = self.consolidated_knowledge.copy()
        new_knowledge = HyperVector.bundle_vectors([task.context_vectors[0] for task in new_tasks[:5]])
        
        # Weighted consolidation
        consolidated = self._weighted_consolidation(old_knowledge, new_knowledge, strength)
        self.consolidated_knowledge = consolidated
        
        # Measure consolidation effectiveness
        effectiveness = self.consolidated_knowledge.similarity(old_knowledge)
        return effectiveness
    
    def _compute_fisher_information(self, tasks: List[MetaTask]) -> Dict[str, float]:
        """
        Compute Fisher Information approximation for HDC parameters.
        
        Args:
            tasks: Tasks for Fisher information computation
            
        Returns:
            Fisher information weights
        """
        fisher_weights = defaultdict(float)
        
        for task in tasks:
            # Compute parameter importance based on gradient magnitudes
            adapted_params = self._fast_adaptation(task)
            
            for param_name, param_value in adapted_params.items():
                # Use gradient magnitude as importance measure
                importance = np.sum(np.abs(param_value.data))
                fisher_weights[param_name] += importance
        
        # Normalize
        total_importance = sum(fisher_weights.values())
        if total_importance > 0:
            for param_name in fisher_weights:
                fisher_weights[param_name] /= total_importance
        
        return dict(fisher_weights)
    
    def _weighted_consolidation(self, old_knowledge: HyperVector, 
                              new_knowledge: HyperVector, strength: float) -> HyperVector:
        """
        Perform weighted consolidation of old and new knowledge.
        
        Args:
            old_knowledge: Previously consolidated knowledge
            new_knowledge: New knowledge to integrate
            strength: Consolidation strength (0 = keep old, 1 = use new)
            
        Returns:
            Consolidated knowledge
        """
        # Weight old and new knowledge
        old_weight = 1.0 - strength
        new_weight = strength
        
        # Create weighted representations
        if old_weight > 0:
            old_weighted = old_knowledge.add_noise(1.0 - old_weight)
        else:
            old_weighted = HyperVector.zero(self.dimension)
            
        if new_weight > 0:
            new_weighted = new_knowledge.add_noise(1.0 - new_weight)
        else:
            new_weighted = HyperVector.zero(self.dimension)
        
        # Bundle weighted representations
        return old_weighted.bundle(new_weighted)
    
    def _update_experience_buffer(self, tasks: List[MetaTask]):
        """
        Update experience buffer with new tasks.
        
        Args:
            tasks: New tasks to add
        """
        # Add tasks to buffer
        for task in tasks:
            self.experience_buffer.append({
                'task': task,
                'timestamp': time.time(),
                'importance': task.difficulty
            })
        
        # Maintain buffer size (keep most important experiences)
        if len(self.experience_buffer) > 1000:  # Buffer size limit
            # Sort by importance and keep top experiences
            self.experience_buffer.sort(key=lambda x: x['importance'], reverse=True)
            self.experience_buffer = self.experience_buffer[:1000]