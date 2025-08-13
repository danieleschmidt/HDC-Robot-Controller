"""
Distributed HDC Processing Engine
High-performance distributed computing for hyperdimensional operations.
"""

import numpy as np
import asyncio
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import threading
import queue
import concurrent.futures
import pickle
import hashlib
import zmq
import redis
from pathlib import Path

from ..core.hypervector import HyperVector, weighted_bundle

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of distributed HDC tasks."""
    BUNDLE_OPERATION = "bundle"
    BIND_OPERATION = "bind"
    SIMILARITY_COMPUTATION = "similarity"
    MEMORY_QUERY = "memory_query"
    BATCH_ENCODING = "batch_encoding"
    LEARNING_UPDATE = "learning_update"
    OPTIMIZATION = "optimization"


class WorkerStatus(Enum):
    """Worker node status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class HDCTask:
    """Distributed HDC computation task."""
    task_id: str
    task_type: TaskType
    data: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    retries: int = 3
    created_at: float = field(default_factory=time.time)
    assigned_worker: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class WorkerNode:
    """Distributed worker node information."""
    worker_id: str
    address: str
    port: int
    status: WorkerStatus
    capabilities: List[TaskType]
    current_task: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class HDCTaskScheduler:
    """Advanced task scheduler for distributed HDC operations."""
    
    def __init__(self, max_workers: int = mp.cpu_count()):
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.worker_pool = {}
        
        # Task scheduling metrics
        self.scheduling_stats = {
            'total_tasks_scheduled': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'average_execution_time': 0.0,
            'throughput_per_second': 0.0
        }
        
        # Load balancing strategy
        self.load_balancing_strategy = "round_robin"  # round_robin, least_loaded, capability_based
        self.worker_selector_index = 0
        
    def submit_task(self, task: HDCTask) -> str:
        """Submit task for distributed execution."""
        try:
            # Add to task queue with priority
            priority_score = self._calculate_priority_score(task)
            self.task_queue.put((priority_score, task.task_id, task))
            
            # Track active task
            self.active_tasks[task.task_id] = task
            
            self.scheduling_stats['total_tasks_scheduled'] += 1
            
            logger.debug(f"Task {task.task_id} submitted with priority {priority_score}")
            
            return task.task_id
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise
            
    def _calculate_priority_score(self, task: HDCTask) -> int:
        """Calculate priority score for task scheduling."""
        # Lower score = higher priority
        base_priority = task.priority
        
        # Adjust based on task type
        task_type_priorities = {
            TaskType.SIMILARITY_COMPUTATION: 1,
            TaskType.MEMORY_QUERY: 2,
            TaskType.BUNDLE_OPERATION: 3,
            TaskType.BIND_OPERATION: 3,
            TaskType.BATCH_ENCODING: 4,
            TaskType.LEARNING_UPDATE: 5,
            TaskType.OPTIMIZATION: 6
        }
        
        type_priority = task_type_priorities.get(task.task_type, 5)
        
        # Consider task age (older tasks get higher priority)
        age_factor = int((time.time() - task.created_at) / 10)  # +1 priority per 10 seconds
        
        final_priority = max(1, base_priority + type_priority - age_factor)
        
        return final_priority
        
    def select_worker(self, task: HDCTask) -> Optional[str]:
        """Select optimal worker for task execution."""
        available_workers = [
            worker_id for worker_id, worker in self.worker_pool.items()
            if worker.status == WorkerStatus.IDLE and 
            task.task_type in worker.capabilities
        ]
        
        if not available_workers:
            return None
            
        if self.load_balancing_strategy == "round_robin":
            return self._select_round_robin(available_workers)
        elif self.load_balancing_strategy == "least_loaded":
            return self._select_least_loaded(available_workers)
        elif self.load_balancing_strategy == "capability_based":
            return self._select_capability_based(available_workers, task)
        else:
            return available_workers[0]  # Default to first available
            
    def _select_round_robin(self, available_workers: List[str]) -> str:
        """Round-robin worker selection."""
        selected_worker = available_workers[self.worker_selector_index % len(available_workers)]
        self.worker_selector_index += 1
        return selected_worker
        
    def _select_least_loaded(self, available_workers: List[str]) -> str:
        """Select worker with least current load."""
        def get_load_score(worker_id: str) -> float:
            worker = self.worker_pool[worker_id]
            return worker.performance_metrics.get('current_load', 0.0)
            
        return min(available_workers, key=get_load_score)
        
    def _select_capability_based(self, available_workers: List[str], task: HDCTask) -> str:
        """Select worker based on task-specific capabilities."""
        def get_capability_score(worker_id: str) -> float:
            worker = self.worker_pool[worker_id]
            task_performance_key = f"{task.task_type.value}_performance"
            return worker.performance_metrics.get(task_performance_key, 1.0)
            
        return max(available_workers, key=get_capability_score)
        
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of completed task."""
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return task.result
            
        # Wait for task completion if still active
        if task_id in self.active_tasks:
            start_time = time.time()
            max_wait = timeout or 60.0
            
            while task_id in self.active_tasks:
                if time.time() - start_time > max_wait:
                    logger.warning(f"Task {task_id} timed out waiting for result")
                    return None
                    
                time.sleep(0.1)
                
            # Check if completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].result
                
        return None
        
    def update_worker_metrics(self, worker_id: str, metrics: Dict[str, float]):
        """Update worker performance metrics."""
        if worker_id in self.worker_pool:
            self.worker_pool[worker_id].performance_metrics.update(metrics)


class DistributedHDCWorker:
    """Distributed HDC computation worker."""
    
    def __init__(self, 
                 worker_id: str,
                 dimension: int = 10000,
                 capabilities: Optional[List[TaskType]] = None):
        self.worker_id = worker_id
        self.dimension = dimension
        self.capabilities = capabilities or list(TaskType)
        
        # Worker state
        self.status = WorkerStatus.IDLE
        self.current_task = None
        self.performance_metrics = {}
        
        # Task execution statistics
        self.execution_stats = {
            'total_tasks_executed': 0,
            'total_execution_time': 0.0,
            'tasks_by_type': {task_type: 0 for task_type in TaskType},
            'average_task_time': 0.0
        }
        
    def execute_task(self, task: HDCTask) -> Any:
        """Execute HDC task and return result."""
        try:
            self.status = WorkerStatus.BUSY
            self.current_task = task.task_id
            
            start_time = time.time()
            
            # Execute based on task type
            if task.task_type == TaskType.BUNDLE_OPERATION:
                result = self._execute_bundle_operation(task)
            elif task.task_type == TaskType.BIND_OPERATION:
                result = self._execute_bind_operation(task)
            elif task.task_type == TaskType.SIMILARITY_COMPUTATION:
                result = self._execute_similarity_computation(task)
            elif task.task_type == TaskType.MEMORY_QUERY:
                result = self._execute_memory_query(task)
            elif task.task_type == TaskType.BATCH_ENCODING:
                result = self._execute_batch_encoding(task)
            elif task.task_type == TaskType.LEARNING_UPDATE:
                result = self._execute_learning_update(task)
            elif task.task_type == TaskType.OPTIMIZATION:
                result = self._execute_optimization(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
            execution_time = time.time() - start_time
            
            # Update statistics
            self.execution_stats['total_tasks_executed'] += 1
            self.execution_stats['total_execution_time'] += execution_time
            self.execution_stats['tasks_by_type'][task.task_type] += 1
            
            if self.execution_stats['total_tasks_executed'] > 0:
                self.execution_stats['average_task_time'] = (
                    self.execution_stats['total_execution_time'] / 
                    self.execution_stats['total_tasks_executed']
                )
                
            # Update performance metrics
            task_performance_key = f"{task.task_type.value}_performance"
            if execution_time > 0:
                self.performance_metrics[task_performance_key] = 1.0 / execution_time
                
            self.status = WorkerStatus.IDLE
            self.current_task = None
            
            logger.debug(f"Worker {self.worker_id} completed task {task.task_id} in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.status = WorkerStatus.ERROR
            logger.error(f"Worker {self.worker_id} task execution failed: {e}")
            raise
            
    def _execute_bundle_operation(self, task: HDCTask) -> HyperVector:
        """Execute hypervector bundling operation."""
        vectors = task.data.get('vectors', [])
        weights = task.data.get('weights', None)
        
        if not vectors:
            raise ValueError("No vectors provided for bundling")
            
        if weights:
            # Weighted bundling
            weighted_vectors = list(zip(vectors, weights))
            return weighted_bundle(weighted_vectors)
        else:
            # Standard bundling
            return HyperVector.bundle_vectors(vectors)
            
    def _execute_bind_operation(self, task: HDCTask) -> HyperVector:
        """Execute hypervector binding operation."""
        vector_a = task.data.get('vector_a')
        vector_b = task.data.get('vector_b')
        
        if vector_a is None or vector_b is None:
            raise ValueError("Both vectors required for binding")
            
        return vector_a.bind(vector_b)
        
    def _execute_similarity_computation(self, task: HDCTask) -> float:
        """Execute similarity computation between vectors."""
        vector_a = task.data.get('vector_a')
        vector_b = task.data.get('vector_b')
        
        if vector_a is None or vector_b is None:
            raise ValueError("Both vectors required for similarity computation")
            
        return vector_a.similarity(vector_b)
        
    def _execute_memory_query(self, task: HDCTask) -> List[Dict[str, Any]]:
        """Execute memory query operation."""
        query_vector = task.data.get('query_vector')
        memory_vectors = task.data.get('memory_vectors', [])
        top_k = task.data.get('top_k', 10)
        threshold = task.data.get('threshold', 0.0)
        
        if query_vector is None:
            raise ValueError("Query vector required for memory search")
            
        # Compute similarities
        similarities = []
        for i, memory_vector in enumerate(memory_vectors):
            similarity = query_vector.similarity(memory_vector)
            if similarity >= threshold:
                similarities.append({
                    'index': i,
                    'vector': memory_vector,
                    'similarity': similarity
                })
                
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
        
    def _execute_batch_encoding(self, task: HDCTask) -> List[HyperVector]:
        """Execute batch encoding operation."""
        data_items = task.data.get('data_items', [])
        encoding_function = task.data.get('encoding_function')
        
        if not data_items:
            raise ValueError("No data items provided for batch encoding")
            
        if encoding_function is None:
            raise ValueError("Encoding function required for batch encoding")
            
        # Encode all items
        encoded_vectors = []
        for item in data_items:
            encoded_vector = encoding_function(item)
            encoded_vectors.append(encoded_vector)
            
        return encoded_vectors
        
    def _execute_learning_update(self, task: HDCTask) -> HyperVector:
        """Execute learning update operation."""
        current_vector = task.data.get('current_vector')
        update_vector = task.data.get('update_vector')
        learning_rate = task.data.get('learning_rate', 0.1)
        
        if current_vector is None or update_vector is None:
            raise ValueError("Both current and update vectors required")
            
        # Apply learning update
        updated_vector = weighted_bundle([
            (current_vector, 1.0 - learning_rate),
            (update_vector, learning_rate)
        ])
        
        return updated_vector
        
    def _execute_optimization(self, task: HDCTask) -> Dict[str, Any]:
        """Execute optimization operation."""
        objective_function = task.data.get('objective_function')
        initial_parameters = task.data.get('initial_parameters')
        optimization_steps = task.data.get('optimization_steps', 100)
        
        if objective_function is None or initial_parameters is None:
            raise ValueError("Objective function and initial parameters required")
            
        # Simple gradient-free optimization
        best_params = initial_parameters.copy()
        best_score = objective_function(best_params)
        
        for step in range(optimization_steps):
            # Random perturbation
            perturbation = np.random.normal(0, 0.1, size=len(initial_parameters))
            candidate_params = best_params + perturbation
            
            candidate_score = objective_function(candidate_params)
            
            if candidate_score > best_score:
                best_params = candidate_params
                best_score = candidate_score
                
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_steps': optimization_steps
        }


class DistributedHDCCluster:
    """Distributed HDC processing cluster manager."""
    
    def __init__(self, 
                 cluster_size: int = mp.cpu_count(),
                 dimension: int = 10000,
                 enable_redis_cache: bool = True):
        self.cluster_size = cluster_size
        self.dimension = dimension
        self.enable_redis_cache = enable_redis_cache
        
        # Cluster components
        self.scheduler = HDCTaskScheduler(cluster_size)
        self.workers = {}
        self.worker_processes = {}
        
        # Communication infrastructure
        self.zmq_context = zmq.Context()
        self.task_sender = None
        self.result_receiver = None
        
        # Redis cache for shared state
        self.redis_client = None
        if enable_redis_cache:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache enabled")
            except Exception as e:
                logger.warning(f"Redis cache disabled: {e}")
                self.redis_client = None
                
        # Cluster statistics
        self.cluster_stats = {
            'total_tasks_processed': 0,
            'cluster_uptime': time.time(),
            'average_cluster_utilization': 0.0,
            'peak_throughput': 0.0,
            'cache_hit_rate': 0.0
        }
        
    def start_cluster(self):
        """Start the distributed HDC cluster."""
        try:
            logger.info(f"Starting HDC cluster with {self.cluster_size} workers")
            
            # Initialize communication infrastructure
            self._setup_communication()
            
            # Start worker processes
            for i in range(self.cluster_size):
                worker_id = f"worker_{i}"
                self._start_worker(worker_id)
                
            # Start cluster management thread
            self.management_thread = threading.Thread(
                target=self._cluster_management_loop,
                daemon=True
            )
            self.management_thread.start()
            
            logger.info("HDC cluster started successfully")
            
        except Exception as e:
            logger.error(f"Cluster startup failed: {e}")
            raise
            
    def stop_cluster(self):
        """Stop the distributed HDC cluster."""
        try:
            logger.info("Stopping HDC cluster")
            
            # Stop worker processes
            for worker_id, process in self.worker_processes.items():
                process.terminate()
                process.join(timeout=5.0)
                
            # Close communication infrastructure
            if self.task_sender:
                self.task_sender.close()
            if self.result_receiver:
                self.result_receiver.close()
                
            self.zmq_context.term()
            
            logger.info("HDC cluster stopped")
            
        except Exception as e:
            logger.error(f"Cluster shutdown failed: {e}")
            
    def _setup_communication(self):
        """Setup ZeroMQ communication infrastructure."""
        # Task distribution socket
        self.task_sender = self.zmq_context.socket(zmq.PUSH)
        self.task_sender.bind("tcp://*:5555")
        
        # Result collection socket
        self.result_receiver = self.zmq_context.socket(zmq.PULL)
        self.result_receiver.bind("tcp://*:5556")
        
    def _start_worker(self, worker_id: str):
        """Start individual worker process."""
        try:
            worker = DistributedHDCWorker(worker_id, self.dimension)
            
            # Create worker process
            process = mp.Process(
                target=self._worker_process_main,
                args=(worker_id, self.dimension)
            )
            process.start()
            
            self.workers[worker_id] = worker
            self.worker_processes[worker_id] = process
            
            # Register worker in scheduler
            worker_node = WorkerNode(
                worker_id=worker_id,
                address="localhost",
                port=5555,
                status=WorkerStatus.IDLE,
                capabilities=list(TaskType)
            )
            self.scheduler.worker_pool[worker_id] = worker_node
            
            logger.debug(f"Started worker: {worker_id}")
            
        except Exception as e:
            logger.error(f"Worker startup failed: {e}")
            
    def _worker_process_main(self, worker_id: str, dimension: int):
        """Main function for worker process."""
        try:
            # Setup worker communication
            context = zmq.Context()
            task_receiver = context.socket(zmq.PULL)
            task_receiver.connect("tcp://localhost:5555")
            
            result_sender = context.socket(zmq.PUSH)
            result_sender.connect("tcp://localhost:5556")
            
            # Create worker instance
            worker = DistributedHDCWorker(worker_id, dimension)
            
            logger.info(f"Worker {worker_id} started")
            
            while True:
                try:
                    # Receive task
                    message = task_receiver.recv_pyobj(zmq.NOBLOCK)
                    task = pickle.loads(message)
                    
                    # Execute task
                    result = worker.execute_task(task)
                    
                    # Send result
                    result_message = {
                        'task_id': task.task_id,
                        'worker_id': worker_id,
                        'result': result,
                        'execution_time': time.time() - task.created_at,
                        'success': True
                    }
                    result_sender.send_pyobj(pickle.dumps(result_message))
                    
                except zmq.Again:
                    # No task available, continue
                    time.sleep(0.01)
                    continue
                    
                except Exception as e:
                    # Send error result
                    error_message = {
                        'task_id': task.task_id if 'task' in locals() else 'unknown',
                        'worker_id': worker_id,
                        'result': None,
                        'error': str(e),
                        'success': False
                    }
                    result_sender.send_pyobj(pickle.dumps(error_message))
                    
        except Exception as e:
            logger.error(f"Worker {worker_id} process failed: {e}")
            
    def _cluster_management_loop(self):
        """Main cluster management loop."""
        logger.info("Starting cluster management loop")
        
        while True:
            try:
                # Process task queue
                self._process_task_queue()
                
                # Collect results
                self._collect_results()
                
                # Update cluster statistics
                self._update_cluster_statistics()
                
                # Health monitoring
                self._monitor_worker_health()
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Cluster management error: {e}")
                time.sleep(1.0)
                
    def _process_task_queue(self):
        """Process pending tasks in the queue."""
        try:
            while not self.scheduler.task_queue.empty():
                # Get next task
                priority, task_id, task = self.scheduler.task_queue.get_nowait()
                
                # Select worker
                selected_worker = self.scheduler.select_worker(task)
                
                if selected_worker:
                    # Assign task to worker
                    task.assigned_worker = selected_worker
                    self.scheduler.worker_pool[selected_worker].status = WorkerStatus.BUSY
                    self.scheduler.worker_pool[selected_worker].current_task = task_id
                    
                    # Send task to worker
                    task_message = pickle.dumps(task)
                    self.task_sender.send_pyobj(task_message, zmq.NOBLOCK)
                    
                    logger.debug(f"Assigned task {task_id} to worker {selected_worker}")
                    
                else:
                    # No available worker, put task back in queue
                    self.scheduler.task_queue.put((priority, task_id, task))
                    break
                    
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            
    def _collect_results(self):
        """Collect results from worker processes."""
        try:
            while True:
                try:
                    # Receive result
                    result_message = self.result_receiver.recv_pyobj(zmq.NOBLOCK)
                    result_data = pickle.loads(result_message)
                    
                    task_id = result_data['task_id']
                    worker_id = result_data['worker_id']
                    
                    # Update task with result
                    if task_id in self.scheduler.active_tasks:
                        task = self.scheduler.active_tasks[task_id]
                        
                        if result_data['success']:
                            task.result = result_data['result']
                        else:
                            task.error = result_data.get('error')
                            
                        # Move to completed tasks
                        self.scheduler.completed_tasks[task_id] = task
                        del self.scheduler.active_tasks[task_id]
                        
                        # Update worker status
                        if worker_id in self.scheduler.worker_pool:
                            worker_node = self.scheduler.worker_pool[worker_id]
                            worker_node.status = WorkerStatus.IDLE
                            worker_node.current_task = None
                            worker_node.last_heartbeat = time.time()
                            
                        self.cluster_stats['total_tasks_processed'] += 1
                        
                        if result_data['success']:
                            self.scheduler.scheduling_stats['total_tasks_completed'] += 1
                        else:
                            self.scheduler.scheduling_stats['total_tasks_failed'] += 1
                            
                except zmq.Again:
                    break
                    
        except Exception as e:
            logger.error(f"Result collection failed: {e}")
            
    def _update_cluster_statistics(self):
        """Update cluster performance statistics."""
        try:
            current_time = time.time()
            
            # Calculate cluster utilization
            busy_workers = sum(1 for worker in self.scheduler.worker_pool.values() 
                             if worker.status == WorkerStatus.BUSY)
            total_workers = len(self.scheduler.worker_pool)
            
            if total_workers > 0:
                current_utilization = busy_workers / total_workers
                self.cluster_stats['average_cluster_utilization'] = (
                    (self.cluster_stats['average_cluster_utilization'] * 0.9) +
                    (current_utilization * 0.1)
                )
                
            # Calculate throughput
            uptime = current_time - self.cluster_stats['cluster_uptime']
            if uptime > 0:
                current_throughput = self.cluster_stats['total_tasks_processed'] / uptime
                self.cluster_stats['peak_throughput'] = max(
                    self.cluster_stats['peak_throughput'],
                    current_throughput
                )
                
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")
            
    def _monitor_worker_health(self):
        """Monitor worker health and restart failed workers."""
        try:
            current_time = time.time()
            
            for worker_id, worker_node in self.scheduler.worker_pool.items():
                # Check for unresponsive workers
                if current_time - worker_node.last_heartbeat > 60.0:  # 1 minute timeout
                    logger.warning(f"Worker {worker_id} appears unresponsive")
                    worker_node.status = WorkerStatus.OFFLINE
                    
                    # Restart worker process if needed
                    if worker_id in self.worker_processes:
                        process = self.worker_processes[worker_id]
                        if not process.is_alive():
                            logger.info(f"Restarting failed worker: {worker_id}")
                            self._start_worker(worker_id)
                            
        except Exception as e:
            logger.error(f"Worker health monitoring failed: {e}")
            
    def submit_bundle_task(self, vectors: List[HyperVector], weights: Optional[List[float]] = None) -> str:
        """Submit bundle operation task."""
        task = HDCTask(
            task_id=f"bundle_{int(time.time() * 1000000)}",
            task_type=TaskType.BUNDLE_OPERATION,
            data={
                'vectors': vectors,
                'weights': weights
            }
        )
        
        return self.scheduler.submit_task(task)
        
    def submit_similarity_task(self, vector_a: HyperVector, vector_b: HyperVector) -> str:
        """Submit similarity computation task."""
        task = HDCTask(
            task_id=f"similarity_{int(time.time() * 1000000)}",
            task_type=TaskType.SIMILARITY_COMPUTATION,
            data={
                'vector_a': vector_a,
                'vector_b': vector_b
            }
        )
        
        return self.scheduler.submit_task(task)
        
    def submit_memory_query_task(self, 
                                query_vector: HyperVector,
                                memory_vectors: List[HyperVector],
                                top_k: int = 10,
                                threshold: float = 0.0) -> str:
        """Submit memory query task."""
        task = HDCTask(
            task_id=f"query_{int(time.time() * 1000000)}",
            task_type=TaskType.MEMORY_QUERY,
            data={
                'query_vector': query_vector,
                'memory_vectors': memory_vectors,
                'top_k': top_k,
                'threshold': threshold
            }
        )
        
        return self.scheduler.submit_task(task)
        
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of submitted task."""
        return self.scheduler.get_task_result(task_id, timeout)
        
    async def submit_bundle_task_async(self, vectors: List[HyperVector], weights: Optional[List[float]] = None) -> HyperVector:
        """Submit bundle task asynchronously."""
        task_id = self.submit_bundle_task(vectors, weights)
        
        # Wait for result asynchronously
        loop = asyncio.get_event_loop()
        
        def get_result():
            return self.get_task_result(task_id, timeout=30.0)
            
        result = await loop.run_in_executor(None, get_result)
        
        if result is None:
            raise TimeoutError(f"Task {task_id} timed out")
            
        return result
        
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        stats = self.cluster_stats.copy()
        
        # Add scheduler statistics
        stats['scheduler_stats'] = self.scheduler.scheduling_stats
        
        # Add worker statistics
        worker_stats = {}
        for worker_id, worker_node in self.scheduler.worker_pool.items():
            worker_stats[worker_id] = {
                'status': worker_node.status.value,
                'current_task': worker_node.current_task,
                'last_heartbeat': worker_node.last_heartbeat,
                'performance_metrics': worker_node.performance_metrics
            }
        stats['worker_stats'] = worker_stats
        
        # Add cache statistics if Redis is enabled
        if self.redis_client:
            try:
                cache_info = self.redis_client.info('stats')
                cache_hits = cache_info.get('keyspace_hits', 0)
                cache_misses = cache_info.get('keyspace_misses', 0)
                
                if cache_hits + cache_misses > 0:
                    stats['cache_hit_rate'] = cache_hits / (cache_hits + cache_misses)
                    
            except Exception as e:
                logger.warning(f"Failed to get cache statistics: {e}")
                
        return stats
        
    def optimize_cluster_performance(self):
        """Optimize cluster performance based on current metrics."""
        try:
            stats = self.get_cluster_statistics()
            
            # Adjust load balancing strategy based on performance
            current_utilization = stats['average_cluster_utilization']
            
            if current_utilization < 0.3:
                # Low utilization - use round robin for simplicity
                self.scheduler.load_balancing_strategy = "round_robin"
            elif current_utilization > 0.8:
                # High utilization - use least loaded for better distribution
                self.scheduler.load_balancing_strategy = "least_loaded"
            else:
                # Medium utilization - use capability-based for efficiency
                self.scheduler.load_balancing_strategy = "capability_based"
                
            logger.info(f"Load balancing strategy updated to: {self.scheduler.load_balancing_strategy}")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")


# Utility functions for distributed HDC operations
def create_distributed_cluster(cluster_size: int = None, dimension: int = 10000) -> DistributedHDCCluster:
    """Create and start distributed HDC cluster."""
    if cluster_size is None:
        cluster_size = min(mp.cpu_count(), 8)  # Reasonable default
        
    cluster = DistributedHDCCluster(cluster_size, dimension)
    cluster.start_cluster()
    
    return cluster


async def distributed_bundle_operation(cluster: DistributedHDCCluster,
                                     vectors: List[HyperVector],
                                     weights: Optional[List[float]] = None) -> HyperVector:
    """Perform distributed bundle operation."""
    return await cluster.submit_bundle_task_async(vectors, weights)


def batch_similarity_computation(cluster: DistributedHDCCluster,
                                query_vectors: List[HyperVector],
                                reference_vectors: List[HyperVector]) -> List[List[float]]:
    """Compute similarities between all query and reference vectors."""
    task_ids = []
    
    # Submit all similarity tasks
    for query_vector in query_vectors:
        for reference_vector in reference_vectors:
            task_id = cluster.submit_similarity_task(query_vector, reference_vector)
            task_ids.append(task_id)
            
    # Collect all results
    results = []
    for i, query_vector in enumerate(query_vectors):
        query_results = []
        for j, reference_vector in enumerate(reference_vectors):
            task_index = i * len(reference_vectors) + j
            task_id = task_ids[task_index]
            
            result = cluster.get_task_result(task_id, timeout=30.0)
            query_results.append(result if result is not None else 0.0)
            
        results.append(query_results)
        
    return results