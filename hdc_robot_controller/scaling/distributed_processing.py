"""
Distributed processing module for HDC Robot Controller.
Enables horizontal scaling across multiple nodes for large-scale operations.
"""

import json
import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
import pickle
import logging
import hashlib
import queue
from enum import Enum
import redis
import numpy as np

from ..core.hypervector import HyperVector
from ..core.error_handling import HDCException, robust_hdc_operation
from ..core.security import SecurityManager, get_security_manager


class NodeRole(Enum):
    """Roles for distributed nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    role: NodeRole
    address: str
    port: int
    capabilities: Dict[str, Any]
    status: str
    last_heartbeat: float
    workload: float = 0.0


@dataclass
class DistributedTask:
    """Distributed computation task."""
    task_id: str
    task_type: str
    data: Any
    status: TaskStatus
    assigned_node: Optional[str] = None
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    result: Any = None
    error: Optional[str] = None


@dataclass
class WorkloadDistribution:
    """Workload distribution strategy."""
    strategy: str  # "round_robin", "load_balanced", "capability_based"
    parameters: Dict[str, Any]


class DistributedCoordinator:
    """Coordinates distributed HDC processing across multiple nodes."""
    
    def __init__(self, node_id: str, redis_host: str = "localhost", redis_port: int = 6379):
        self.node_id = node_id
        self.role = NodeRole.COORDINATOR
        self.logger = logging.getLogger(__name__)
        
        # Redis for coordination and communication
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Node management
        self.active_nodes: Dict[str, NodeInfo] = {}
        self.task_queue = queue.Queue()
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.running_tasks: Dict[str, DistributedTask] = {}
        
        # Configuration
        self.heartbeat_interval = 30.0  # seconds
        self.task_timeout = 300.0  # 5 minutes
        self.max_retries = 3
        
        # Threading
        self.running = False
        self.threads = []
        
        # Security
        self.security_manager = get_security_manager()
        
        # Start coordinator services
        self.start_services()
    
    def start_services(self):
        """Start coordinator background services."""
        self.running = True
        
        # Start heartbeat monitor
        heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Start task scheduler
        scheduler_thread = threading.Thread(target=self._task_scheduler, daemon=True)
        scheduler_thread.start()
        self.threads.append(scheduler_thread)
        
        # Start health monitor
        health_thread = threading.Thread(target=self._health_monitor, daemon=True)
        health_thread.start()
        self.threads.append(health_thread)
        
        self.logger.info("Coordinator services started")
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new worker node."""
        try:
            # Validate node info
            if not node_info.node_id or not node_info.address:
                return False
            
            # Store node information
            self.active_nodes[node_info.node_id] = node_info
            
            # Store in Redis for persistence
            node_key = f"nodes:{node_info.node_id}"
            self.redis_client.hset(node_key, mapping=asdict(node_info))
            
            self.logger.info(f"Registered node {node_info.node_id} ({node_info.role.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_info.node_id}: {e}")
            return False
    
    def submit_task(self, task_type: str, data: Any, 
                   priority: int = 1, session_token: str = None) -> str:
        """Submit a distributed task."""
        # Security check
        if session_token and not self.security_manager.check_access(session_token, "write_control"):
            raise HDCException("Access denied for task submission")
        
        task_id = self._generate_task_id()
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            status=TaskStatus.PENDING,
            created_at=time.time()
        )
        
        # Store task
        self.task_queue.put((priority, task))
        
        # Store in Redis
        task_key = f"tasks:{task_id}"
        self.redis_client.hset(task_key, mapping={
            "task_id": task_id,
            "task_type": task_type,
            "status": task.status.value,
            "created_at": task.created_at,
            "data": pickle.dumps(data).hex()
        })
        
        self.logger.info(f"Submitted task {task_id} of type {task_type}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get status of a distributed task."""
        # Check running tasks
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check Redis
        task_key = f"tasks:{task_id}"
        task_data = self.redis_client.hgetall(task_key)
        
        if task_data:
            return DistributedTask(
                task_id=task_data["task_id"],
                task_type=task_data["task_type"],
                data=pickle.loads(bytes.fromhex(task_data["data"])),
                status=TaskStatus(task_data["status"]),
                created_at=float(task_data["created_at"]),
                assigned_node=task_data.get("assigned_node"),
                result=pickle.loads(bytes.fromhex(task_data["result"])) if task_data.get("result") else None
            )
        
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        active_nodes = len([n for n in self.active_nodes.values() 
                           if time.time() - n.last_heartbeat < 60])
        
        queue_size = self.task_queue.qsize()
        running_count = len(self.running_tasks)
        completed_count = len(self.completed_tasks)
        
        total_workload = sum(n.workload for n in self.active_nodes.values())
        avg_workload = total_workload / max(1, active_nodes)
        
        return {
            "coordinator_id": self.node_id,
            "active_nodes": active_nodes,
            "total_nodes": len(self.active_nodes),
            "queue_size": queue_size,
            "running_tasks": running_count,
            "completed_tasks": completed_count,
            "average_workload": avg_workload,
            "cluster_health": self._calculate_cluster_health()
        }
    
    def distribute_hdc_operation(self, operation: str, vectors: List[HyperVector],
                               chunk_size: int = 100) -> List[Any]:
        """Distribute HDC operation across worker nodes."""
        if not vectors:
            return []
        
        # Split vectors into chunks
        chunks = [vectors[i:i+chunk_size] for i in range(0, len(vectors), chunk_size)]
        
        # Submit tasks for each chunk
        task_ids = []
        for i, chunk in enumerate(chunks):
            task_data = {
                "operation": operation,
                "vectors": [{"dimension": v.dimension, "data": v.data.tolist()} for v in chunk],
                "chunk_id": i
            }
            
            task_id = self.submit_task("hdc_operation", task_data)
            task_ids.append(task_id)
        
        # Wait for completion and collect results
        results = []
        max_wait_time = len(chunks) * 30  # 30 seconds per chunk
        start_time = time.time()
        
        while len(results) < len(task_ids) and (time.time() - start_time) < max_wait_time:
            for task_id in task_ids:
                if task_id not in [r[0] for r in results]:  # Not yet collected
                    task = self.get_task_status(task_id)
                    if task and task.status == TaskStatus.COMPLETED:
                        results.append((task_id, task.result))
            
            time.sleep(0.1)  # Brief pause
        
        # Sort results by chunk_id and return
        sorted_results = sorted(results, key=lambda x: x[1].get("chunk_id", 0))
        return [r[1]["result"] for r in sorted_results]
    
    def _heartbeat_monitor(self):
        """Monitor node heartbeats."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for dead nodes
                dead_nodes = []
                for node_id, node_info in self.active_nodes.items():
                    if current_time - node_info.last_heartbeat > 60:  # 1 minute timeout
                        dead_nodes.append(node_id)
                
                # Remove dead nodes
                for node_id in dead_nodes:
                    self.logger.warning(f"Node {node_id} appears dead, removing")
                    del self.active_nodes[node_id]
                    self.redis_client.delete(f"nodes:{node_id}")
                
                # Update heartbeat info in Redis
                for node_id, node_info in self.active_nodes.items():
                    heartbeat_key = f"heartbeat:{node_id}"
                    heartbeat_data = self.redis_client.get(heartbeat_key)
                    
                    if heartbeat_data:
                        heartbeat_info = json.loads(heartbeat_data)
                        node_info.last_heartbeat = heartbeat_info.get("timestamp", 0)
                        node_info.workload = heartbeat_info.get("workload", 0.0)
                        node_info.status = heartbeat_info.get("status", "unknown")
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                time.sleep(5)
    
    def _task_scheduler(self):
        """Schedule tasks to available workers."""
        while self.running:
            try:
                # Get next task from queue (blocking with timeout)
                priority, task = self.task_queue.get(timeout=1.0)
                
                # Find available worker
                worker_node = self._select_worker_node(task)
                
                if worker_node:
                    # Assign task to worker
                    task.assigned_node = worker_node.node_id
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                    
                    # Move to running tasks
                    self.running_tasks[task.task_id] = task
                    
                    # Send task to worker
                    self._send_task_to_worker(task, worker_node)
                    
                else:
                    # No available workers, put task back in queue
                    self.task_queue.put((priority, task))
                    time.sleep(1)  # Wait before retrying
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {e}")
    
    def _health_monitor(self):
        """Monitor cluster health and performance."""
        while self.running:
            try:
                # Collect health metrics
                health_data = {
                    "timestamp": time.time(),
                    "cluster_status": self.get_cluster_status(),
                    "node_details": {node_id: asdict(node) for node_id, node in self.active_nodes.items()}
                }
                
                # Store health data
                health_key = f"cluster_health:{int(time.time())}"
                self.redis_client.setex(health_key, 3600, json.dumps(health_data))  # Keep for 1 hour
                
                # Clean up old health data
                current_time = int(time.time())
                old_keys = []
                for key in self.redis_client.keys("cluster_health:*"):
                    key_time = int(key.split(":")[1])
                    if current_time - key_time > 3600:  # Older than 1 hour
                        old_keys.append(key)
                
                if old_keys:
                    self.redis_client.delete(*old_keys)
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                time.sleep(10)
    
    def _select_worker_node(self, task: DistributedTask) -> Optional[NodeInfo]:
        """Select best worker node for task."""
        available_workers = [
            node for node in self.active_nodes.values()
            if node.role == NodeRole.WORKER and 
               node.status == "healthy" and
               time.time() - node.last_heartbeat < 30
        ]
        
        if not available_workers:
            return None
        
        # Load balancing strategy
        if task.task_type == "hdc_operation":
            # For HDC operations, prefer nodes with lower workload
            return min(available_workers, key=lambda n: n.workload)
        else:
            # Default: round-robin
            return available_workers[len(self.running_tasks) % len(available_workers)]
    
    def _send_task_to_worker(self, task: DistributedTask, worker_node: NodeInfo):
        """Send task to worker node."""
        try:
            # Store task in Redis for worker to pick up
            work_key = f"work:{worker_node.node_id}:{task.task_id}"
            work_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "data": pickle.dumps(task.data).hex(),
                "assigned_at": time.time()
            }
            
            self.redis_client.hset(work_key, mapping=work_data)
            self.redis_client.expire(work_key, int(self.task_timeout))
            
            # Notify worker through pub/sub
            self.redis_client.publish(f"work_channel:{worker_node.node_id}", task.task_id)
            
            self.logger.info(f"Sent task {task.task_id} to worker {worker_node.node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send task to worker: {e}")
            # Mark task as failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._complete_task(task)
    
    def _complete_task(self, task: DistributedTask):
        """Mark task as completed."""
        # Move from running to completed
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        
        task.completed_at = time.time()
        self.completed_tasks[task.task_id] = task
        
        # Update in Redis
        task_key = f"tasks:{task.task_id}"
        update_data = {
            "status": task.status.value,
            "completed_at": task.completed_at
        }
        
        if task.result is not None:
            update_data["result"] = pickle.dumps(task.result).hex()
        
        if task.error:
            update_data["error"] = task.error
        
        self.redis_client.hset(task_key, mapping=update_data)
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        timestamp = str(int(time.time() * 1000000))  # microsecond precision
        node_hash = hashlib.md5(self.node_id.encode()).hexdigest()[:8]
        return f"task_{timestamp}_{node_hash}"
    
    def _calculate_cluster_health(self) -> float:
        """Calculate overall cluster health score."""
        if not self.active_nodes:
            return 0.0
        
        health_scores = []
        current_time = time.time()
        
        for node in self.active_nodes.values():
            # Time since last heartbeat
            heartbeat_age = current_time - node.last_heartbeat
            heartbeat_score = max(0, 1.0 - (heartbeat_age / 60.0))  # 60 second max
            
            # Workload score (inverted - lower workload is better)
            workload_score = max(0, 1.0 - (node.workload / 100.0))  # Assume 100% is max workload
            
            # Status score
            status_score = 1.0 if node.status == "healthy" else 0.5 if node.status == "warning" else 0.0
            
            # Combined score
            node_health = (heartbeat_score + workload_score + status_score) / 3.0
            health_scores.append(node_health)
        
        return sum(health_scores) / len(health_scores)
    
    def shutdown(self):
        """Shutdown coordinator."""
        self.logger.info("Shutting down coordinator...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Close Redis connection
        if hasattr(self, 'redis_client'):
            self.redis_client.close()


class DistributedWorker:
    """Worker node for distributed HDC processing."""
    
    def __init__(self, node_id: str, coordinator_address: str, 
                 redis_host: str = "localhost", redis_port: int = 6379):
        self.node_id = node_id
        self.coordinator_address = coordinator_address
        self.logger = logging.getLogger(__name__)
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Worker state
        self.running = False
        self.current_workload = 0.0
        self.status = "initializing"
        self.capabilities = {
            "hdc_operations": True,
            "max_dimension": 50000,
            "parallel_processing": True,
            "gpu_acceleration": False  # Could be detected
        }
        
        # Task processing
        self.task_processors = {
            "hdc_operation": self._process_hdc_operation
        }
        
        # Threading
        self.threads = []
        
        # Register with coordinator
        self._register_with_coordinator()
        
        # Start worker services
        self.start_services()
    
    def start_services(self):
        """Start worker background services."""
        self.running = True
        self.status = "healthy"
        
        # Start work listener
        work_thread = threading.Thread(target=self._work_listener, daemon=True)
        work_thread.start()
        self.threads.append(work_thread)
        
        # Start heartbeat sender
        heartbeat_thread = threading.Thread(target=self._heartbeat_sender, daemon=True)
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        self.logger.info("Worker services started")
    
    def _register_with_coordinator(self):
        """Register this worker with the coordinator."""
        node_info = NodeInfo(
            node_id=self.node_id,
            role=NodeRole.WORKER,
            address=socket.gethostname(),
            port=0,  # Not using direct socket communication
            capabilities=self.capabilities,
            status="healthy",
            last_heartbeat=time.time(),
            workload=0.0
        )
        
        # Store registration in Redis
        node_key = f"nodes:{self.node_id}"
        self.redis_client.hset(node_key, mapping=asdict(node_info))
        
        self.logger.info(f"Registered worker {self.node_id}")
    
    def _work_listener(self):
        """Listen for work assignments."""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(f"work_channel:{self.node_id}")
        
        self.logger.info(f"Listening for work on channel work_channel:{self.node_id}")
        
        for message in pubsub.listen():
            if not self.running:
                break
            
            if message['type'] == 'message':
                try:
                    task_id = message['data']
                    self._process_work(task_id)
                except Exception as e:
                    self.logger.error(f"Error processing work message: {e}")
    
    def _process_work(self, task_id: str):
        """Process assigned work."""
        work_key = f"work:{self.node_id}:{task_id}"
        
        try:
            # Get work data
            work_data = self.redis_client.hgetall(work_key)
            if not work_data:
                self.logger.warning(f"No work data found for task {task_id}")
                return
            
            # Parse work data
            task_type = work_data["task_type"]
            task_data = pickle.loads(bytes.fromhex(work_data["data"]))
            
            self.logger.info(f"Processing task {task_id} of type {task_type}")
            
            # Update workload
            self.current_workload = 50.0  # Simplified workload tracking
            
            # Process task
            processor = self.task_processors.get(task_type)
            if processor:
                result = processor(task_data)
                
                # Store result
                result_key = f"results:{task_id}"
                result_data = {
                    "task_id": task_id,
                    "status": "completed",
                    "result": pickle.dumps(result).hex(),
                    "completed_at": time.time(),
                    "worker_id": self.node_id
                }
                
                self.redis_client.hset(result_key, mapping=result_data)
                self.redis_client.expire(result_key, 3600)  # Keep for 1 hour
                
                # Notify coordinator
                self.redis_client.publish("results_channel", task_id)
                
                self.logger.info(f"Completed task {task_id}")
            else:
                self.logger.error(f"No processor for task type {task_type}")
            
            # Clean up work data
            self.redis_client.delete(work_key)
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            
            # Report error
            error_key = f"results:{task_id}"
            error_data = {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "completed_at": time.time(),
                "worker_id": self.node_id
            }
            
            self.redis_client.hset(error_key, mapping=error_data)
            self.redis_client.publish("results_channel", task_id)
        
        finally:
            # Reset workload
            self.current_workload = 0.0
    
    def _process_hdc_operation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process HDC operation task."""
        operation = task_data["operation"]
        vector_data = task_data["vectors"]
        chunk_id = task_data.get("chunk_id", 0)
        
        # Reconstruct HyperVectors
        vectors = []
        for v_data in vector_data:
            dimension = v_data["dimension"]
            data = np.array(v_data["data"], dtype=np.int8)
            vectors.append(HyperVector(dimension, data))
        
        # Process based on operation type
        if operation == "bundle":
            result_vector = HyperVector.bundle_vectors(vectors)
            result = {
                "chunk_id": chunk_id,
                "result": {
                    "dimension": result_vector.dimension,
                    "data": result_vector.data.tolist()
                }
            }
        elif operation == "similarities":
            if len(vectors) >= 2:
                query = vectors[0]
                database = vectors[1:]
                similarities = [query.similarity(v) for v in database]
                result = {
                    "chunk_id": chunk_id,
                    "result": similarities
                }
            else:
                result = {"chunk_id": chunk_id, "result": []}
        elif operation == "batch_invert":
            inverted = [v.invert() for v in vectors]
            result = {
                "chunk_id": chunk_id,
                "result": [{"dimension": v.dimension, "data": v.data.tolist()} for v in inverted]
            }
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return result
    
    def _heartbeat_sender(self):
        """Send periodic heartbeats to coordinator."""
        while self.running:
            try:
                heartbeat_data = {
                    "timestamp": time.time(),
                    "status": self.status,
                    "workload": self.current_workload,
                    "node_id": self.node_id
                }
                
                heartbeat_key = f"heartbeat:{self.node_id}"
                self.redis_client.setex(heartbeat_key, 60, json.dumps(heartbeat_data))
                
                time.sleep(30)  # Send every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error sending heartbeat: {e}")
                time.sleep(5)
    
    def shutdown(self):
        """Shutdown worker."""
        self.logger.info("Shutting down worker...")
        self.running = False
        self.status = "shutting_down"
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Unregister from coordinator
        node_key = f"nodes:{self.node_id}"
        self.redis_client.delete(node_key)
        
        # Close Redis connection
        if hasattr(self, 'redis_client'):
            self.redis_client.close()


class ClusterManager:
    """High-level manager for distributed HDC cluster."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.coordinator = None
        self.workers = []
        self.logger = logging.getLogger(__name__)
        self.redis_host = redis_host
        self.redis_port = redis_port
    
    def start_coordinator(self, node_id: str = "coordinator_0") -> DistributedCoordinator:
        """Start cluster coordinator."""
        if self.coordinator:
            self.logger.warning("Coordinator already running")
            return self.coordinator
        
        self.coordinator = DistributedCoordinator(node_id, self.redis_host, self.redis_port)
        self.logger.info(f"Started coordinator {node_id}")
        return self.coordinator
    
    def start_worker(self, node_id: str = None) -> DistributedWorker:
        """Start a worker node."""
        if not node_id:
            node_id = f"worker_{len(self.workers)}"
        
        if not self.coordinator:
            raise RuntimeError("Coordinator must be started before workers")
        
        coordinator_address = f"{self.redis_host}:{self.redis_port}"
        worker = DistributedWorker(node_id, coordinator_address, self.redis_host, self.redis_port)
        self.workers.append(worker)
        
        self.logger.info(f"Started worker {node_id}")
        return worker
    
    def scale_cluster(self, target_workers: int):
        """Scale cluster to target number of workers."""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Add workers
            for i in range(current_workers, target_workers):
                self.start_worker(f"worker_{i}")
        elif target_workers < current_workers:
            # Remove workers
            for i in range(target_workers, current_workers):
                if i < len(self.workers):
                    self.workers[i].shutdown()
            
            self.workers = self.workers[:target_workers]
        
        self.logger.info(f"Scaled cluster to {target_workers} workers")
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cluster metrics."""
        if not self.coordinator:
            return {"error": "No coordinator running"}
        
        return self.coordinator.get_cluster_status()
    
    def shutdown_cluster(self):
        """Shutdown entire cluster."""
        self.logger.info("Shutting down cluster...")
        
        # Shutdown workers
        for worker in self.workers:
            worker.shutdown()
        
        # Shutdown coordinator
        if self.coordinator:
            self.coordinator.shutdown()
        
        self.workers.clear()
        self.coordinator = None


# Distributed HDC operations
class DistributedHDCOperations:
    """High-level interface for distributed HDC operations."""
    
    def __init__(self, coordinator: DistributedCoordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger(__name__)
    
    @robust_hdc_operation(max_retries=3)
    def distributed_bundle(self, vectors: List[HyperVector], 
                          chunk_size: int = 1000) -> HyperVector:
        """Bundle vectors across distributed nodes."""
        if not vectors:
            return HyperVector.zero()
        
        if len(vectors) <= chunk_size:
            # Small enough for single node
            task_id = self.coordinator.submit_task("hdc_operation", {
                "operation": "bundle",
                "vectors": [{"dimension": v.dimension, "data": v.data.tolist()} for v in vectors],
                "chunk_id": 0
            })
            
            # Wait for result
            result = self._wait_for_task(task_id)
            return self._reconstruct_hypervector(result["result"])
        
        # Large dataset - distribute across nodes
        results = self.coordinator.distribute_hdc_operation("bundle", vectors, chunk_size)
        
        # Bundle the chunk results
        chunk_vectors = [self._reconstruct_hypervector(r["result"]) for r in results]
        return HyperVector.bundle_vectors(chunk_vectors)
    
    @robust_hdc_operation(max_retries=3)
    def distributed_similarities(self, query: HyperVector, 
                                database: List[HyperVector],
                                chunk_size: int = 1000) -> List[float]:
        """Compute similarities across distributed nodes."""
        if not database:
            return []
        
        # Prepare chunks with query vector
        chunks = []
        for i in range(0, len(database), chunk_size):
            chunk = [query] + database[i:i+chunk_size]
            chunks.append(chunk)
        
        # Submit tasks
        task_ids = []
        for i, chunk in enumerate(chunks):
            task_data = {
                "operation": "similarities",
                "vectors": [{"dimension": v.dimension, "data": v.data.tolist()} for v in chunk],
                "chunk_id": i
            }
            task_id = self.coordinator.submit_task("hdc_operation", task_data)
            task_ids.append(task_id)
        
        # Collect results
        all_similarities = []
        for task_id in task_ids:
            result = self._wait_for_task(task_id)
            all_similarities.extend(result["result"])
        
        return all_similarities
    
    def _wait_for_task(self, task_id: str, timeout: float = 60.0) -> Any:
        """Wait for task completion and return result."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.coordinator.get_task_status(task_id)
            
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
            elif task and task.status == TaskStatus.FAILED:
                raise HDCException(f"Distributed task failed: {task.error}")
            
            time.sleep(0.1)
        
        raise HDCException(f"Distributed task {task_id} timed out")
    
    def _reconstruct_hypervector(self, data: Dict[str, Any]) -> HyperVector:
        """Reconstruct HyperVector from serialized data."""
        dimension = data["dimension"]
        vector_data = np.array(data["data"], dtype=np.int8)
        return HyperVector(dimension, vector_data)