"""
Distributed Computing Coordinator for HDC Robot Controller.

Provides distributed processing capabilities for scaling HDC operations
across multiple nodes, with load balancing and fault tolerance.
"""

import time
import json
import threading
import socket
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import multiprocessing as mp
from enum import Enum

from ..core.hypervector import HyperVector
from ..core.memory import HierarchicalMemory
from ..core.logging_system import get_logger


class NodeStatus(Enum):
    """Status of distributed nodes."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class TaskType(Enum):
    """Types of distributed tasks."""
    BUNDLE = "bundle"
    SIMILARITY_SEARCH = "similarity_search"
    BIND_BATCH = "bind_batch"
    MEMORY_QUERY = "memory_query"
    CUSTOM = "custom"


class DistributedTask:
    """Represents a distributed computation task."""
    
    def __init__(self, task_id: str, task_type: TaskType, data: Any,
                 priority: int = 1, timeout: float = 60.0):
        self.task_id = task_id
        self.task_type = task_type
        self.data = data
        self.priority = priority
        self.timeout = timeout
        self.created_time = time.time()
        self.assigned_node = None
        self.start_time = None
        self.completion_time = None
        self.result = None
        self.error = None
        self.retry_count = 0


class NodeInfo:
    """Information about a distributed computing node."""
    
    def __init__(self, node_id: str, host: str, port: int,
                 capabilities: Dict[str, Any]):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.capabilities = capabilities
        self.status = NodeStatus.INITIALIZING
        self.last_heartbeat = time.time()
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
        self.load_score = 0.0


class DistributedCoordinator:
    """Coordinates distributed HDC processing across multiple nodes."""
    
    def __init__(self, coordinator_port: int = 8888, max_workers: int = 10):
        """Initialize distributed coordinator.
        
        Args:
            coordinator_port: Port for coordinator communication
            max_workers: Maximum number of worker threads
        """
        self.coordinator_port = coordinator_port
        self.max_workers = max_workers
        self.logger = get_logger()
        
        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.task_queue = deque()
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks = deque(maxlen=1000)
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        self.fault_tolerance = DistributedFaultTolerance()
        
        # Communication
        self.server_socket = None
        self.server_thread = None
        self.is_running = False
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_dispatcher_thread = None
        self.heartbeat_monitor_thread = None
        
        # Performance monitoring
        self.stats = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Distributed coordinator initialized",
                        port=coordinator_port,
                        max_workers=max_workers)
    
    def start_coordinator(self):
        """Start the distributed coordinator."""
        with self._lock:
            if self.is_running:
                return
            
            self.is_running = True
            
            # Start server
            self._start_server()
            
            # Start task dispatcher
            self.task_dispatcher_thread = threading.Thread(
                target=self._task_dispatcher_loop, daemon=True)
            self.task_dispatcher_thread.start()
            
            # Start heartbeat monitor
            self.heartbeat_monitor_thread = threading.Thread(
                target=self._heartbeat_monitor_loop, daemon=True)
            self.heartbeat_monitor_thread.start()
            
            self.logger.info("Distributed coordinator started",
                           port=self.coordinator_port)
    
    def stop_coordinator(self):
        """Stop the distributed coordinator."""
        with self._lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Close server socket
            if self.server_socket:
                self.server_socket.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Distributed coordinator stopped")
    
    def register_node(self, node_id: str, host: str, port: int,
                     capabilities: Dict[str, Any]) -> bool:
        """Register a new compute node.
        
        Args:
            node_id: Unique node identifier
            host: Node hostname/IP
            port: Node port
            capabilities: Node capabilities (CPU count, memory, GPU, etc.)
            
        Returns:
            True if registration successful
        """
        with self._lock:
            if node_id in self.nodes:
                self.logger.warning(f"Node {node_id} already registered")
                return False
            
            node = NodeInfo(node_id, host, port, capabilities)
            self.nodes[node_id] = node
            
            # Test connection
            if self._test_node_connection(node):
                node.status = NodeStatus.READY
                self.logger.info(f"Node {node_id} registered successfully",
                               host=host, port=port, capabilities=capabilities)
                return True
            else:
                node.status = NodeStatus.ERROR
                self.logger.error(f"Failed to connect to node {node_id}")
                return False
    
    def submit_task(self, task_type: TaskType, data: Any,
                   priority: int = 1, timeout: float = 60.0) -> str:
        """Submit a task for distributed processing.
        
        Args:
            task_type: Type of task to execute
            data: Task data/parameters
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            
        Returns:
            Task ID for tracking
        """
        task_id = self._generate_task_id()
        task = DistributedTask(task_id, task_type, data, priority, timeout)
        
        with self._lock:
            self.task_queue.append(task)
            self.stats['total_tasks_submitted'] += 1
            
            # Sort by priority
            self.task_queue = deque(sorted(self.task_queue, 
                                          key=lambda t: t.priority, reverse=True))
        
        self.logger.debug(f"Task submitted: {task_id}",
                         task_type=task_type.value,
                         priority=priority)
        
        return task_id
    
    def get_task_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Get result of a submitted task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for result
            
        Returns:
            Task result or raises exception if failed/timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                # Check if task is completed
                for completed_task in self.completed_tasks:
                    if completed_task.task_id == task_id:
                        if completed_task.error:
                            raise RuntimeError(f"Task failed: {completed_task.error}")
                        return completed_task.result
                
                # Check if task is still active
                if task_id in self.active_tasks:
                    time.sleep(0.1)
                    continue
                
                # Check if task is in queue
                if any(t.task_id == task_id for t in self.task_queue):
                    time.sleep(0.1)
                    continue
            
            # Task not found
            raise ValueError(f"Task {task_id} not found")
        
        raise TimeoutError(f"Task {task_id} timed out")
    
    def submit_and_wait(self, task_type: TaskType, data: Any,
                       priority: int = 1, timeout: float = 60.0) -> Any:
        """Submit task and wait for result.
        
        Args:
            task_type: Type of task
            data: Task data
            priority: Task priority
            timeout: Timeout for both task execution and waiting
            
        Returns:
            Task result
        """
        task_id = self.submit_task(task_type, data, priority, timeout)
        return self.get_task_result(task_id, timeout)
    
    def distributed_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """Perform distributed vector bundling.
        
        Args:
            vectors: List of vectors to bundle
            
        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty vector list")
        
        if len(vectors) < 100:  # Use local processing for small tasks
            return HyperVector.bundle_vectors(vectors)
        
        return self.submit_and_wait(TaskType.BUNDLE, {'vectors': vectors})
    
    def distributed_similarity_search(self, query_vector: HyperVector,
                                    target_vectors: List[HyperVector],
                                    top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform distributed similarity search.
        
        Args:
            query_vector: Query vector
            target_vectors: Target vectors to search
            top_k: Number of top results
            
        Returns:
            List of (index, similarity) tuples
        """
        if not target_vectors:
            return []
        
        data = {
            'query_vector': query_vector,
            'target_vectors': target_vectors,
            'top_k': top_k
        }
        
        return self.submit_and_wait(TaskType.SIMILARITY_SEARCH, data)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status and statistics.
        
        Returns:
            Dictionary with cluster information
        """
        with self._lock:
            node_stats = {}
            healthy_nodes = 0
            total_capacity = 0
            
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    'status': node.status.value,
                    'host': node.host,
                    'port': node.port,
                    'active_tasks': node.active_tasks,
                    'completed_tasks': node.completed_tasks,
                    'failed_tasks': node.failed_tasks,
                    'load_score': node.load_score,
                    'last_heartbeat': node.last_heartbeat
                }
                
                if node.status == NodeStatus.READY:
                    healthy_nodes += 1
                
                total_capacity += node.capabilities.get('cpu_count', 1)
            
            return {
                'total_nodes': len(self.nodes),
                'healthy_nodes': healthy_nodes,
                'total_capacity': total_capacity,
                'queued_tasks': len(self.task_queue),
                'active_tasks': len(self.active_tasks),
                'node_details': node_stats,
                'performance_stats': self.stats.copy()
            }
    
    def _start_server(self):
        """Start coordinator server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.coordinator_port))
            self.server_socket.listen(10)
            
            self.server_thread = threading.Thread(
                target=self._server_loop, daemon=True)
            self.server_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start coordinator server: {e}")
            raise
    
    def _server_loop(self):
        """Main server loop for handling node communications."""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Server loop error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle communication with a client node."""
        try:
            # Receive message
            data = self._receive_message(client_socket)
            if not data:
                return
            
            message = json.loads(data.decode('utf-8'))
            message_type = message.get('type')
            
            response = {'status': 'error', 'message': 'Unknown message type'}
            
            if message_type == 'heartbeat':
                response = self._handle_heartbeat(message)
            elif message_type == 'task_result':
                response = self._handle_task_result(message)
            elif message_type == 'node_status':
                response = self._handle_node_status(message)
            
            # Send response
            response_data = json.dumps(response).encode('utf-8')
            self._send_message(client_socket, response_data)
            
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
    
    def _task_dispatcher_loop(self):
        """Main loop for dispatching tasks to nodes."""
        while self.is_running:
            try:
                with self._lock:
                    if not self.task_queue:
                        time.sleep(0.1)
                        continue
                    
                    # Get next task
                    task = self.task_queue.popleft()
                    
                    # Find available node
                    available_node = self.load_balancer.select_node(
                        self.nodes, task.task_type)
                    
                    if available_node:
                        # Dispatch task
                        self._dispatch_task(task, available_node)
                    else:
                        # No available nodes, put task back
                        self.task_queue.appendleft(task)
                        time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Task dispatcher error: {e}")
                time.sleep(1.0)
    
    def _dispatch_task(self, task: DistributedTask, node: NodeInfo):
        """Dispatch a task to a specific node."""
        try:
            task.assigned_node = node.node_id
            task.start_time = time.time()
            
            # Send task to node
            success = self._send_task_to_node(task, node)
            
            if success:
                self.active_tasks[task.task_id] = task
                node.active_tasks += 1
                node.status = NodeStatus.BUSY
                
                self.logger.debug(f"Task {task.task_id} dispatched to {node.node_id}")
            else:
                self.logger.error(f"Failed to send task {task.task_id} to {node.node_id}")
                self._handle_task_failure(task, "Failed to send to node")
        
        except Exception as e:
            self.logger.error(f"Error dispatching task {task.task_id}: {e}")
            self._handle_task_failure(task, str(e))
    
    def _heartbeat_monitor_loop(self):
        """Monitor node heartbeats and health."""
        while self.is_running:
            try:
                current_time = time.time()
                
                with self._lock:
                    for node_id, node in self.nodes.items():
                        # Check if node is offline
                        if current_time - node.last_heartbeat > 30.0:  # 30 second timeout
                            if node.status != NodeStatus.OFFLINE:
                                self.logger.warning(f"Node {node_id} appears offline")
                                node.status = NodeStatus.OFFLINE
                                
                                # Handle active tasks on offline node
                                self._handle_node_failure(node_id)
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(10.0)
    
    def _test_node_connection(self, node: NodeInfo) -> bool:
        """Test connection to a node."""
        try:
            # Simple connection test
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(5.0)
            result = test_socket.connect_ex((node.host, node.port))
            test_socket.close()
            
            return result == 0
        except:
            return False
    
    def _send_task_to_node(self, task: DistributedTask, node: NodeInfo) -> bool:
        """Send task to a compute node."""
        try:
            # Create connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((node.host, node.port))
            
            # Prepare task message
            message = {
                'type': 'task',
                'task_id': task.task_id,
                'task_type': task.task_type.value,
                'data': self._serialize_task_data(task.data),
                'timeout': task.timeout
            }
            
            # Send message
            message_data = json.dumps(message).encode('utf-8')
            self._send_message(sock, message_data)
            
            # Get acknowledgment
            response_data = self._receive_message(sock)
            if response_data:
                response = json.loads(response_data.decode('utf-8'))
                return response.get('status') == 'accepted'
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending task to node {node.node_id}: {e}")
            return False
        finally:
            try:
                sock.close()
            except:
                pass
    
    def _serialize_task_data(self, data: Any) -> str:
        """Serialize task data for transmission."""
        # Convert HyperVectors to serializable format
        if isinstance(data, dict):
            serialized = {}
            for key, value in data.items():
                if isinstance(value, HyperVector):
                    serialized[key] = {
                        'type': 'hypervector',
                        'dimension': value.dimension,
                        'data': value.data.tolist()
                    }
                elif isinstance(value, list) and value and isinstance(value[0], HyperVector):
                    serialized[key] = {
                        'type': 'hypervector_list',
                        'vectors': [
                            {
                                'dimension': hv.dimension,
                                'data': hv.data.tolist()
                            } for hv in value
                        ]
                    }
                else:
                    serialized[key] = value
            
            return json.dumps(serialized)
        
        return json.dumps(data)
    
    def _send_message(self, sock: socket.socket, data: bytes):
        """Send message with length prefix."""
        message_length = len(data)
        length_bytes = message_length.to_bytes(4, byteorder='big')
        sock.sendall(length_bytes + data)
    
    def _receive_message(self, sock: socket.socket) -> Optional[bytes]:
        """Receive message with length prefix."""
        try:
            # Receive length
            length_bytes = sock.recv(4)
            if len(length_bytes) != 4:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive message
            data = b''
            while len(data) < message_length:
                chunk = sock.recv(min(message_length - len(data), 4096))
                if not chunk:
                    return None
                data += chunk
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            return None
    
    def _handle_heartbeat(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle heartbeat message from node."""
        node_id = message.get('node_id')
        
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_heartbeat = time.time()
            
            # Update node status
            if node.status == NodeStatus.OFFLINE:
                node.status = NodeStatus.READY
                self.logger.info(f"Node {node_id} back online")
            
            return {'status': 'ok'}
        
        return {'status': 'error', 'message': 'Unknown node'}
    
    def _handle_task_result(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task result from node."""
        task_id = message.get('task_id')
        node_id = message.get('node_id')
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            if message.get('success'):
                task.result = message.get('result')
                task.completion_time = time.time()
                self._complete_task(task, True)
            else:
                task.error = message.get('error', 'Unknown error')
                self._complete_task(task, False)
            
            return {'status': 'ok'}
        
        return {'status': 'error', 'message': 'Unknown task'}
    
    def _handle_node_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle node status update."""
        node_id = message.get('node_id')
        
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.active_tasks = message.get('active_tasks', 0)
            node.load_score = message.get('load_score', 0.0)
            
            return {'status': 'ok'}
        
        return {'status': 'error', 'message': 'Unknown node'}
    
    def _complete_task(self, task: DistributedTask, success: bool):
        """Mark task as completed."""
        with self._lock:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update node status
            if task.assigned_node and task.assigned_node in self.nodes:
                node = self.nodes[task.assigned_node]
                node.active_tasks = max(0, node.active_tasks - 1)
                
                if success:
                    node.completed_tasks += 1
                else:
                    node.failed_tasks += 1
                
                if node.active_tasks == 0:
                    node.status = NodeStatus.READY
                
                # Update processing time
                if task.start_time and task.completion_time:
                    processing_time = task.completion_time - task.start_time
                    node.total_processing_time += processing_time
            
            # Add to completed tasks
            self.completed_tasks.append(task)
            
            # Update statistics
            if success:
                self.stats['total_tasks_completed'] += 1
            else:
                self.stats['total_tasks_failed'] += 1
            
            if task.start_time and task.completion_time:
                processing_time = task.completion_time - task.start_time
                self.stats['total_processing_time'] += processing_time
                
                # Update average
                total_completed = self.stats['total_tasks_completed']
                if total_completed > 0:
                    self.stats['average_task_time'] = (
                        self.stats['total_processing_time'] / total_completed
                    )
    
    def _handle_task_failure(self, task: DistributedTask, error: str):
        """Handle task failure."""
        task.error = error
        task.retry_count += 1
        
        # Retry logic
        if task.retry_count < 3:
            self.logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
            with self._lock:
                self.task_queue.append(task)
        else:
            self.logger.error(f"Task {task.task_id} failed permanently: {error}")
            self._complete_task(task, False)
    
    def _handle_node_failure(self, node_id: str):
        """Handle node failure by reassigning tasks."""
        with self._lock:
            failed_tasks = []
            
            # Find tasks assigned to failed node
            for task_id, task in self.active_tasks.items():
                if task.assigned_node == node_id:
                    failed_tasks.append(task)
            
            # Reassign tasks
            for task in failed_tasks:
                self.logger.warning(f"Reassigning task {task.task_id} due to node failure")
                del self.active_tasks[task.task_id]
                task.assigned_node = None
                task.start_time = None
                self.task_queue.append(task)
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        timestamp = str(int(time.time() * 1000000))
        hash_input = f"{timestamp}_{threading.get_ident()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


class LoadBalancer:
    """Load balancer for selecting optimal compute nodes."""
    
    def select_node(self, nodes: Dict[str, NodeInfo], 
                   task_type: TaskType) -> Optional[NodeInfo]:
        """Select best node for task execution.
        
        Args:
            nodes: Available compute nodes
            task_type: Type of task to execute
            
        Returns:
            Selected node or None if no suitable node available
        """
        # Filter ready nodes
        available_nodes = [
            node for node in nodes.values()
            if node.status == NodeStatus.READY
        ]
        
        if not available_nodes:
            return None
        
        # Score nodes based on load and capabilities
        best_node = None
        best_score = float('inf')
        
        for node in available_nodes:
            score = self._calculate_node_score(node, task_type)
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _calculate_node_score(self, node: NodeInfo, task_type: TaskType) -> float:
        """Calculate node selection score (lower is better)."""
        # Base score from current load
        load_score = node.active_tasks * 1.0
        
        # Add capability bonuses
        cpu_bonus = node.capabilities.get('cpu_count', 1) * 0.1
        memory_bonus = node.capabilities.get('memory_gb', 4) * 0.02
        
        # Task-specific scoring
        if task_type == TaskType.BUNDLE and node.capabilities.get('gpu_available'):
            gpu_bonus = 0.5
        else:
            gpu_bonus = 0.0
        
        # Historical performance
        if node.completed_tasks > 0:
            avg_time = node.total_processing_time / node.completed_tasks
            performance_penalty = avg_time * 0.01
        else:
            performance_penalty = 0.0
        
        # Failure rate penalty
        total_tasks = node.completed_tasks + node.failed_tasks
        if total_tasks > 0:
            failure_rate = node.failed_tasks / total_tasks
            failure_penalty = failure_rate * 2.0
        else:
            failure_penalty = 0.0
        
        return (load_score - cpu_bonus - memory_bonus - gpu_bonus + 
                performance_penalty + failure_penalty)


class DistributedFaultTolerance:
    """Fault tolerance mechanisms for distributed processing."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def detect_node_failures(self, nodes: Dict[str, NodeInfo]) -> List[str]:
        """Detect failed nodes based on heartbeat timeouts."""
        current_time = time.time()
        failed_nodes = []
        
        for node_id, node in nodes.items():
            if current_time - node.last_heartbeat > 60.0:  # 1 minute timeout
                if node.status != NodeStatus.OFFLINE:
                    failed_nodes.append(node_id)
        
        return failed_nodes
    
    def handle_partial_failures(self, task: DistributedTask) -> bool:
        """Handle partial task failures with recovery strategies."""
        # For now, simple retry logic
        return task.retry_count < 3