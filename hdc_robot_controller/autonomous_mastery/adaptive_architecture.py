"""
Adaptive Architecture Manager

Dynamically reconfigures system architecture, network topology, and
computational resources based on runtime requirements and performance analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path
import psutil
import threading
import queue
from collections import defaultdict

from ..core.hypervector import HyperVector


class ComponentType(Enum):
    """Types of system components."""
    PROCESSOR = "processor"
    MEMORY = "memory"  
    NETWORK = "network"
    STORAGE = "storage"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    NEURAL_NET = "neural_net"
    HDC_MODULE = "hdc_module"


class ArchitectureState(Enum):
    """Architecture configuration states."""
    STABLE = "stable"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    EMERGENCY = "emergency"
    LEARNING = "learning"


@dataclass
class SystemComponent:
    """Represents a system component."""
    component_id: str
    component_type: ComponentType
    capabilities: Dict[str, Any]
    current_load: float
    max_capacity: float
    connections: Set[str]  # IDs of connected components
    performance_metrics: Dict[str, float]
    adaptation_history: List[Dict[str, Any]]
    priority_level: int = 1  # 1=low, 5=critical
    
    def __post_init__(self):
        if not self.adaptation_history:
            self.adaptation_history = []
    
    @property
    def utilization_ratio(self) -> float:
        """Current utilization as ratio of capacity."""
        return self.current_load / max(self.max_capacity, 0.001)
    
    @property
    def is_overloaded(self) -> bool:
        """Check if component is overloaded."""
        return self.utilization_ratio > 0.9


@dataclass
class ArchitectureConfiguration:
    """Represents a complete architecture configuration."""
    config_id: str
    components: Dict[str, SystemComponent]
    topology: nx.Graph
    performance_score: float
    resource_efficiency: float
    fault_tolerance: float
    adaptation_cost: float
    timestamp: float
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class TopologyOptimizer:
    """Optimizes network topology for performance and resilience."""
    
    def __init__(self):
        self.optimization_strategies = {
            'performance': self._optimize_for_performance,
            'resilience': self._optimize_for_resilience,
            'energy': self._optimize_for_energy,
            'latency': self._optimize_for_latency
        }
    
    def optimize_topology(self, 
                         current_topology: nx.Graph,
                         components: Dict[str, SystemComponent],
                         strategy: str = 'performance') -> nx.Graph:
        """Optimize network topology based on strategy."""
        
        if strategy not in self.optimization_strategies:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        optimizer = self.optimization_strategies[strategy]
        return optimizer(current_topology, components)
    
    def _optimize_for_performance(self, 
                                topology: nx.Graph, 
                                components: Dict[str, SystemComponent]) -> nx.Graph:
        """Optimize topology for maximum performance."""
        optimized = topology.copy()
        
        # Add high-bandwidth connections between frequently communicating components
        communication_matrix = self._analyze_communication_patterns(components)
        
        # Find component pairs with high communication but no direct connection
        for comp1, comp2 in communication_matrix:
            if (not optimized.has_edge(comp1, comp2) and 
                communication_matrix[(comp1, comp2)] > 0.7):
                
                # Add direct connection if beneficial
                optimized.add_edge(comp1, comp2, weight=communication_matrix[(comp1, comp2)])
        
        return optimized
    
    def _optimize_for_resilience(self, 
                               topology: nx.Graph, 
                               components: Dict[str, SystemComponent]) -> nx.Graph:
        """Optimize topology for fault tolerance."""
        optimized = topology.copy()
        
        # Ensure minimum connectivity for critical components
        critical_components = [
            comp_id for comp_id, comp in components.items() 
            if comp.priority_level >= 4
        ]
        
        for critical_comp in critical_components:
            current_degree = optimized.degree(critical_comp)
            
            # Ensure at least 3 connections for critical components
            if current_degree < 3:
                candidates = [
                    node for node in optimized.nodes() 
                    if (node != critical_comp and 
                        not optimized.has_edge(critical_comp, node))
                ]
                
                # Add connections to closest components
                for candidate in candidates[:3-current_degree]:
                    optimized.add_edge(critical_comp, candidate, weight=0.5)
        
        return optimized
    
    def _optimize_for_energy(self, 
                           topology: nx.Graph, 
                           components: Dict[str, SystemComponent]) -> nx.Graph:
        """Optimize topology for energy efficiency."""
        optimized = topology.copy()
        
        # Remove redundant connections with low utilization
        edges_to_remove = []
        
        for edge in optimized.edges():
            comp1, comp2 = edge
            edge_weight = optimized[comp1][comp2].get('weight', 1.0)
            
            # Remove low-weight connections
            if edge_weight < 0.2:
                # Check if removing this edge maintains connectivity
                optimized.remove_edge(comp1, comp2)
                if nx.is_connected(optimized):
                    edges_to_remove.append(edge)
                else:
                    optimized.add_edge(comp1, comp2, weight=edge_weight)
        
        return optimized
    
    def _optimize_for_latency(self, 
                            topology: nx.Graph, 
                            components: Dict[str, SystemComponent]) -> nx.Graph:
        """Optimize topology for minimum latency."""
        optimized = topology.copy()
        
        # Add shortcuts for long paths
        all_pairs_paths = dict(nx.all_pairs_shortest_path_length(optimized))
        
        for comp1 in optimized.nodes():
            for comp2 in optimized.nodes():
                if comp1 != comp2 and all_pairs_paths[comp1][comp2] > 3:
                    # Add direct connection for long paths
                    optimized.add_edge(comp1, comp2, weight=0.3)
        
        return optimized
    
    def _analyze_communication_patterns(self, 
                                      components: Dict[str, SystemComponent]) -> Dict[Tuple[str, str], float]:
        """Analyze communication patterns between components."""
        patterns = {}
        
        for comp1_id, comp1 in components.items():
            for comp2_id in comp1.connections:
                if comp2_id in components:
                    # Calculate communication intensity based on metrics
                    intensity = min(1.0, 
                        (comp1.current_load + components[comp2_id].current_load) / 2.0
                    )
                    patterns[(comp1_id, comp2_id)] = intensity
        
        return patterns


class ResourceAllocator:
    """Manages dynamic resource allocation across components."""
    
    def __init__(self):
        self.allocation_history = []
        self.prediction_model = None
        
    def allocate_resources(self, 
                         components: Dict[str, SystemComponent],
                         available_resources: Dict[str, float],
                         allocation_strategy: str = 'demand_based') -> Dict[str, Dict[str, float]]:
        """
        Allocate resources to components based on strategy.
        
        Args:
            components: System components
            available_resources: Available resource pools
            allocation_strategy: Allocation strategy
            
        Returns:
            Resource allocation map
        """
        
        if allocation_strategy == 'demand_based':
            return self._demand_based_allocation(components, available_resources)
        elif allocation_strategy == 'predictive':
            return self._predictive_allocation(components, available_resources)
        elif allocation_strategy == 'priority_based':
            return self._priority_based_allocation(components, available_resources)
        else:
            return self._fair_allocation(components, available_resources)
    
    def _demand_based_allocation(self, 
                               components: Dict[str, SystemComponent],
                               available_resources: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Allocate based on current demand."""
        allocation = {}
        
        # Calculate total demand
        total_demand = {}
        for resource_type in available_resources:
            total_demand[resource_type] = sum(
                comp.current_load * self._get_resource_multiplier(comp.component_type, resource_type)
                for comp in components.values()
            )
        
        # Allocate proportionally
        for comp_id, component in components.items():
            allocation[comp_id] = {}
            
            for resource_type, available in available_resources.items():
                if total_demand[resource_type] > 0:
                    demand_ratio = (component.current_load * 
                                  self._get_resource_multiplier(component.component_type, resource_type) /
                                  total_demand[resource_type])
                    
                    allocation[comp_id][resource_type] = available * demand_ratio
                else:
                    allocation[comp_id][resource_type] = available / len(components)
        
        return allocation
    
    def _predictive_allocation(self, 
                             components: Dict[str, SystemComponent],
                             available_resources: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Allocate based on predicted future demand."""
        # Simplified prediction - in practice, would use ML models
        allocation = {}
        
        for comp_id, component in components.items():
            allocation[comp_id] = {}
            
            # Predict future load based on trend
            if len(component.adaptation_history) >= 2:
                recent_loads = [
                    entry.get('load', component.current_load) 
                    for entry in component.adaptation_history[-5:]
                ]
                
                # Simple linear prediction
                if len(recent_loads) >= 2:
                    trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
                    predicted_load = max(0, component.current_load + trend * 3)  # 3 steps ahead
                else:
                    predicted_load = component.current_load
            else:
                predicted_load = component.current_load
            
            # Allocate based on predicted load
            for resource_type, available in available_resources.items():
                multiplier = self._get_resource_multiplier(component.component_type, resource_type)
                allocation[comp_id][resource_type] = min(
                    available * 0.8,  # Don't allocate everything
                    predicted_load * multiplier
                )
        
        return allocation
    
    def _priority_based_allocation(self, 
                                 components: Dict[str, SystemComponent],
                                 available_resources: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Allocate based on component priority."""
        allocation = {}
        
        # Sort components by priority
        sorted_components = sorted(
            components.items(),
            key=lambda x: x[1].priority_level,
            reverse=True
        )
        
        remaining_resources = available_resources.copy()
        
        for comp_id, component in sorted_components:
            allocation[comp_id] = {}
            
            for resource_type in available_resources:
                # Higher priority gets larger allocation
                priority_factor = component.priority_level / 5.0  # Normalize to 0-1
                base_allocation = remaining_resources[resource_type] * priority_factor
                
                # Consider current demand
                demand_factor = min(1.0, component.current_load / component.max_capacity)
                final_allocation = base_allocation * demand_factor
                
                allocation[comp_id][resource_type] = final_allocation
                remaining_resources[resource_type] = max(0, 
                    remaining_resources[resource_type] - final_allocation
                )
        
        return allocation
    
    def _fair_allocation(self, 
                       components: Dict[str, SystemComponent],
                       available_resources: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Fair allocation across all components."""
        allocation = {}
        num_components = len(components)
        
        for comp_id in components:
            allocation[comp_id] = {
                resource_type: available / num_components
                for resource_type, available in available_resources.items()
            }
        
        return allocation
    
    def _get_resource_multiplier(self, 
                               component_type: ComponentType, 
                               resource_type: str) -> float:
        """Get resource multiplier for component type."""
        multipliers = {
            ComponentType.PROCESSOR: {
                'cpu': 1.0,
                'memory': 0.3,
                'bandwidth': 0.2
            },
            ComponentType.MEMORY: {
                'cpu': 0.1,
                'memory': 1.0,
                'bandwidth': 0.3
            },
            ComponentType.NEURAL_NET: {
                'cpu': 0.8,
                'memory': 0.7,
                'bandwidth': 0.4
            },
            ComponentType.HDC_MODULE: {
                'cpu': 0.6,
                'memory': 0.8,
                'bandwidth': 0.3
            }
        }
        
        return multipliers.get(component_type, {}).get(resource_type, 0.5)


class AdaptiveArchitectureManager:
    """
    Adaptive Architecture Manager.
    
    Dynamically reconfigures system architecture based on:
    - Performance requirements
    - Resource availability
    - Failure conditions
    - Workload patterns
    """
    
    def __init__(self, 
                 hdc_dimension: int = 10000,
                 adaptation_threshold: float = 0.8,
                 monitoring_interval: float = 1.0):
        """
        Initialize adaptive architecture manager.
        
        Args:
            hdc_dimension: HDC dimension for encoding architectures
            adaptation_threshold: Threshold for triggering adaptations
            monitoring_interval: Interval for monitoring system state
        """
        self.hdc_dimension = hdc_dimension
        self.adaptation_threshold = adaptation_threshold
        self.monitoring_interval = monitoring_interval
        
        # Core components
        self.topology_optimizer = TopologyOptimizer()
        self.resource_allocator = ResourceAllocator()
        
        # System state
        self.components = {}  # component_id -> SystemComponent
        self.current_topology = nx.Graph()
        self.architecture_state = ArchitectureState.STABLE
        self.configuration_history = []
        
        # Monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.system_metrics_queue = queue.Queue()
        
        # Adaptation rules
        self.adaptation_rules = {}
        self.performance_targets = {
            'latency': 100.0,  # ms
            'throughput': 1000.0,  # ops/sec
            'error_rate': 0.01,  # 1%
            'resource_utilization': 0.7  # 70%
        }
        
        # Performance tracking
        self.adaptation_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'performance_improvements': [],
            'adaptation_latency': [],
            'resource_savings': [],
            'topology_changes': 0
        }
        
    def register_component(self, 
                          component_id: str,
                          component_type: ComponentType,
                          capabilities: Dict[str, Any],
                          max_capacity: float,
                          priority_level: int = 1) -> SystemComponent:
        """Register a new system component."""
        
        component = SystemComponent(
            component_id=component_id,
            component_type=component_type,
            capabilities=capabilities,
            current_load=0.0,
            max_capacity=max_capacity,
            connections=set(),
            performance_metrics={},
            adaptation_history=[],
            priority_level=priority_level
        )
        
        self.components[component_id] = component
        self.current_topology.add_node(component_id)
        
        return component
    
    def connect_components(self, 
                          component1_id: str, 
                          component2_id: str,
                          connection_weight: float = 1.0):
        """Create connection between components."""
        if (component1_id in self.components and 
            component2_id in self.components):
            
            self.components[component1_id].connections.add(component2_id)
            self.components[component2_id].connections.add(component1_id)
            
            self.current_topology.add_edge(
                component1_id, 
                component2_id, 
                weight=connection_weight
            )
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics_queue.put(metrics)
                
                # Analyze metrics for adaptation triggers
                adaptation_needed = self._analyze_adaptation_need(metrics)
                
                if adaptation_needed:
                    self._trigger_adaptation(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            'timestamp': time.time(),
            'system_resources': self._get_system_resources(),
            'component_metrics': {},
            'topology_metrics': {},
            'performance_indicators': {}
        }
        
        # Component metrics
        for comp_id, component in self.components.items():
            component_metrics = {
                'utilization': component.utilization_ratio,
                'load': component.current_load,
                'capacity': component.max_capacity,
                'connections': len(component.connections),
                'is_overloaded': component.is_overloaded
            }
            
            # Add component-specific metrics
            component_metrics.update(component.performance_metrics)
            metrics['component_metrics'][comp_id] = component_metrics
        
        # Topology metrics
        metrics['topology_metrics'] = {
            'node_count': self.current_topology.number_of_nodes(),
            'edge_count': self.current_topology.number_of_edges(),
            'density': nx.density(self.current_topology),
            'average_clustering': nx.average_clustering(self.current_topology),
            'diameter': nx.diameter(self.current_topology) if nx.is_connected(self.current_topology) else float('inf')
        }
        
        # Performance indicators
        metrics['performance_indicators'] = self._calculate_performance_indicators()
        
        return metrics
    
    def _get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': sum([
                psutil.net_io_counters().bytes_sent,
                psutil.net_io_counters().bytes_recv
            ]) / (1024 * 1024)  # MB
        }
    
    def _calculate_performance_indicators(self) -> Dict[str, float]:
        """Calculate overall performance indicators."""
        if not self.components:
            return {}
        
        # Average utilization
        avg_utilization = np.mean([
            comp.utilization_ratio for comp in self.components.values()
        ])
        
        # Load balance variance
        utilizations = [comp.utilization_ratio for comp in self.components.values()]
        load_balance_variance = np.var(utilizations)
        
        # Overload percentage
        overloaded_count = sum(1 for comp in self.components.values() if comp.is_overloaded)
        overload_percentage = overloaded_count / len(self.components)
        
        return {
            'average_utilization': avg_utilization,
            'load_balance_variance': load_balance_variance,
            'overload_percentage': overload_percentage,
            'topology_efficiency': 1.0 / (nx.average_shortest_path_length(self.current_topology) + 1)
            if nx.is_connected(self.current_topology) else 0.0
        }
    
    def _analyze_adaptation_need(self, metrics: Dict[str, Any]) -> bool:
        """Analyze if architecture adaptation is needed."""
        performance = metrics['performance_indicators']
        
        # Check adaptation triggers
        triggers = [
            # High overall utilization
            performance.get('average_utilization', 0) > self.adaptation_threshold,
            
            # High load imbalance
            performance.get('load_balance_variance', 0) > 0.3,
            
            # Too many overloaded components
            performance.get('overload_percentage', 0) > 0.2,
            
            # Poor topology efficiency
            performance.get('topology_efficiency', 1.0) < 0.3,
            
            # Individual component overloads
            any(comp.is_overloaded for comp in self.components.values())
        ]
        
        return any(triggers)
    
    def _trigger_adaptation(self, metrics: Dict[str, Any]):
        """Trigger architecture adaptation."""
        if self.architecture_state != ArchitectureState.STABLE:
            return  # Already adapting
        
        self.architecture_state = ArchitectureState.ADAPTING
        adaptation_start = time.time()
        
        try:
            # Determine adaptation strategy
            strategy = self._determine_adaptation_strategy(metrics)
            
            # Execute adaptation
            success = self._execute_adaptation(strategy, metrics)
            
            # Update metrics
            adaptation_time = time.time() - adaptation_start
            self.adaptation_metrics['total_adaptations'] += 1
            self.adaptation_metrics['adaptation_latency'].append(adaptation_time)
            
            if success:
                self.adaptation_metrics['successful_adaptations'] += 1
                
                # Measure performance improvement
                improvement = self._measure_performance_improvement(metrics)
                self.adaptation_metrics['performance_improvements'].append(improvement)
            
        except Exception as e:
            print(f"Adaptation failed: {e}")
        finally:
            self.architecture_state = ArchitectureState.STABLE
    
    def _determine_adaptation_strategy(self, metrics: Dict[str, Any]) -> str:
        """Determine the best adaptation strategy."""
        performance = metrics['performance_indicators']
        
        # Priority-based strategy selection
        if performance.get('overload_percentage', 0) > 0.3:
            return 'load_balancing'
        elif performance.get('load_balance_variance', 0) > 0.4:
            return 'redistribution'
        elif performance.get('topology_efficiency', 1.0) < 0.4:
            return 'topology_optimization'
        else:
            return 'resource_reallocation'
    
    def _execute_adaptation(self, strategy: str, metrics: Dict[str, Any]) -> bool:
        """Execute the selected adaptation strategy."""
        try:
            if strategy == 'load_balancing':
                return self._execute_load_balancing(metrics)
            elif strategy == 'redistribution':
                return self._execute_redistribution(metrics)
            elif strategy == 'topology_optimization':
                return self._execute_topology_optimization(metrics)
            elif strategy == 'resource_reallocation':
                return self._execute_resource_reallocation(metrics)
            else:
                return False
        except Exception as e:
            print(f"Strategy execution failed: {e}")
            return False
    
    def _execute_load_balancing(self, metrics: Dict[str, Any]) -> bool:
        """Execute load balancing adaptation."""
        # Find overloaded and underloaded components
        overloaded = []
        underloaded = []
        
        for comp_id, component in self.components.items():
            if component.is_overloaded:
                overloaded.append(comp_id)
            elif component.utilization_ratio < 0.5:
                underloaded.append(comp_id)
        
        # Redistribute load
        redistributed = 0
        for overloaded_comp in overloaded:
            if underloaded:
                # Move some load to underloaded component
                target_comp = underloaded[redistributed % len(underloaded)]
                
                # Calculate load to transfer
                source = self.components[overloaded_comp]
                target = self.components[target_comp]
                
                transfer_amount = min(
                    source.current_load * 0.3,  # Transfer up to 30%
                    target.max_capacity - target.current_load
                )
                
                if transfer_amount > 0:
                    source.current_load -= transfer_amount
                    target.current_load += transfer_amount
                    
                    # Record adaptation
                    adaptation_record = {
                        'type': 'load_transfer',
                        'from': overloaded_comp,
                        'to': target_comp,
                        'amount': transfer_amount,
                        'timestamp': time.time()
                    }
                    
                    source.adaptation_history.append(adaptation_record)
                    target.adaptation_history.append(adaptation_record)
                    
                redistributed += 1
        
        return redistributed > 0
    
    def _execute_redistribution(self, metrics: Dict[str, Any]) -> bool:
        """Execute component redistribution."""
        # Get current resource allocation
        available_resources = {
            'cpu': 100.0 - metrics['system_resources']['cpu_percent'],
            'memory': 100.0 - metrics['system_resources']['memory_percent']
        }
        
        # Reallocate resources
        new_allocation = self.resource_allocator.allocate_resources(
            self.components,
            available_resources,
            'demand_based'
        )
        
        # Apply new allocation
        for comp_id, allocation in new_allocation.items():
            if comp_id in self.components:
                component = self.components[comp_id]
                
                # Update component capacity based on allocation
                cpu_ratio = allocation.get('cpu', 0) / 100.0
                memory_ratio = allocation.get('memory', 0) / 100.0
                
                # Simple capacity update (in practice, would interface with actual resources)
                component.max_capacity = component.max_capacity * (cpu_ratio + memory_ratio) / 2
        
        return True
    
    def _execute_topology_optimization(self, metrics: Dict[str, Any]) -> bool:
        """Execute topology optimization."""
        # Optimize topology for performance
        optimized_topology = self.topology_optimizer.optimize_topology(
            self.current_topology,
            self.components,
            'performance'
        )
        
        # Apply topology changes
        changes_made = False
        
        # Add new edges
        for edge in optimized_topology.edges():
            if not self.current_topology.has_edge(*edge):
                self.connect_components(edge[0], edge[1], 
                    optimized_topology[edge[0]][edge[1]].get('weight', 1.0))
                changes_made = True
        
        # Remove old edges (carefully)
        for edge in list(self.current_topology.edges()):
            if not optimized_topology.has_edge(*edge):
                # Only remove if it doesn't break connectivity
                self.current_topology.remove_edge(*edge)
                if nx.is_connected(self.current_topology):
                    changes_made = True
                else:
                    # Restore the edge
                    self.current_topology.add_edge(*edge)
        
        if changes_made:
            self.adaptation_metrics['topology_changes'] += 1
        
        return changes_made
    
    def _execute_resource_reallocation(self, metrics: Dict[str, Any]) -> bool:
        """Execute resource reallocation."""
        # Use predictive allocation
        available_resources = {
            'cpu': 100.0 - metrics['system_resources']['cpu_percent'],
            'memory': 100.0 - metrics['system_resources']['memory_percent']
        }
        
        new_allocation = self.resource_allocator.allocate_resources(
            self.components,
            available_resources,
            'predictive'
        )
        
        # Apply allocation
        total_savings = 0.0
        
        for comp_id, allocation in new_allocation.items():
            if comp_id in self.components:
                component = self.components[comp_id]
                
                # Calculate resource savings
                old_allocation = component.performance_metrics.get('allocated_resources', 0)
                new_total = sum(allocation.values())
                
                if old_allocation > 0:
                    savings = (old_allocation - new_total) / old_allocation
                    total_savings += savings
                
                # Update component
                component.performance_metrics['allocated_resources'] = new_total
        
        if total_savings > 0:
            self.adaptation_metrics['resource_savings'].append(total_savings)
        
        return True
    
    def _measure_performance_improvement(self, baseline_metrics: Dict[str, Any]) -> float:
        """Measure performance improvement after adaptation."""
        # Simple improvement metric based on utilization balance
        old_variance = baseline_metrics['performance_indicators'].get('load_balance_variance', 0)
        
        # Get new metrics
        new_metrics = self._collect_system_metrics()
        new_variance = new_metrics['performance_indicators'].get('load_balance_variance', 0)
        
        # Improvement is reduction in variance
        if old_variance > 0:
            improvement = (old_variance - new_variance) / old_variance
            return max(0, improvement)  # Only positive improvements
        
        return 0.0
    
    def add_adaptation_rule(self, 
                          rule_name: str,
                          condition: Callable[[Dict[str, Any]], bool],
                          action: Callable[[Dict[str, Any]], bool]):
        """Add custom adaptation rule."""
        self.adaptation_rules[rule_name] = {
            'condition': condition,
            'action': action
        }
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get comprehensive architecture summary."""
        summary = {
            'architecture_state': self.architecture_state.value,
            'component_count': len(self.components),
            'topology_metrics': {
                'nodes': self.current_topology.number_of_nodes(),
                'edges': self.current_topology.number_of_edges(),
                'density': nx.density(self.current_topology),
                'is_connected': nx.is_connected(self.current_topology)
            },
            'component_summary': {},
            'adaptation_metrics': self.adaptation_metrics.copy(),
            'performance_targets': self.performance_targets.copy()
        }
        
        # Component summary
        for comp_id, component in self.components.items():
            summary['component_summary'][comp_id] = {
                'type': component.component_type.value,
                'utilization': component.utilization_ratio,
                'connections': len(component.connections),
                'priority': component.priority_level,
                'adaptations': len(component.adaptation_history)
            }
        
        return summary
    
    def save_architecture_config(self, filepath: str):
        """Save current architecture configuration."""
        config_data = {
            'timestamp': time.time(),
            'components': {},
            'topology': nx.node_link_data(self.current_topology),
            'adaptation_metrics': self.adaptation_metrics,
            'performance_targets': self.performance_targets
        }
        
        # Serialize components
        for comp_id, component in self.components.items():
            config_data['components'][comp_id] = {
                'component_type': component.component_type.value,
                'capabilities': component.capabilities,
                'current_load': component.current_load,
                'max_capacity': component.max_capacity,
                'connections': list(component.connections),
                'performance_metrics': component.performance_metrics,
                'priority_level': component.priority_level,
                'adaptation_history': component.adaptation_history
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def load_architecture_config(self, filepath: str):
        """Load architecture configuration from file."""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        # Clear current state
        self.components.clear()
        self.current_topology.clear()
        
        # Load components
        for comp_id, comp_data in config_data['components'].items():
            component = SystemComponent(
                component_id=comp_id,
                component_type=ComponentType(comp_data['component_type']),
                capabilities=comp_data['capabilities'],
                current_load=comp_data['current_load'],
                max_capacity=comp_data['max_capacity'],
                connections=set(comp_data['connections']),
                performance_metrics=comp_data['performance_metrics'],
                adaptation_history=comp_data['adaptation_history'],
                priority_level=comp_data.get('priority_level', 1)
            )
            
            self.components[comp_id] = component
        
        # Load topology
        self.current_topology = nx.node_link_graph(config_data['topology'])
        
        # Load metrics and targets
        self.adaptation_metrics.update(config_data.get('adaptation_metrics', {}))
        self.performance_targets.update(config_data.get('performance_targets', {}))