"""
Main CDN simulator orchestrator.

This module provides the CDNSimulator class that coordinates multiple edge nodes,
manages request routing, and aggregates system-wide metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import hashlib
import json

from .node import EdgeNode, RequestResult
from .trace_generator import TraceGenerator, RequestTrace
from .utils.hashing import ConsistentHasher


@dataclass
class SimulationConfig:
    """Configuration for CDN simulation."""
    
    num_nodes: int = 8
    cache_capacity: int = 100
    policy: str = "LRU"
    ttl_seconds: float = 300.0
    origin_latency_ms: float = 100.0
    edge_latency_ms: float = 1.0
    enable_prefetch: bool = False
    partition_strategy: str = "static"
    use_consistent_hashing: bool = True
    virtual_nodes_per_node: int = 150
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_nodes": self.num_nodes,
            "cache_capacity": self.cache_capacity,
            "policy": self.policy,
            "ttl_seconds": self.ttl_seconds,
            "origin_latency_ms": self.origin_latency_ms,
            "edge_latency_ms": self.edge_latency_ms,
            "enable_prefetch": self.enable_prefetch,
            "partition_strategy": self.partition_strategy,
            "use_consistent_hashing": self.use_consistent_hashing,
            "virtual_nodes_per_node": self.virtual_nodes_per_node,
        }


@dataclass
class SimulationResult:
    """Results from a CDN simulation run."""
    
    config: SimulationConfig
    trace: RequestTrace
    results: List[RequestResult]
    node_metrics: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, Any]
    execution_time_seconds: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "trace_metadata": self.trace.metadata,
            "results": [r.to_dict() for r in self.results],
            "node_metrics": self.node_metrics,
            "aggregate_metrics": self.aggregate_metrics,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp,
        }
    
    def to_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of results."""
        metrics = self.aggregate_metrics
        return (
            f"Simulation Summary:\n"
            f"  Nodes: {self.config.num_nodes}\n"
            f"  Policy: {self.config.policy}\n"
            f"  Requests: {len(self.results):,}\n"
            f"  Hit Ratio: {metrics['avg_hit_ratio']:.2%}\n"
            f"  Bandwidth Saved: {metrics['bandwidth_saved_pct']:.1f}%\n"
            f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms\n"
            f"  Execution Time: {self.execution_time_seconds:.2f}s"
        )


class CDNSimulator:
    """
    Main orchestrator for CDN cache simulation.
    
    Manages multiple edge nodes, routes requests, and provides
    comprehensive analytics and reporting.
    """
    
    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize the CDN simulator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.nodes: List[EdgeNode] = []
        self.hasher: Optional[ConsistentHasher] = None
        
        # Initialize nodes
        self._initialize_nodes()
        
        # Initialize request routing
        self._initialize_routing()
    
    def _initialize_nodes(self) -> None:
        """Initialize edge nodes based on configuration."""
        self.nodes = []
        
        for i in range(self.config.num_nodes):
            node = EdgeNode(
                node_id=i,
                cache_capacity=self.config.cache_capacity,
                policy=self.config.policy,
                ttl_seconds=self.config.ttl_seconds,
                origin_latency_ms=self.config.origin_latency_ms,
                edge_latency_ms=self.config.edge_latency_ms,
            )
            self.nodes.append(node)
    
    def _initialize_routing(self) -> None:
        """Initialize request routing mechanism."""
        if self.config.use_consistent_hashing:
            self.hasher = ConsistentHasher(
                nodes=self.nodes,
                virtual_nodes_per_node=self.config.virtual_nodes_per_node
            )
    
    def _route_request(self, object_id: str) -> EdgeNode:
        """
        Route a request to the appropriate edge node.
        
        Args:
            object_id: ID of the requested object
            
        Returns:
            EdgeNode that should handle this request
        """
        if self.hasher is not None:
            return self.hasher.get_node(object_id)
        else:
            # Simple hash-based routing
            hash_value = int(hashlib.md5(object_id.encode()).hexdigest(), 16)
            node_index = hash_value % len(self.nodes)
            return self.nodes[node_index]
    
    def run(self, trace: RequestTrace) -> SimulationResult:
        """
        Run a complete simulation with the given trace.
        
        Args:
            trace: RequestTrace to simulate
            
        Returns:
            SimulationResult containing all metrics and results
        """
        start_time = time.time()
        results: List[RequestResult] = []
        
        print(f"Starting simulation with {len(trace.requests)} requests...")
        
        # Process each request
        for i, (object_id, timestamp) in enumerate(zip(trace.requests, trace.timestamps)):
            if i % 10000 == 0 and i > 0:
                print(f"Processed {i:,} requests...")
            
            # Route request to appropriate node
            node = self._route_request(object_id)
            
            # Get object size
            object_size = trace.object_sizes.get(object_id, 1024)
            
            # Process request
            result = node.process_request(object_id, object_size, timestamp)
            results.append(result)
        
        execution_time = time.time() - start_time
        print(f"Simulation completed in {execution_time:.2f} seconds")
        
        # Collect metrics from all nodes
        node_metrics = [node.get_metrics() for node in self.nodes]
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results, node_metrics)
        
        return SimulationResult(
            config=self.config,
            trace=trace,
            results=results,
            node_metrics=node_metrics,
            aggregate_metrics=aggregate_metrics,
            execution_time_seconds=execution_time,
            timestamp=time.time(),
        )
    
    def _calculate_aggregate_metrics(
        self,
        results: List[RequestResult],
        node_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate system-wide aggregate metrics.
        
        Args:
            results: List of all request results
            node_metrics: Metrics from each node
            
        Returns:
            Dictionary containing aggregate metrics
        """
        if not results:
            return {}
        
        # Basic statistics
        total_requests = len(results)
        total_hits = sum(1 for r in results if r.hit)
        total_misses = total_requests - total_hits
        avg_hit_ratio = total_hits / total_requests if total_requests > 0 else 0.0
        
        # Latency statistics
        hit_latencies = [r.latency_ms for r in results if r.hit]
        miss_latencies = [r.latency_ms for r in results if not r.hit]
        all_latencies = [r.latency_ms for r in results]
        
        avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
        avg_miss_latency = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0
        avg_latency = sum(all_latencies) / len(all_latencies)
        
        # Bandwidth statistics
        total_bytes_served = sum(r.object_size for r in results)
        total_bytes_from_origin = sum(r.object_size for r in results if not r.hit)
        bandwidth_saved_pct = (
            (total_bytes_served - total_bytes_from_origin) 
            / total_bytes_served * 100
            if total_bytes_served > 0 else 0.0
        )
        
        # Node-level statistics
        node_hit_ratios = [metrics['hit_ratio'] for metrics in node_metrics]
        node_utilizations = [metrics['utilization'] for metrics in node_metrics]
        
        avg_node_hit_ratio = sum(node_hit_ratios) / len(node_hit_ratios) if node_hit_ratios else 0.0
        avg_node_utilization = sum(node_utilizations) / len(node_utilizations) if node_utilizations else 0.0
        
        # Load balancing metrics
        node_request_counts = [metrics['total_requests'] for metrics in node_metrics]
        if node_request_counts:
            max_requests = max(node_request_counts)
            min_requests = min(node_request_counts)
            load_balance_ratio = min_requests / max_requests if max_requests > 0 else 1.0
        else:
            load_balance_ratio = 1.0
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "avg_hit_ratio": avg_hit_ratio,
            "avg_hit_latency_ms": avg_hit_latency,
            "avg_miss_latency_ms": avg_miss_latency,
            "avg_latency_ms": avg_latency,
            "total_bytes_served": total_bytes_served,
            "total_bytes_from_origin": total_bytes_from_origin,
            "bandwidth_saved_pct": bandwidth_saved_pct,
            "avg_node_hit_ratio": avg_node_hit_ratio,
            "avg_node_utilization": avg_node_utilization,
            "load_balance_ratio": load_balance_ratio,
            "num_nodes": len(self.nodes),
            "cache_capacity_per_node": self.config.cache_capacity,
            "total_cache_capacity": self.config.cache_capacity * len(self.nodes),
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and metrics.
        
        Returns:
            Dictionary containing current system state
        """
        node_metrics = [node.get_metrics() for node in self.nodes]
        
        return {
            "config": self.config.to_dict(),
            "num_nodes": len(self.nodes),
            "node_metrics": node_metrics,
            "system_uptime": time.time() - getattr(self, 'start_time', time.time()),
        }
    
    def reset_system(self) -> None:
        """Reset all nodes and clear metrics."""
        for node in self.nodes:
            node.reset_metrics()
        self.start_time = time.time()
    
    def get_node(self, node_id: int) -> Optional[EdgeNode]:
        """
        Get a specific node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            EdgeNode if found, None otherwise
        """
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def __str__(self) -> str:
        """String representation of the simulator."""
        return (
            f"CDNSimulator(nodes={len(self.nodes)}, "
            f"policy={self.config.policy}, "
            f"capacity={self.config.cache_capacity})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
