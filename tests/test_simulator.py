"""
Unit tests for EdgeNode and CDNSimulator.

Tests edge node functionality, request processing, metrics collection,
and system-wide simulation orchestration.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.node import EdgeNode, RequestResult
from src.simulator import CDNSimulator, SimulationConfig, SimulationResult
from src.trace_generator import TraceGenerator, RequestTrace
from src.cache.lru import LRUCache


class TestEdgeNode:
    """Test EdgeNode class."""
    
    def test_node_initialization(self):
        """Test node initialization."""
        node = EdgeNode(
            node_id=1,
            cache_capacity=100,
            policy="LRU",
            ttl_seconds=300.0,
            origin_latency_ms=100.0,
            edge_latency_ms=1.0
        )
        
        assert node.node_id == 1
        assert node.cache_capacity == 100
        assert node.policy == "LRU"
        assert node.origin_latency_ms == 100.0
        assert node.edge_latency_ms == 1.0
        assert isinstance(node.cache, LRUCache)
        assert node.total_requests == 0
        assert node.total_bytes_served == 0
        assert node.total_bytes_from_origin == 0
    
    def test_node_with_different_policies(self):
        """Test node initialization with different policies."""
        from src.cache.lfu import LFUCache
        from src.cache.ttl import TTLCache
        
        # Test LRU
        node_lru = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        assert isinstance(node_lru.cache, LRUCache)
        
        # Test LFU
        node_lfu = EdgeNode(node_id=2, cache_capacity=100, policy="LFU")
        assert isinstance(node_lfu.cache, LFUCache)
        
        # Test TTL
        node_ttl = EdgeNode(node_id=3, cache_capacity=100, policy="TTL")
        assert isinstance(node_ttl.cache, TTLCache)
    
    def test_invalid_policy(self):
        """Test node initialization with invalid policy."""
        with pytest.raises(ValueError):
            EdgeNode(node_id=1, cache_capacity=100, policy="INVALID")
    
    def test_process_request_cache_hit(self):
        """Test processing a cache hit request."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        # First request (cache miss)
        result1 = node.process_request("obj1", 1024, 0.0)
        
        assert result1.object_id == "obj1"
        assert result1.hit == False
        assert result1.latency_ms == 100.0  # Origin latency
        assert result1.object_size == 1024
        assert result1.timestamp == 0.0
        assert result1.node_id == 1
        
        assert node.total_requests == 1
        assert node.total_bytes_served == 1024
        assert node.total_bytes_from_origin == 1024
        
        # Second request (cache hit)
        result2 = node.process_request("obj1", 1024, 1.0)
        
        assert result2.object_id == "obj1"
        assert result2.hit == True
        assert result2.latency_ms == 1.0  # Edge latency
        assert result2.object_size == 1024
        assert result2.timestamp == 1.0
        assert result2.node_id == 1
        
        assert node.total_requests == 2
        assert node.total_bytes_served == 2048
        assert node.total_bytes_from_origin == 1024  # Unchanged
    
    def test_process_request_cache_miss(self):
        """Test processing a cache miss request."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        result = node.process_request("obj1", 1024, 0.0)
        
        assert result.hit == False
        assert result.latency_ms == 100.0
        assert node.total_bytes_from_origin == 1024
    
    def test_node_metrics(self):
        """Test node metrics collection."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        # Process some requests
        node.process_request("obj1", 1024, 0.0)  # Miss
        node.process_request("obj1", 1024, 1.0)  # Hit
        node.process_request("obj2", 2048, 2.0)  # Miss
        
        metrics = node.get_metrics()
        
        assert metrics["node_id"] == 1
        assert metrics["policy"] == "LRU"
        assert metrics["total_requests"] == 3
        assert metrics["total_bytes_served"] == 4096
        assert metrics["total_bytes_from_origin"] == 3072
        assert metrics["bandwidth_saved_pct"] == 25.0  # 1024/4096 * 100
        assert metrics["hit_ratio"] == 1/3
        assert metrics["capacity"] == 100
        assert metrics["size"] == 2  # Two objects in cache
    
    def test_recent_metrics(self):
        """Test recent metrics calculation."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        current_time = time.time()
        
        # Process requests
        node.process_request("obj1", 1024, current_time - 30)  # Old request
        node.process_request("obj2", 1024, current_time - 10)  # Recent request
        node.process_request("obj3", 1024, current_time - 5)   # Recent request
        
        # Get recent metrics (last 20 seconds)
        recent_metrics = node.get_recent_metrics(window_seconds=20.0)
        
        assert recent_metrics["recent_requests"] == 2  # Only recent requests
        assert recent_metrics["recent_hit_ratio"] == 0.0  # All misses
        assert recent_metrics["recent_avg_latency_ms"] == 100.0  # Origin latency
        assert recent_metrics["recent_bandwidth_saved_pct"] == 0.0  # No hits
    
    def test_clear_history(self):
        """Test clearing request history."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        node.process_request("obj1", 1024, 0.0)
        node.process_request("obj2", 1024, 1.0)
        
        assert len(node.request_history) == 2
        
        node.clear_history()
        
        assert len(node.request_history) == 0
        assert node.total_requests == 2  # Metrics preserved
        assert node.total_bytes_served == 2048
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        node.process_request("obj1", 1024, 0.0)
        node.process_request("obj2", 1024, 1.0)
        
        assert node.total_requests == 2
        assert len(node.request_history) == 2
        assert node.cache.size == 2
        
        node.reset_metrics()
        
        assert node.total_requests == 0
        assert len(node.request_history) == 0
        assert node.cache.size == 0
        assert node.total_bytes_served == 0
        assert node.total_bytes_from_origin == 0
    
    def test_cache_contents(self):
        """Test getting cache contents."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        node.process_request("obj1", 1024, 0.0)
        node.process_request("obj2", 1024, 1.0)
        
        contents = node.get_cache_contents()
        
        assert contents["cache_size"] == 2
        assert contents["cache_utilization"] == 0.02  # 2/100
        assert "obj1" in contents["cached_objects"]
        assert "obj2" in contents["cached_objects"]
    
    def test_node_string_representation(self):
        """Test node string representation."""
        node = EdgeNode(node_id=1, cache_capacity=100, policy="LRU")
        
        node.process_request("obj1", 1024, 0.0)
        node.process_request("obj1", 1024, 1.0)  # Hit
        
        str_repr = str(node)
        
        assert "EdgeNode" in str_repr
        assert "id=1" in str_repr
        assert "policy=LRU" in str_repr
        assert "requests=2" in str_repr


class TestCDNSimulator:
    """Test CDNSimulator class."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        config = SimulationConfig(
            num_nodes=4,
            cache_capacity=100,
            policy="LRU"
        )
        
        simulator = CDNSimulator(config)
        
        assert len(simulator.nodes) == 4
        assert simulator.config == config
        assert simulator.hasher is not None
        
        # Check all nodes are initialized
        for i, node in enumerate(simulator.nodes):
            assert node.node_id == i
            assert node.cache_capacity == 100
            assert node.policy == "LRU"
    
    def test_simulator_without_consistent_hashing(self):
        """Test simulator without consistent hashing."""
        config = SimulationConfig(
            num_nodes=4,
            cache_capacity=100,
            policy="LRU",
            use_consistent_hashing=False
        )
        
        simulator = CDNSimulator(config)
        
        assert simulator.hasher is None
    
    def test_request_routing(self):
        """Test request routing to nodes."""
        config = SimulationConfig(num_nodes=4, cache_capacity=100, policy="LRU")
        simulator = CDNSimulator(config)
        
        # Test routing with consistent hashing
        node1 = simulator._route_request("obj1")
        node2 = simulator._route_request("obj2")
        
        assert node1 in simulator.nodes
        assert node2 in simulator.nodes
        
        # Same object should route to same node
        node1_again = simulator._route_request("obj1")
        assert node1 == node1_again
    
    def test_simulator_run(self):
        """Test running a complete simulation."""
        config = SimulationConfig(
            num_nodes=2,
            cache_capacity=10,
            policy="LRU"
        )
        
        simulator = CDNSimulator(config)
        
        # Generate a simple trace
        generator = TraceGenerator(seed=42)
        trace = generator.generate_zipf_trace(
            num_requests=100,
            catalog_size=50,
            skew=0.9
        )
        
        # Run simulation
        result = simulator.run(trace)
        
        assert isinstance(result, SimulationResult)
        assert result.config == config
        assert result.trace == trace
        assert len(result.results) == 100
        assert len(result.node_metrics) == 2
        assert result.execution_time_seconds > 0
        
        # Check aggregate metrics
        metrics = result.aggregate_metrics
        assert metrics["total_requests"] == 100
        assert metrics["num_nodes"] == 2
        assert 0 <= metrics["avg_hit_ratio"] <= 1
        assert metrics["avg_latency_ms"] > 0
    
    def test_aggregate_metrics_calculation(self):
        """Test aggregate metrics calculation."""
        config = SimulationConfig(num_nodes=2, cache_capacity=10, policy="LRU")
        simulator = CDNSimulator(config)
        
        # Create mock results
        mock_results = [
            Mock(hit=True, latency_ms=1.0, object_size=100),
            Mock(hit=False, latency_ms=100.0, object_size=200),
            Mock(hit=True, latency_ms=1.0, object_size=150),
        ]
        
        mock_node_metrics = [
            {"hit_ratio": 0.5, "utilization": 0.3, "total_requests": 2},
            {"hit_ratio": 0.8, "utilization": 0.7, "total_requests": 1},
        ]
        
        aggregate_metrics = simulator._calculate_aggregate_metrics(
            mock_results, mock_node_metrics
        )
        
        assert aggregate_metrics["total_requests"] == 3
        assert aggregate_metrics["total_hits"] == 2
        assert aggregate_metrics["total_misses"] == 1
        assert aggregate_metrics["avg_hit_ratio"] == 2/3
        assert aggregate_metrics["avg_latency_ms"] == (1.0 + 100.0 + 1.0) / 3
        assert aggregate_metrics["bandwidth_saved_pct"] == (450 - 200) / 450 * 100
        assert aggregate_metrics["num_nodes"] == 2
    
    def test_system_status(self):
        """Test getting system status."""
        config = SimulationConfig(num_nodes=2, cache_capacity=100, policy="LRU")
        simulator = CDNSimulator(config)
        
        status = simulator.get_system_status()
        
        assert "config" in status
        assert "num_nodes" in status
        assert "node_metrics" in status
        assert "system_uptime" in status
        
        assert status["num_nodes"] == 2
        assert len(status["node_metrics"]) == 2
    
    def test_reset_system(self):
        """Test resetting the system."""
        config = SimulationConfig(num_nodes=2, cache_capacity=100, policy="LRU")
        simulator = CDNSimulator(config)
        
        # Process some requests
        generator = TraceGenerator(seed=42)
        trace = generator.generate_zipf_trace(num_requests=10, catalog_size=20)
        simulator.run(trace)
        
        # Check nodes have processed requests
        assert simulator.nodes[0].total_requests > 0
        
        # Reset system
        simulator.reset_system()
        
        # Check all nodes are reset
        for node in simulator.nodes:
            assert node.total_requests == 0
            assert len(node.request_history) == 0
            assert node.cache.size == 0
    
    def test_get_node(self):
        """Test getting specific node."""
        config = SimulationConfig(num_nodes=4, cache_capacity=100, policy="LRU")
        simulator = CDNSimulator(config)
        
        node = simulator.get_node(2)
        assert node is not None
        assert node.node_id == 2
        
        node = simulator.get_node(10)  # Non-existent
        assert node is None
    
    def test_simulator_string_representation(self):
        """Test simulator string representation."""
        config = SimulationConfig(num_nodes=4, cache_capacity=100, policy="LRU")
        simulator = CDNSimulator(config)
        
        str_repr = str(simulator)
        
        assert "CDNSimulator" in str_repr
        assert "nodes=4" in str_repr
        assert "policy=LRU" in str_repr
        assert "capacity=100" in str_repr


class TestSimulationConfig:
    """Test SimulationConfig class."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = SimulationConfig(
            num_nodes=8,
            cache_capacity=100,
            policy="LRU",
            ttl_seconds=300.0,
            origin_latency_ms=100.0,
            edge_latency_ms=1.0,
            enable_prefetch=True,
            partition_strategy="static",
            use_consistent_hashing=True,
            virtual_nodes_per_node=150
        )
        
        assert config.num_nodes == 8
        assert config.cache_capacity == 100
        assert config.policy == "LRU"
        assert config.ttl_seconds == 300.0
        assert config.origin_latency_ms == 100.0
        assert config.edge_latency_ms == 1.0
        assert config.enable_prefetch == True
        assert config.partition_strategy == "static"
        assert config.use_consistent_hashing == True
        assert config.virtual_nodes_per_node == 150
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = SimulationConfig()
        
        assert config.num_nodes == 8
        assert config.cache_capacity == 100
        assert config.policy == "LRU"
        assert config.ttl_seconds == 300.0
        assert config.origin_latency_ms == 100.0
        assert config.edge_latency_ms == 1.0
        assert config.enable_prefetch == False
        assert config.partition_strategy == "static"
        assert config.use_consistent_hashing == True
        assert config.virtual_nodes_per_node == 150
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = SimulationConfig(num_nodes=4, cache_capacity=50, policy="LFU")
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["num_nodes"] == 4
        assert config_dict["cache_capacity"] == 50
        assert config_dict["policy"] == "LFU"


class TestSimulationResult:
    """Test SimulationResult class."""
    
    def test_result_creation(self):
        """Test simulation result creation."""
        config = SimulationConfig(num_nodes=2, cache_capacity=100, policy="LRU")
        
        # Create mock trace
        trace = RequestTrace(
            requests=["obj1", "obj2"],
            timestamps=[0.0, 1.0],
            object_sizes={"obj1": 100, "obj2": 200},
            metadata={"test": "data"}
        )
        
        # Create mock results
        results = [
            RequestResult("obj1", True, 1.0, 100, 0.0, 0),
            RequestResult("obj2", False, 100.0, 200, 1.0, 1),
        ]
        
        node_metrics = [
            {"node_id": 0, "hit_ratio": 1.0, "utilization": 0.5},
            {"node_id": 1, "hit_ratio": 0.0, "utilization": 0.3},
        ]
        
        aggregate_metrics = {
            "total_requests": 2,
            "avg_hit_ratio": 0.5,
            "avg_latency_ms": 50.5,
        }
        
        result = SimulationResult(
            config=config,
            trace=trace,
            results=results,
            node_metrics=node_metrics,
            aggregate_metrics=aggregate_metrics,
            execution_time_seconds=0.1,
            timestamp=time.time()
        )
        
        assert result.config == config
        assert result.trace == trace
        assert result.results == results
        assert result.node_metrics == node_metrics
        assert result.aggregate_metrics == aggregate_metrics
        assert result.execution_time_seconds == 0.1
    
    def test_result_to_dict(self):
        """Test result to dictionary conversion."""
        config = SimulationConfig(num_nodes=2, cache_capacity=100, policy="LRU")
        trace = RequestTrace(requests=[], timestamps=[], object_sizes={}, metadata={})
        
        result = SimulationResult(
            config=config,
            trace=trace,
            results=[],
            node_metrics=[],
            aggregate_metrics={},
            execution_time_seconds=0.1,
            timestamp=time.time()
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "config" in result_dict
        assert "trace_metadata" in result_dict
        assert "results" in result_dict
        assert "node_metrics" in result_dict
        assert "aggregate_metrics" in result_dict
        assert "execution_time_seconds" in result_dict
        assert "timestamp" in result_dict
    
    def test_result_summary(self):
        """Test result summary generation."""
        config = SimulationConfig(num_nodes=4, cache_capacity=100, policy="LRU")
        trace = RequestTrace(requests=[], timestamps=[], object_sizes={}, metadata={})
        
        result = SimulationResult(
            config=config,
            trace=trace,
            results=[],
            node_metrics=[],
            aggregate_metrics={
                "total_requests": 1000,
                "avg_hit_ratio": 0.75,
                "bandwidth_saved_pct": 60.0,
                "avg_latency_ms": 25.5,
            },
            execution_time_seconds=1.5,
            timestamp=time.time()
        )
        
        summary = result.get_summary()
        
        assert isinstance(summary, str)
        assert "Nodes: 4" in summary
        assert "Policy: LRU" in summary
        assert "Requests: 1,000" in summary
        assert "Hit Ratio: 75%" in summary
        assert "Bandwidth Saved: 60.0%" in summary
        assert "Avg Latency: 25.5ms" in summary
        assert "Execution Time: 1.50s" in summary


class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_simulation(self):
        """Test complete end-to-end simulation."""
        config = SimulationConfig(
            num_nodes=4,
            cache_capacity=50,
            policy="LRU"
        )
        
        simulator = CDNSimulator(config)
        
        generator = TraceGenerator(seed=42)
        trace = generator.generate_zipf_trace(
            num_requests=500,
            catalog_size=100,
            skew=0.9
        )
        
        result = simulator.run(trace)
        
        # Verify result integrity
        assert len(result.results) == 500
        assert len(result.node_metrics) == 4
        
        # Verify all requests were processed
        total_requests = sum(node["total_requests"] for node in result.node_metrics)
        assert total_requests == 500
        
        # Verify metrics consistency
        assert result.aggregate_metrics["total_requests"] == 500
        assert result.aggregate_metrics["num_nodes"] == 4
        
        # Verify hit ratio is reasonable
        hit_ratio = result.aggregate_metrics["avg_hit_ratio"]
        assert 0 <= hit_ratio <= 1
    
    def test_different_policies_comparison(self):
        """Test simulation with different policies."""
        policies = ["LRU", "LFU", "TTL"]
        results = []
        
        generator = TraceGenerator(seed=42)
        trace = generator.generate_zipf_trace(
            num_requests=200,
            catalog_size=50,
            skew=0.9
        )
        
        for policy in policies:
            config = SimulationConfig(
                num_nodes=2,
                cache_capacity=20,
                policy=policy
            )
            
            simulator = CDNSimulator(config)
            result = simulator.run(trace)
            results.append(result)
        
        # All simulations should complete successfully
        assert len(results) == 3
        
        # All should have same number of requests
        for result in results:
            assert result.aggregate_metrics["total_requests"] == 200
        
        # Hit ratios should be different (policies behave differently)
        hit_ratios = [r.aggregate_metrics["avg_hit_ratio"] for r in results]
        assert len(set(hit_ratios)) > 1  # At least some variation
