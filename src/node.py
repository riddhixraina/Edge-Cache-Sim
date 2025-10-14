"""
Edge node implementation for CDN simulation.

This module provides the EdgeNode class that represents a single edge cache
node in the CDN network, with integrated caching and metrics tracking.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import hashlib

from .cache import CachePolicy, LRUCache, LFUCache, TTLCache


@dataclass
class RequestResult:
    """Result of processing a single request."""
    
    object_id: str
    hit: bool
    latency_ms: float
    object_size: int
    timestamp: float
    node_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "object_id": self.object_id,
            "hit": self.hit,
            "latency_ms": self.latency_ms,
            "object_size": self.object_size,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
        }


class EdgeNode:
    """
    Represents a single edge cache node in the CDN network.
    
    Each node has its own cache instance and tracks performance metrics.
    Supports different caching policies and provides detailed analytics.
    """
    
    def __init__(
        self,
        node_id: int,
        cache_capacity: int,
        policy: str = "LRU",
        ttl_seconds: float = 300.0,
        origin_latency_ms: float = 100.0,
        edge_latency_ms: float = 1.0,
    ) -> None:
        """
        Initialize an edge node.
        
        Args:
            node_id: Unique identifier for this node
            cache_capacity: Maximum number of objects this node can cache
            policy: Caching policy ("LRU", "LFU", or "TTL")
            ttl_seconds: TTL for cached objects (only used with TTL policy)
            origin_latency_ms: Simulated latency to origin server
            edge_latency_ms: Simulated latency for cache hits
        """
        self.node_id = node_id
        self.cache_capacity = cache_capacity
        self.policy = policy
        self.origin_latency_ms = origin_latency_ms
        self.edge_latency_ms = edge_latency_ms
        
        # Initialize cache based on policy
        if policy == "LRU":
            self.cache: CachePolicy = LRUCache(cache_capacity)
        elif policy == "LFU":
            self.cache: CachePolicy = LFUCache(cache_capacity)
        elif policy == "TTL":
            self.cache: CachePolicy = TTLCache(cache_capacity, ttl_seconds)
        else:
            raise ValueError(f"Unknown cache policy: {policy}")
        
        # Metrics tracking
        self.total_requests = 0
        self.total_bytes_served = 0
        self.total_bytes_from_origin = 0
        self.request_history: List[RequestResult] = []
        
        # Performance counters
        self.start_time = time.time()
    
    def process_request(
        self,
        object_id: str,
        object_size: int,
        timestamp: float,
    ) -> RequestResult:
        """
        Process a single request for an object.
        
        Args:
            object_id: ID of the requested object
            object_size: Size of the object in bytes
            timestamp: Request timestamp
            
        Returns:
            RequestResult containing hit/miss status and latency
        """
        self.total_requests += 1
        
        # Check cache
        cache_entry = self.cache.get(object_id)
        
        if cache_entry is not None:
            # Cache hit
            hit = True
            latency_ms = self.edge_latency_ms
            self.total_bytes_served += object_size
        else:
            # Cache miss - fetch from origin
            hit = False
            latency_ms = self.origin_latency_ms
            
            # Store in cache
            self.cache.put(object_id, f"data_{object_id}", object_size)
            
            self.total_bytes_served += object_size
            self.total_bytes_from_origin += object_size
        
        # Create result
        result = RequestResult(
            object_id=object_id,
            hit=hit,
            latency_ms=latency_ms,
            object_size=object_size,
            timestamp=timestamp,
            node_id=self.node_id,
        )
        
        # Store in history
        self.request_history.append(result)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for this node.
        
        Returns:
            Dictionary containing node performance metrics
        """
        cache_metrics = self.cache.get_metrics()
        
        # Calculate additional metrics
        current_time = time.time()
        uptime_seconds = current_time - self.start_time
        
        # Bandwidth metrics
        bandwidth_saved_pct = 0.0
        if self.total_bytes_served > 0:
            bandwidth_saved_pct = (
                (self.total_bytes_served - self.total_bytes_from_origin) 
                / self.total_bytes_served * 100
            )
        
        # Latency metrics
        if self.request_history:
            hit_latencies = [r.latency_ms for r in self.request_history if r.hit]
            miss_latencies = [r.latency_ms for r in self.request_history if not r.hit]
            
            avg_hit_latency = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
            avg_miss_latency = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0
            avg_latency = sum(r.latency_ms for r in self.request_history) / len(self.request_history)
        else:
            avg_hit_latency = 0.0
            avg_miss_latency = 0.0
            avg_latency = 0.0
        
        # Request rate
        request_rate = self.total_requests / uptime_seconds if uptime_seconds > 0 else 0.0
        
        return {
            "node_id": self.node_id,
            "policy": self.policy,
            "uptime_seconds": uptime_seconds,
            "total_requests": self.total_requests,
            "request_rate": request_rate,
            "total_bytes_served": self.total_bytes_served,
            "total_bytes_from_origin": self.total_bytes_from_origin,
            "bandwidth_saved_pct": bandwidth_saved_pct,
            "avg_hit_latency_ms": avg_hit_latency,
            "avg_miss_latency_ms": avg_miss_latency,
            "avg_latency_ms": avg_latency,
            **cache_metrics,
        }
    
    def get_recent_metrics(self, window_seconds: float = 60.0) -> Dict[str, Any]:
        """
        Get metrics for recent requests within a time window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary containing recent performance metrics
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_requests = [
            r for r in self.request_history
            if r.timestamp >= cutoff_time
        ]
        
        if not recent_requests:
            return {
                "recent_requests": 0,
                "recent_hit_ratio": 0.0,
                "recent_avg_latency_ms": 0.0,
                "recent_bandwidth_saved_pct": 0.0,
            }
        
        recent_hits = sum(1 for r in recent_requests if r.hit)
        recent_total = len(recent_requests)
        recent_hit_ratio = recent_hits / recent_total if recent_total > 0 else 0.0
        
        recent_avg_latency = sum(r.latency_ms for r in recent_requests) / recent_total
        
        recent_bytes_served = sum(r.object_size for r in recent_requests)
        recent_bytes_from_origin = sum(
            r.object_size for r in recent_requests if not r.hit
        )
        
        recent_bandwidth_saved_pct = (
            (recent_bytes_served - recent_bytes_from_origin) 
            / recent_bytes_served * 100
            if recent_bytes_served > 0 else 0.0
        )
        
        return {
            "recent_requests": recent_total,
            "recent_hit_ratio": recent_hit_ratio,
            "recent_avg_latency_ms": recent_avg_latency,
            "recent_bandwidth_saved_pct": recent_bandwidth_saved_pct,
        }
    
    def clear_history(self) -> None:
        """Clear request history while preserving cache and basic metrics."""
        self.request_history.clear()
    
    def reset_metrics(self) -> None:
        """Reset all metrics and clear cache."""
        self.cache.clear()
        self.total_requests = 0
        self.total_bytes_served = 0
        self.total_bytes_from_origin = 0
        self.request_history.clear()
        self.start_time = time.time()
    
    def get_cache_contents(self) -> Dict[str, Any]:
        """
        Get current cache contents for debugging/analysis.
        
        Returns:
            Dictionary containing cache state information
        """
        if hasattr(self.cache, '_cache'):
            cache_dict = self.cache._cache
            if isinstance(cache_dict, dict):
                return {
                    "cached_objects": list(cache_dict.keys()),
                    "cache_size": len(cache_dict),
                    "cache_utilization": len(cache_dict) / self.cache_capacity,
                }
        
        return {
            "cached_objects": [],
            "cache_size": 0,
            "cache_utilization": 0.0,
        }
    
    def __str__(self) -> str:
        """String representation of the node."""
        metrics = self.get_metrics()
        return (
            f"EdgeNode(id={self.node_id}, policy={self.policy}, "
            f"requests={self.total_requests}, "
            f"hit_ratio={metrics['hit_ratio']:.2%}, "
            f"utilization={metrics['utilization']:.2%})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
