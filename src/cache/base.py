"""
Abstract base class for cache policies.

This module defines the interface that all caching algorithms must implement,
providing a clean abstraction for different eviction strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    key: str
    value: Any
    size: int
    timestamp: float
    access_count: int = 0
    
    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp == 0:
            self.timestamp = time.time()


class CachePolicy(ABC):
    """
    Abstract base class for cache eviction policies.
    
    All cache implementations must inherit from this class and implement
    the required abstract methods for cache operations.
    """
    
    def __init__(self, capacity: int) -> None:
        """
        Initialize the cache policy.
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")
        
        self.capacity = capacity
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Retrieve an item from the cache.
        
        Args:
            key: The key to look up
            
        Returns:
            CacheEntry if found, None otherwise
        """
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, size: int) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: The key to store
            value: The value to store
            size: Size of the item in bytes
        """
        pass
    
    @abstractmethod
    def evict(self) -> str:
        """
        Evict an item from the cache according to the policy.
        
        Returns:
            The key of the evicted item
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary containing hit ratio, miss ratio, eviction count, etc.
        """
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "capacity": self.capacity,
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "evictions": self.evictions,
            "utilization": self.size / self.capacity if self.capacity > 0 else 0.0,
        }
    
    def clear(self) -> None:
        """Clear all items from the cache and reset metrics."""
        self.size = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._clear_impl()
    
    @abstractmethod
    def _clear_impl(self) -> None:
        """Implementation-specific cache clearing logic."""
        pass
    
    def __len__(self) -> int:
        """Return the current number of items in the cache."""
        return self.size
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self.get(key) is not None
