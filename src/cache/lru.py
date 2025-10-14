"""
Least Recently Used (LRU) cache implementation.

LRU evicts the item that has not been accessed for the longest time.
This policy is optimal for workloads with temporal locality.
"""

from typing import Optional, Dict, Any
from collections import OrderedDict
import time

from .base import CachePolicy, CacheEntry


class LRUCache(CachePolicy):
    """
    Least Recently Used cache implementation.
    
    Uses OrderedDict to maintain access order and achieve O(1) operations
    for both get and put operations.
    """
    
    def __init__(self, capacity: int) -> None:
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        super().__init__(capacity)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Retrieve an item from the cache.
        
        Moves the accessed item to the end of the OrderedDict to mark it as recently used.
        
        Args:
            key: The key to look up
            
        Returns:
            CacheEntry if found, None otherwise
        """
        if key in self._cache:
            # Move to end (most recently used)
            entry = self._cache.pop(key)
            entry.timestamp = time.time()
            entry.access_count += 1
            self._cache[key] = entry
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size: int) -> None:
        """
        Store an item in the cache.
        
        If the cache is full, evicts the least recently used item.
        
        Args:
            key: The key to store
            value: The value to store
            size: Size of the item in bytes
        """
        # If key already exists, update it
        if key in self._cache:
            self._cache.pop(key)
            self.size -= 1
        
        # If cache is full, evict LRU item
        if self.size >= self.capacity:
            self.evict()
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            size=size,
            timestamp=time.time(),
            access_count=1
        )
        self._cache[key] = entry
        self.size += 1
    
    def evict(self) -> str:
        """
        Evict the least recently used item.
        
        Returns:
            The key of the evicted item
        """
        if not self._cache:
            raise RuntimeError("Cannot evict from empty cache")
        
        # Remove first item (least recently used)
        key, _ = self._cache.popitem(last=False)
        self.size -= 1
        self.evictions += 1
        return key
    
    def _clear_impl(self) -> None:
        """Clear the internal cache storage."""
        self._cache.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get LRU-specific metrics.
        
        Returns:
            Dictionary containing standard metrics plus LRU-specific data
        """
        base_metrics = super().get_metrics()
        
        # Calculate average age of cached items
        current_time = time.time()
        if self._cache:
            avg_age = sum(current_time - entry.timestamp for entry in self._cache.values()) / len(self._cache)
        else:
            avg_age = 0.0
        
        base_metrics.update({
            "avg_age_seconds": avg_age,
            "oldest_item_age": max(current_time - entry.timestamp for entry in self._cache.values()) if self._cache else 0.0,
            "newest_item_age": min(current_time - entry.timestamp for entry in self._cache.values()) if self._cache else 0.0,
        })
        
        return base_metrics
