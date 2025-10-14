"""
Least Frequently Used (LFU) cache implementation.

LFU evicts the item with the lowest access frequency.
This policy is optimal for workloads with static popularity distributions.
"""

from typing import Optional, Dict, Any, Tuple
import heapq
import time

from .base import CachePolicy, CacheEntry


class LFUCache(CachePolicy):
    """
    Least Frequently Used cache implementation.
    
    Uses a min-heap to efficiently track access frequencies and evict
    the least frequently used item when the cache is full.
    """
    
    def __init__(self, capacity: int) -> None:
        """
        Initialize LFU cache.
        
        Args:
            capacity: Maximum number of items the cache can hold
        """
        super().__init__(capacity)
        self._cache: Dict[str, CacheEntry] = {}
        self._frequency_heap: list[Tuple[int, float, str]] = []  # (frequency, timestamp, key)
        self._key_to_freq: Dict[str, int] = {}
        self._counter = 0  # For tie-breaking in heap
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Retrieve an item from the cache.
        
        Increments the access frequency of the retrieved item.
        
        Args:
            key: The key to look up
            
        Returns:
            CacheEntry if found, None otherwise
        """
        if key in self._cache:
            entry = self._cache[key]
            entry.access_count += 1
            entry.timestamp = time.time()
            
            # Update frequency
            old_freq = self._key_to_freq[key]
            new_freq = old_freq + 1
            self._key_to_freq[key] = new_freq
            
            # Add new entry to heap (old entry will be ignored)
            self._counter += 1
            heapq.heappush(self._frequency_heap, (new_freq, self._counter, key))
            
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size: int) -> None:
        """
        Store an item in the cache.
        
        If the cache is full, evicts the least frequently used item.
        
        Args:
            key: The key to store
            value: The value to store
            size: Size of the item in bytes
        """
        # If key already exists, update it
        if key in self._cache:
            self._cache.pop(key)
            self.size -= 1
        
        # If cache is full, evict LFU item
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
        self._key_to_freq[key] = 1
        
        # Add to frequency heap
        self._counter += 1
        heapq.heappush(self._frequency_heap, (1, self._counter, key))
        
        self.size += 1
    
    def evict(self) -> str:
        """
        Evict the least frequently used item.
        
        Returns:
            The key of the evicted item
        """
        if not self._cache:
            raise RuntimeError("Cannot evict from empty cache")
        
        # Find the LFU item by popping from heap until we find a valid entry
        while self._frequency_heap:
            freq, _, key = heapq.heappop(self._frequency_heap)
            
            # Check if this entry is still valid and has the correct frequency
            if key in self._cache and self._key_to_freq[key] == freq:
                # Remove from cache
                self._cache.pop(key)
                self._key_to_freq.pop(key)
                self.size -= 1
                self.evictions += 1
                return key
        
        # If heap is empty but cache is not, evict arbitrary item
        if self._cache:
            key = next(iter(self._cache))
            self._cache.pop(key)
            self._key_to_freq.pop(key)
            self.size -= 1
            self.evictions += 1
            return key
        
        raise RuntimeError("Cannot evict from empty cache")
    
    def _clear_impl(self) -> None:
        """Clear the internal cache storage."""
        self._cache.clear()
        self._frequency_heap.clear()
        self._key_to_freq.clear()
        self._counter = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get LFU-specific metrics.
        
        Returns:
            Dictionary containing standard metrics plus LFU-specific data
        """
        base_metrics = super().get_metrics()
        
        if self._cache:
            frequencies = list(self._key_to_freq.values())
            avg_frequency = sum(frequencies) / len(frequencies)
            max_frequency = max(frequencies)
            min_frequency = min(frequencies)
        else:
            avg_frequency = 0.0
            max_frequency = 0
            min_frequency = 0
        
        base_metrics.update({
            "avg_frequency": avg_frequency,
            "max_frequency": max_frequency,
            "min_frequency": min_frequency,
            "heap_size": len(self._frequency_heap),
        })
        
        return base_metrics
