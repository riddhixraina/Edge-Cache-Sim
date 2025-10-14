"""
Time-To-Live (TTL) cache implementation.

TTL expires items after a fixed duration, regardless of access patterns.
This policy is optimal for content with freshness requirements.
"""

from typing import Optional, Dict, Any
import time

from .base import CachePolicy, CacheEntry


class TTLCache(CachePolicy):
    """
    Time-To-Live cache implementation.
    
    Items expire after a fixed duration (TTL) regardless of access patterns.
    Expired items are removed on access or during eviction.
    """
    
    def __init__(self, capacity: int, ttl_seconds: float = 300.0) -> None:
        """
        Initialize TTL cache.
        
        Args:
            capacity: Maximum number of items the cache can hold
            ttl_seconds: Time-to-live for cached items in seconds (default: 5 minutes)
        """
        super().__init__(capacity)
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._expired_count = 0
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Retrieve an item from the cache.
        
        Checks if the item has expired and removes it if so.
        
        Args:
            key: The key to look up
            
        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        if key in self._cache:
            entry = self._cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                self._cache.pop(key)
                self.size -= 1
                self._expired_count += 1
                self.misses += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.timestamp = time.time()
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, size: int) -> None:
        """
        Store an item in the cache.
        
        If the cache is full, evicts an expired item or the oldest item.
        
        Args:
            key: The key to store
            value: The value to store
            size: Size of the item in bytes
        """
        # If key already exists, update it
        if key in self._cache:
            self._cache.pop(key)
            self.size -= 1
        
        # Clean up expired items first
        self._cleanup_expired()
        
        # If cache is still full, evict oldest item
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
        Evict the oldest item from the cache.
        
        Returns:
            The key of the evicted item
        """
        if not self._cache:
            raise RuntimeError("Cannot evict from empty cache")
        
        # Find the oldest item
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        self._cache.pop(oldest_key)
        self.size -= 1
        self.evictions += 1
        return oldest_key
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if a cache entry has expired.
        
        Args:
            entry: The cache entry to check
            
        Returns:
            True if expired, False otherwise
        """
        return time.time() - entry.timestamp > self.ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from the cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._cache.pop(key)
            self.size -= 1
            self._expired_count += 1
    
    def _clear_impl(self) -> None:
        """Clear the internal cache storage."""
        self._cache.clear()
        self._expired_count = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get TTL-specific metrics.
        
        Returns:
            Dictionary containing standard metrics plus TTL-specific data
        """
        base_metrics = super().get_metrics()
        
        # Calculate age statistics
        current_time = time.time()
        if self._cache:
            ages = [current_time - entry.timestamp for entry in self._cache.values()]
            avg_age = sum(ages) / len(ages)
            max_age = max(ages)
            min_age = min(ages)
        else:
            avg_age = 0.0
            max_age = 0.0
            min_age = 0.0
        
        base_metrics.update({
            "ttl_seconds": self.ttl_seconds,
            "expired_count": self._expired_count,
            "avg_age_seconds": avg_age,
            "max_age_seconds": max_age,
            "min_age_seconds": min_age,
            "expiration_rate": self._expired_count / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0,
        })
        
        return base_metrics
    
    def set_ttl(self, ttl_seconds: float) -> None:
        """
        Update the TTL value for future entries.
        
        Args:
            ttl_seconds: New TTL value in seconds
        """
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        self.ttl_seconds = ttl_seconds
