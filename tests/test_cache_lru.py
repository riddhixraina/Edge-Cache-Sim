"""
Comprehensive unit tests for cache policies.

Tests LRU, LFU, and TTL cache implementations with edge cases,
performance validation, and correctness verification.
"""

import pytest
import time
from unittest.mock import patch

from src.cache.base import CachePolicy, CacheEntry
from src.cache.lru import LRUCache
from src.cache.lfu import LFUCache
from src.cache.ttl import TTLCache


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size=1024,
            timestamp=time.time()
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size == 1024
        assert entry.access_count == 0
        assert entry.timestamp > 0
    
    def test_cache_entry_post_init(self):
        """Test automatic timestamp initialization."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size=1024,
            timestamp=0
        )
        
        assert entry.timestamp > 0


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_lru_basic_operations(self):
        """Test basic LRU cache operations."""
        cache = LRUCache(capacity=3)
        
        # Test put and get
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        cache.put("key3", "value3", 300)
        
        assert len(cache) == 3
        
        # Test get
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"
        assert entry.access_count == 2  # 1 from put, 1 from get
        
        # Test cache hit
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(capacity=2)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3", 300)
        
        assert len(cache) == 2
        assert cache.get("key1") is not None  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") is not None  # Should be there
        
        assert cache.evictions == 1
    
    def test_lru_metrics(self):
        """Test LRU cache metrics."""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        metrics = cache.get_metrics()
        
        assert metrics["capacity"] == 3
        assert metrics["size"] == 2
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_ratio"] == 0.5
        assert metrics["evictions"] == 0
        assert metrics["utilization"] == 2/3
    
    def test_lru_clear(self):
        """Test cache clearing."""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        cache.get("key1")
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0
    
    def test_lru_edge_cases(self):
        """Test LRU edge cases."""
        cache = LRUCache(capacity=1)
        
        # Test single item
        cache.put("key1", "value1", 100)
        assert cache.get("key1") is not None
        
        # Test eviction with single item
        cache.put("key2", "value2", 200)
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        
        # Test empty cache
        cache.clear()
        assert cache.get("nonexistent") is None
        assert cache.misses == 1


class TestLFUCache:
    """Test LFU cache implementation."""
    
    def test_lfu_basic_operations(self):
        """Test basic LFU cache operations."""
        cache = LFUCache(capacity=3)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        cache.put("key3", "value3", 300)
        
        assert len(cache) == 3
        
        # Test get
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"
        assert entry.access_count == 2
    
    def test_lfu_eviction(self):
        """Test LFU eviction policy."""
        cache = LFUCache(capacity=2)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        # Access key1 multiple times to increase frequency
        cache.get("key1")
        cache.get("key1")
        
        # Add key3, should evict key2 (least frequently used)
        cache.put("key3", "value3", 300)
        
        assert len(cache) == 2
        assert cache.get("key1") is not None  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") is not None  # Should be there
        
        assert cache.evictions == 1
    
    def test_lfu_frequency_tracking(self):
        """Test LFU frequency tracking."""
        cache = LFUCache(capacity=3)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        # Access key1 more frequently
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")
        
        metrics = cache.get_metrics()
        
        assert metrics["avg_frequency"] > 1.0
        assert metrics["max_frequency"] >= 3  # key1 accessed 3 times
        assert metrics["min_frequency"] >= 1  # key2 accessed 1 time
    
    def test_lfu_metrics(self):
        """Test LFU cache metrics."""
        cache = LFUCache(capacity=3)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        metrics = cache.get_metrics()
        
        assert metrics["capacity"] == 3
        assert metrics["size"] == 2
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_ratio"] == 0.5
        assert metrics["evictions"] == 0
    
    def test_lfu_clear(self):
        """Test LFU cache clearing."""
        cache = LFUCache(capacity=3)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        cache.get("key1")
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0


class TestTTLCache:
    """Test TTL cache implementation."""
    
    def test_ttl_basic_operations(self):
        """Test basic TTL cache operations."""
        cache = TTLCache(capacity=3, ttl_seconds=1.0)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        assert len(cache) == 2
        
        # Test get before expiration
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = TTLCache(capacity=3, ttl_seconds=0.1)  # Very short TTL
        
        cache.put("key1", "value1", 100)
        
        # Should be available immediately
        assert cache.get("key1") is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        assert cache.get("key1") is None
        assert cache.misses == 1  # Counted as miss due to expiration
    
    def test_ttl_eviction(self):
        """Test TTL eviction of oldest items."""
        cache = TTLCache(capacity=2, ttl_seconds=10.0)  # Long TTL
        
        cache.put("key1", "value1", 100)
        time.sleep(0.01)  # Small delay
        cache.put("key2", "value2", 200)
        time.sleep(0.01)  # Small delay
        cache.put("key3", "value3", 300)  # Should evict key1 (oldest)
        
        assert len(cache) == 2
        assert cache.get("key1") is None      # Should be evicted
        assert cache.get("key2") is not None  # Should be there
        assert cache.get("key3") is not None  # Should be there
        
        assert cache.evictions == 1
    
    def test_ttl_cleanup(self):
        """Test TTL cleanup of expired items."""
        cache = TTLCache(capacity=3, ttl_seconds=0.1)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Add new item, should trigger cleanup
        cache.put("key3", "value3", 300)
        
        assert len(cache) == 1  # Only key3 should remain
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
    
    def test_ttl_metrics(self):
        """Test TTL cache metrics."""
        cache = TTLCache(capacity=3, ttl_seconds=1.0)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        metrics = cache.get_metrics()
        
        assert metrics["capacity"] == 3
        assert metrics["size"] == 2
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_ratio"] == 0.5
        assert metrics["ttl_seconds"] == 1.0
        assert metrics["expired_count"] == 0
    
    def test_ttl_set_ttl(self):
        """Test TTL value update."""
        cache = TTLCache(capacity=3, ttl_seconds=1.0)
        
        assert cache.ttl_seconds == 1.0
        
        cache.set_ttl(2.0)
        assert cache.ttl_seconds == 2.0
        
        # Test invalid TTL
        with pytest.raises(ValueError):
            cache.set_ttl(-1.0)
    
    def test_ttl_clear(self):
        """Test TTL cache clearing."""
        cache = TTLCache(capacity=3, ttl_seconds=1.0)
        
        cache.put("key1", "value1", 100)
        cache.put("key2", "value2", 200)
        cache.get("key1")
        
        cache.clear()
        
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0
        assert cache._expired_count == 0


class TestCachePolicyInterface:
    """Test cache policy interface compliance."""
    
    def test_lru_interface_compliance(self):
        """Test LRU implements CachePolicy interface."""
        cache = LRUCache(capacity=3)
        
        # Test abstract methods are implemented
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'put')
        assert hasattr(cache, 'evict')
        assert hasattr(cache, 'get_metrics')
        assert hasattr(cache, 'clear')
        
        # Test methods work
        cache.put("key1", "value1", 100)
        entry = cache.get("key1")
        assert entry is not None
        
        metrics = cache.get_metrics()
        assert isinstance(metrics, dict)
        
        cache.clear()
        assert len(cache) == 0
    
    def test_lfu_interface_compliance(self):
        """Test LFU implements CachePolicy interface."""
        cache = LFUCache(capacity=3)
        
        # Test abstract methods are implemented
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'put')
        assert hasattr(cache, 'evict')
        assert hasattr(cache, 'get_metrics')
        assert hasattr(cache, 'clear')
        
        # Test methods work
        cache.put("key1", "value1", 100)
        entry = cache.get("key1")
        assert entry is not None
        
        metrics = cache.get_metrics()
        assert isinstance(metrics, dict)
        
        cache.clear()
        assert len(cache) == 0
    
    def test_ttl_interface_compliance(self):
        """Test TTL implements CachePolicy interface."""
        cache = TTLCache(capacity=3)
        
        # Test abstract methods are implemented
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'put')
        assert hasattr(cache, 'evict')
        assert hasattr(cache, 'get_metrics')
        assert hasattr(cache, 'clear')
        
        # Test methods work
        cache.put("key1", "value1", 100)
        entry = cache.get("key1")
        assert entry is not None
        
        metrics = cache.get_metrics()
        assert isinstance(metrics, dict)
        
        cache.clear()
        assert len(cache) == 0


class TestCachePerformance:
    """Test cache performance characteristics."""
    
    def test_lru_performance(self):
        """Test LRU cache performance."""
        cache = LRUCache(capacity=1000)
        
        # Fill cache
        for i in range(1000):
            cache.put(f"key{i}", f"value{i}", 100)
        
        # Test access performance
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key{i}")
        end_time = time.time()
        
        # Should be fast (O(1) operations)
        assert end_time - start_time < 1.0  # Should complete in under 1 second
        assert cache.hits == 1000
    
    def test_lfu_performance(self):
        """Test LFU cache performance."""
        cache = LFUCache(capacity=1000)
        
        # Fill cache
        for i in range(1000):
            cache.put(f"key{i}", f"value{i}", 100)
        
        # Test access performance
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key{i}")
        end_time = time.time()
        
        # Should be reasonably fast
        assert end_time - start_time < 2.0  # Should complete in under 2 seconds
        assert cache.hits == 1000
    
    def test_ttl_performance(self):
        """Test TTL cache performance."""
        cache = TTLCache(capacity=1000, ttl_seconds=10.0)
        
        # Fill cache
        for i in range(1000):
            cache.put(f"key{i}", f"value{i}", 100)
        
        # Test access performance
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key{i}")
        end_time = time.time()
        
        # Should be fast
        assert end_time - start_time < 1.0  # Should complete in under 1 second
        assert cache.hits == 1000


class TestCacheEdgeCases:
    """Test cache edge cases and error conditions."""
    
    def test_zero_capacity(self):
        """Test cache with zero capacity."""
        with pytest.raises(ValueError):
            LRUCache(capacity=0)
        
        with pytest.raises(ValueError):
            LFUCache(capacity=0)
        
        with pytest.raises(ValueError):
            TTLCache(capacity=0)
    
    def test_negative_capacity(self):
        """Test cache with negative capacity."""
        with pytest.raises(ValueError):
            LRUCache(capacity=-1)
        
        with pytest.raises(ValueError):
            LFUCache(capacity=-1)
        
        with pytest.raises(ValueError):
            TTLCache(capacity=-1)
    
    def test_evict_empty_cache(self):
        """Test evicting from empty cache."""
        cache = LRUCache(capacity=3)
        
        with pytest.raises(RuntimeError):
            cache.evict()
        
        cache = LFUCache(capacity=3)
        
        with pytest.raises(RuntimeError):
            cache.evict()
        
        cache = TTLCache(capacity=3)
        
        with pytest.raises(RuntimeError):
            cache.evict()
    
    def test_large_values(self):
        """Test cache with large values."""
        cache = LRUCache(capacity=3)
        
        # Test with large size values
        cache.put("key1", "value1", 1000000)
        cache.put("key2", "value2", 2000000)
        
        assert len(cache) == 2
        
        entry = cache.get("key1")
        assert entry.size == 1000000
    
    def test_unicode_keys(self):
        """Test cache with unicode keys."""
        cache = LRUCache(capacity=3)
        
        cache.put("key_测试", "value1", 100)
        cache.put("key_🚀", "value2", 200)
        
        assert len(cache) == 2
        
        entry = cache.get("key_测试")
        assert entry.value == "value1"
        
        entry = cache.get("key_🚀")
        assert entry.value == "value2"
