"""
Time-To-Live (TTL) cache implementation.

TTL expires items a fixed time after they were cached, regardless of how often
they are read, the way a CDN honours a Cache-Control max-age. This policy suits
content with freshness requirements.
"""

from collections import OrderedDict
from typing import Optional, Dict, Any
import time

from .base import CachePolicy, CacheEntry


class TTLCache(CachePolicy):
    """
    Time-To-Live cache implementation.

    Items expire ``ttl_seconds`` after they were stored. Reads do not extend an
    item's life. When the cache is full, expired items are dropped first, then
    the oldest item.

    Time comes from the ``now`` argument of ``get``/``put`` (the simulator passes
    each request's simulated timestamp) or from the wall clock when it is omitted.

    Entries are kept in insertion order, which is also expiry order for a fixed
    TTL, so cleanup only looks at the front of the queue: O(1) amortised per
    operation instead of a full scan.
    """

    def __init__(self, capacity: int, ttl_seconds: float = 300.0) -> None:
        """
        Initialize TTL cache.

        Args:
            capacity: Maximum number of items the cache can hold
            ttl_seconds: Time-to-live for cached items in seconds (default: 5 minutes)
        """
        super().__init__(capacity)
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        self.ttl_seconds = ttl_seconds
        self._cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._expires_at: Dict[str, float] = {}
        self._expired_count = 0
        self._last_now: Optional[float] = None  # last simulated time seen

    def _clock(self, now: Optional[float]) -> float:
        """Return the time to use for this operation."""
        if now is None:
            return time.time()
        self._last_now = now
        return now

    def get(self, key: str, now: Optional[float] = None) -> Optional[CacheEntry]:
        """
        Retrieve an item from the cache.

        An expired item is removed and counted as a miss.

        Args:
            key: The key to look up
            now: Current time in seconds (simulated or wall clock)

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        current = self._clock(now)
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            return None

        if current >= self._expires_at[key]:
            self._remove(key)
            self._expired_count += 1
            self.misses += 1
            return None

        # Reads count as accesses but do not extend the item's life
        entry.access_count += 1
        self.hits += 1
        return entry

    def put(self, key: str, value: Any, size: int, now: Optional[float] = None) -> None:
        """
        Store an item in the cache.

        Expired items are dropped first. If the cache is still full, the oldest
        item is evicted.

        Args:
            key: The key to store
            value: The value to store
            size: Size of the item in bytes
            now: Current time in seconds (simulated or wall clock)
        """
        current = self._clock(now)

        # A re-fetched key gets a fresh TTL and moves to the back of the queue
        if key in self._cache:
            self._remove(key)

        self._cleanup_expired(current)

        if self.size >= self.capacity:
            self.evict()

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            size=size,
            timestamp=current,
            access_count=1,
        )
        self._expires_at[key] = current + self.ttl_seconds
        self.size += 1

    def evict(self) -> str:
        """
        Evict the oldest item from the cache.

        Returns:
            The key of the evicted item
        """
        if not self._cache:
            raise RuntimeError("Cannot evict from empty cache")

        oldest_key = next(iter(self._cache))
        self._remove(oldest_key)
        self.evictions += 1
        return oldest_key

    def _remove(self, key: str) -> None:
        """Drop a key from both indexes."""
        self._cache.pop(key)
        self._expires_at.pop(key)
        self.size -= 1

    def _is_expired(self, entry: CacheEntry, now: Optional[float] = None) -> bool:
        """
        Check if a cache entry has expired.

        Args:
            entry: The cache entry to check
            now: Current time in seconds (defaults to the wall clock)

        Returns:
            True if expired, False otherwise
        """
        current = time.time() if now is None else now
        return current >= self._expires_at.get(entry.key, entry.timestamp + self.ttl_seconds)

    def _cleanup_expired(self, now: Optional[float] = None) -> None:
        """Remove expired items from the front of the queue."""
        current = time.time() if now is None else now
        while self._cache:
            oldest_key = next(iter(self._cache))
            if current < self._expires_at[oldest_key]:
                break
            self._remove(oldest_key)
            self._expired_count += 1

    def _clear_impl(self) -> None:
        """Clear the internal cache storage."""
        self._cache.clear()
        self._expires_at.clear()
        self._expired_count = 0
        self._last_now = None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get TTL-specific metrics.

        Returns:
            Dictionary containing standard metrics plus TTL-specific data
        """
        base_metrics = super().get_metrics()

        # Ages are measured on the same clock the cache has been running on
        current_time = self._last_now if self._last_now is not None else time.time()
        if self._cache:
            ages = [current_time - entry.timestamp for entry in self._cache.values()]
            avg_age = sum(ages) / len(ages)
            max_age = max(ages)
            min_age = min(ages)
        else:
            avg_age = 0.0
            max_age = 0.0
            min_age = 0.0

        total = self.hits + self.misses
        base_metrics.update({
            "ttl_seconds": self.ttl_seconds,
            "expired_count": self._expired_count,
            "avg_age_seconds": avg_age,
            "max_age_seconds": max_age,
            "min_age_seconds": min_age,
            "expiration_rate": self._expired_count / total if total > 0 else 0.0,
        })

        return base_metrics

    def set_ttl(self, ttl_seconds: float) -> None:
        """
        Update the TTL value for future entries.

        Items already cached keep the expiry they were given.

        Args:
            ttl_seconds: New TTL value in seconds
        """
        if ttl_seconds <= 0:
            raise ValueError("TTL must be positive")
        self.ttl_seconds = ttl_seconds
