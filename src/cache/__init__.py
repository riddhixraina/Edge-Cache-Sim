"""
Cache policy implementations for the CDN simulator.

This module provides abstract base classes and concrete implementations
for various caching algorithms including LRU, LFU, and TTL.
"""

from .base import CachePolicy, CacheEntry
from .lru import LRUCache
from .lfu import LFUCache
from .ttl import TTLCache

__all__ = [
    "CachePolicy",
    "CacheEntry",
    "LRUCache",
    "LFUCache", 
    "TTLCache",
]
