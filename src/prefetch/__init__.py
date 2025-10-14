"""
Prefetching strategies for intelligent cache warming.

This module implements various prefetching algorithms to improve
cache hit ratios through predictive content loading.
"""

from .base import PrefetchStrategy

__all__ = [
    "PrefetchStrategy",
]
