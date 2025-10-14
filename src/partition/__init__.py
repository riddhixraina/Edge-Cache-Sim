"""
Cache partitioning strategies for multi-node systems.

This module implements various approaches to distributing cache
capacity across multiple edge nodes.
"""

from .static import StaticPartitioner

__all__ = [
    "StaticPartitioner",
]
