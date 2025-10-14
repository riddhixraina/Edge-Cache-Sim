"""
Utility modules for the CDN simulator.

This package contains helper functions for hashing, statistics,
logging, and other common operations.
"""

from .hashing import ConsistentHasher

__all__ = [
    "ConsistentHasher",
]
