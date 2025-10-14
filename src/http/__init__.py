"""
HTTP integration components for real-world simulation.

This module provides mock origin servers and HTTP-enabled
edge nodes for realistic network simulation.
"""

from .origin_server import OriginServer

__all__ = [
    "OriginServer",
]
