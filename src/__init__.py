"""
Edge Cache Simulator - A production-grade CDN cache simulation system.

This package provides comprehensive tools for simulating distributed CDN edge caches,
evaluating caching policies, and analyzing performance under realistic workloads.
"""

__version__ = "1.0.0"
__author__ = "Edge Cache Simulator Team"
__email__ = "team@edge-cache-sim.com"

from .simulator import CDNSimulator, SimulationConfig
from .node import EdgeNode
from .trace_generator import TraceGenerator
from .metrics import MetricsAnalyzer

__all__ = [
    "CDNSimulator",
    "SimulationConfig", 
    "EdgeNode",
    "TraceGenerator",
    "MetricsAnalyzer",
]
