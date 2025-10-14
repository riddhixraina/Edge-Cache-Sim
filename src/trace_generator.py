"""
Trace generation utilities for realistic CDN workload simulation.

This module provides generators for creating request traces that follow
real-world patterns like Zipf distributions and bursty Poisson arrivals.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import random
import time


@dataclass
class RequestTrace:
    """Represents a sequence of requests for simulation."""
    
    requests: List[str]  # List of object IDs
    timestamps: List[float]  # Request arrival times
    object_sizes: Dict[str, int]  # Size of each object in bytes
    metadata: Dict[str, Any]  # Additional trace information
    
    def __len__(self) -> int:
        """Return the number of requests in the trace."""
        return len(self.requests)
    
    def get_duration(self) -> float:
        """Get the total duration of the trace in seconds."""
        if not self.timestamps:
            return 0.0
        return max(self.timestamps) - min(self.timestamps)
    
    def get_request_rate(self) -> float:
        """Get the average request rate (requests per second)."""
        duration = self.get_duration()
        return len(self.requests) / duration if duration > 0 else 0.0


class TraceGenerator:
    """
    Generator for creating realistic request traces.
    
    Supports Zipf-distributed popularity patterns and bursty Poisson arrivals
    to model real-world CDN traffic patterns.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the trace generator.
        
        Args:
            seed: Random seed for reproducible traces
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_zipf_trace(
        self,
        num_requests: int,
        catalog_size: int,
        skew: float = 0.9,
        min_object_size: int = 1024,
        max_object_size: int = 10 * 1024 * 1024,  # 10MB
        duration_seconds: Optional[float] = None,
    ) -> RequestTrace:
        """
        Generate a trace following Zipf distribution.
        
        Zipf distribution models the power-law nature of web content popularity,
        where a small number of objects receive the majority of requests.
        
        Args:
            num_requests: Total number of requests to generate
            catalog_size: Number of unique objects in the catalog
            skew: Zipf skew parameter (α). Higher values = more skewed
            min_object_size: Minimum object size in bytes
            max_object_size: Maximum object size in bytes
            duration_seconds: Total trace duration (auto-calculated if None)
            
        Returns:
            RequestTrace object containing the generated requests
        """
        if skew <= 0:
            raise ValueError("Skew parameter must be positive")
        
        # Generate object IDs
        object_ids = [f"obj_{i:06d}" for i in range(catalog_size)]
        
        # Generate Zipf-distributed popularity
        # P(rank) = 1 / (rank^α) / H_N
        # where H_N is the normalization constant
        ranks = np.arange(1, catalog_size + 1)
        probabilities = 1.0 / (ranks ** skew)
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        # Sample requests according to Zipf distribution
        request_indices = np.random.choice(
            catalog_size, 
            size=num_requests, 
            p=probabilities
        )
        
        # Convert indices to object IDs
        requests = [object_ids[i] for i in request_indices]
        
        # Generate object sizes (log-normal distribution for realistic sizes)
        # Most objects are small, few are very large
        log_mean = np.log(min_object_size)
        log_std = 0.5
        sizes = np.random.lognormal(log_mean, log_std, catalog_size)
        sizes = np.clip(sizes, min_object_size, max_object_size).astype(int)
        
        object_sizes = {obj_id: int(sizes[i]) for i, obj_id in enumerate(object_ids)}
        
        # Generate timestamps (uniform distribution if no duration specified)
        if duration_seconds is None:
            # Estimate duration based on request rate
            avg_request_rate = 100  # requests per second
            duration_seconds = num_requests / avg_request_rate
        
        timestamps = np.sort(np.random.uniform(0, duration_seconds, num_requests))
        
        metadata = {
            "generator": "zipf",
            "num_requests": num_requests,
            "catalog_size": catalog_size,
            "skew": skew,
            "duration_seconds": duration_seconds,
            "avg_request_rate": num_requests / duration_seconds,
            "unique_objects_requested": len(set(requests)),
            "popularity_distribution": "zipf",
        }
        
        return RequestTrace(
            requests=requests,
            timestamps=timestamps.tolist(),
            object_sizes=object_sizes,
            metadata=metadata
        )
    
    def generate_bursty_trace(
        self,
        num_requests: int,
        catalog_size: int,
        base_rate: float = 100.0,  # requests per second
        burst_rate: float = 500.0,  # requests per second during bursts
        burst_probability: float = 0.1,  # Probability of burst at any time
        burst_duration: float = 10.0,  # Duration of bursts in seconds
        min_object_size: int = 1024,
        max_object_size: int = 10 * 1024 * 1024,
    ) -> RequestTrace:
        """
        Generate a trace with bursty traffic patterns.
        
        Models flash crowds, viral content, and DDoS patterns where
        traffic arrives in bursts rather than uniformly.
        
        Args:
            num_requests: Total number of requests to generate
            catalog_size: Number of unique objects in the catalog
            base_rate: Base request rate (requests per second)
            burst_rate: Request rate during bursts
            burst_probability: Probability of burst occurring
            burst_duration: Duration of each burst in seconds
            min_object_size: Minimum object size in bytes
            max_object_size: Maximum object size in bytes
            
        Returns:
            RequestTrace object containing the generated requests
        """
        # Generate object IDs and sizes
        object_ids = [f"obj_{i:06d}" for i in range(catalog_size)]
        
        # Generate object sizes (log-normal distribution)
        log_mean = np.log(min_object_size)
        log_std = 0.5
        sizes = np.random.lognormal(log_mean, log_std, catalog_size)
        sizes = np.clip(sizes, min_object_size, max_object_size).astype(int)
        object_sizes = {obj_id: int(sizes[i]) for i, obj_id in enumerate(object_ids)}
        
        # Generate bursty arrival times
        timestamps = []
        current_time = 0.0
        
        while len(timestamps) < num_requests:
            # Decide if we're in a burst or normal period
            if np.random.random() < burst_probability:
                # Burst period
                rate = burst_rate
                duration = burst_duration
            else:
                # Normal period
                rate = base_rate
                duration = np.random.exponential(1.0 / burst_probability)  # Average time between bursts
            
            # Generate Poisson arrivals for this period
            num_arrivals = np.random.poisson(rate * duration)
            arrivals = np.random.uniform(current_time, current_time + duration, num_arrivals)
            timestamps.extend(arrivals)
            
            current_time += duration
        
        # Sort timestamps and truncate to exact number of requests
        timestamps = sorted(timestamps)[:num_requests]
        
        # Generate requests (uniform distribution over catalog)
        requests = np.random.choice(object_ids, size=num_requests).tolist()
        
        duration_seconds = max(timestamps) if timestamps else 0.0
        
        metadata = {
            "generator": "bursty",
            "num_requests": num_requests,
            "catalog_size": catalog_size,
            "base_rate": base_rate,
            "burst_rate": burst_rate,
            "burst_probability": burst_probability,
            "burst_duration": burst_duration,
            "duration_seconds": duration_seconds,
            "avg_request_rate": num_requests / duration_seconds if duration_seconds > 0 else 0.0,
            "unique_objects_requested": len(set(requests)),
            "popularity_distribution": "uniform",
        }
        
        return RequestTrace(
            requests=requests,
            timestamps=timestamps,
            object_sizes=object_sizes,
            metadata=metadata
        )
    
    def generate_mixed_trace(
        self,
        num_requests: int,
        catalog_size: int,
        zipf_ratio: float = 0.7,  # 70% Zipf, 30% uniform
        zipf_skew: float = 0.9,
        burst_probability: float = 0.05,
        **kwargs
    ) -> RequestTrace:
        """
        Generate a trace mixing Zipf popularity with bursty arrivals.
        
        This creates the most realistic traces by combining:
        - Zipf-distributed object popularity
        - Bursty arrival patterns
        
        Args:
            num_requests: Total number of requests to generate
            catalog_size: Number of unique objects in the catalog
            zipf_ratio: Fraction of requests following Zipf distribution
            zipf_skew: Zipf skew parameter
            burst_probability: Probability of bursty periods
            **kwargs: Additional arguments passed to bursty generator
            
        Returns:
            RequestTrace object containing the generated requests
        """
        zipf_requests = int(num_requests * zipf_ratio)
        uniform_requests = num_requests - zipf_requests
        
        # Generate Zipf portion
        zipf_trace = self.generate_zipf_trace(
            num_requests=zipf_requests,
            catalog_size=catalog_size,
            skew=zipf_skew,
            **kwargs
        )
        
        # Generate uniform portion
        uniform_trace = self.generate_bursty_trace(
            num_requests=uniform_requests,
            catalog_size=catalog_size,
            burst_probability=burst_probability,
            **kwargs
        )
        
        # Combine traces
        all_requests = zipf_trace.requests + uniform_trace.requests
        all_timestamps = zipf_trace.timestamps + uniform_trace.timestamps
        
        # Sort by timestamp
        sorted_indices = np.argsort(all_timestamps)
        sorted_requests = [all_requests[i] for i in sorted_indices]
        sorted_timestamps = [all_timestamps[i] for i in sorted_indices]
        
        # Combine object sizes (should be identical)
        combined_object_sizes = zipf_trace.object_sizes.copy()
        combined_object_sizes.update(uniform_trace.object_sizes)
        
        metadata = {
            "generator": "mixed",
            "num_requests": num_requests,
            "catalog_size": catalog_size,
            "zipf_ratio": zipf_ratio,
            "zipf_skew": zipf_skew,
            "burst_probability": burst_probability,
            "duration_seconds": max(sorted_timestamps) if sorted_timestamps else 0.0,
            "avg_request_rate": num_requests / max(sorted_timestamps) if sorted_timestamps else 0.0,
            "unique_objects_requested": len(set(sorted_requests)),
            "popularity_distribution": "mixed",
        }
        
        return RequestTrace(
            requests=sorted_requests,
            timestamps=sorted_timestamps,
            object_sizes=combined_object_sizes,
            metadata=metadata
        )
    
    def analyze_trace(self, trace: RequestTrace) -> Dict[str, Any]:
        """
        Analyze a trace and return statistics.
        
        Args:
            trace: The trace to analyze
            
        Returns:
            Dictionary containing trace statistics
        """
        requests = trace.requests
        timestamps = trace.timestamps
        
        # Basic statistics
        total_requests = len(requests)
        unique_objects = len(set(requests))
        duration = trace.get_duration()
        avg_rate = trace.get_request_rate()
        
        # Object popularity analysis
        from collections import Counter
        object_counts = Counter(requests)
        popularity_ranks = object_counts.most_common()
        
        # Calculate Gini coefficient (measure of inequality)
        if len(popularity_ranks) > 1:
            counts = [count for _, count in popularity_ranks]
            sorted_counts = sorted(counts)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        else:
            gini = 0.0
        
        # Request rate analysis
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv = std_interval / avg_interval if avg_interval > 0 else 0.0  # Coefficient of variation
        else:
            avg_interval = 0.0
            std_interval = 0.0
            cv = 0.0
        
        return {
            "total_requests": total_requests,
            "unique_objects": unique_objects,
            "duration_seconds": duration,
            "avg_request_rate": avg_rate,
            "gini_coefficient": gini,
            "avg_inter_request_interval": avg_interval,
            "std_inter_request_interval": std_interval,
            "coefficient_of_variation": cv,
            "most_popular_object": popularity_ranks[0][0] if popularity_ranks else None,
            "most_popular_count": popularity_ranks[0][1] if popularity_ranks else 0,
            "catalog_coverage": unique_objects / trace.metadata.get("catalog_size", 1),
        }
