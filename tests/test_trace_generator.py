"""
Unit tests for trace generation functionality.

Tests Zipf, Poisson, and mixed trace generation with statistical
validation and edge case handling.
"""

import pytest
import numpy as np
from collections import Counter

from src.trace_generator import TraceGenerator, RequestTrace


class TestTraceGenerator:
    """Test TraceGenerator class."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = TraceGenerator()
        assert generator.seed is None
        
        generator = TraceGenerator(seed=42)
        assert generator.seed == 42
    
    def test_zipf_trace_generation(self):
        """Test Zipf trace generation."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=100,
            skew=0.9
        )
        
        assert len(trace) == 1000
        assert len(trace.requests) == 1000
        assert len(trace.timestamps) == 1000
        assert len(trace.object_sizes) == 100
        
        # Check metadata
        assert trace.metadata["generator"] == "zipf"
        assert trace.metadata["num_requests"] == 1000
        assert trace.metadata["catalog_size"] == 100
        assert trace.metadata["skew"] == 0.9
        
        # Check timestamps are sorted
        assert trace.timestamps == sorted(trace.timestamps)
        
        # Check duration
        assert trace.get_duration() > 0
        assert trace.get_request_rate() > 0
    
    def test_zipf_popularity_distribution(self):
        """Test Zipf popularity distribution."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=10000,
            catalog_size=100,
            skew=1.0
        )
        
        # Count object requests
        object_counts = Counter(trace.requests)
        
        # Check that most popular objects have more requests
        sorted_counts = sorted(object_counts.values(), reverse=True)
        
        # Top 10% should have significantly more requests than bottom 10%
        top_10_count = len(sorted_counts) // 10
        bottom_10_count = len(sorted_counts) // 10
        
        top_10_avg = sum(sorted_counts[:top_10_count]) / top_10_count
        bottom_10_avg = sum(sorted_counts[-bottom_10_count:]) / bottom_10_count
        
        assert top_10_avg > bottom_10_avg * 2  # At least 2x difference
    
    def test_zipf_skew_parameter(self):
        """Test Zipf skew parameter effects."""
        generator = TraceGenerator(seed=42)
        
        # Generate traces with different skew values
        trace_low_skew = generator.generate_zipf_trace(
            num_requests=5000,
            catalog_size=100,
            skew=0.6
        )
        
        trace_high_skew = generator.generate_zipf_trace(
            num_requests=5000,
            catalog_size=100,
            skew=1.4
        )
        
        # Count object requests
        counts_low = Counter(trace_low_skew.requests)
        counts_high = Counter(trace_high_skew.requests)
        
        # Higher skew should result in more concentrated requests
        sorted_low = sorted(counts_low.values(), reverse=True)
        sorted_high = sorted(counts_high.values(), reverse=True)
        
        # Top 10 objects should have more requests with higher skew
        top_10_low = sum(sorted_low[:10])
        top_10_high = sum(sorted_high[:10])
        
        assert top_10_high > top_10_low
    
    def test_bursty_trace_generation(self):
        """Test bursty trace generation."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_bursty_trace(
            num_requests=1000,
            catalog_size=100,
            base_rate=50.0,
            burst_rate=200.0,
            burst_probability=0.1
        )
        
        assert len(trace) == 1000
        assert len(trace.requests) == 1000
        assert len(trace.timestamps) == 1000
        assert len(trace.object_sizes) == 100
        
        # Check metadata
        assert trace.metadata["generator"] == "bursty"
        assert trace.metadata["num_requests"] == 1000
        assert trace.metadata["catalog_size"] == 100
        assert trace.metadata["base_rate"] == 50.0
        assert trace.metadata["burst_rate"] == 200.0
        assert trace.metadata["burst_probability"] == 0.1
        
        # Check timestamps are sorted
        assert trace.timestamps == sorted(trace.timestamps)
    
    def test_bursty_temporal_patterns(self):
        """Test bursty temporal patterns."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_bursty_trace(
            num_requests=5000,
            catalog_size=100,
            base_rate=100.0,
            burst_rate=500.0,
            burst_probability=0.2
        )
        
        # Calculate inter-arrival times
        timestamps = trace.timestamps
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Should have variation in intervals (bursts vs normal periods)
        assert np.std(intervals) > 0
        
        # Some intervals should be very short (bursts)
        min_interval = min(intervals)
        assert min_interval < 0.01  # Some requests within 10ms
    
    def test_mixed_trace_generation(self):
        """Test mixed trace generation."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_mixed_trace(
            num_requests=1000,
            catalog_size=100,
            zipf_ratio=0.7,
            zipf_skew=0.9,
            burst_probability=0.05
        )
        
        assert len(trace) == 1000
        assert len(trace.requests) == 1000
        assert len(trace.timestamps) == 1000
        
        # Check metadata
        assert trace.metadata["generator"] == "mixed"
        assert trace.metadata["num_requests"] == 1000
        assert trace.metadata["catalog_size"] == 100
        assert trace.metadata["zipf_ratio"] == 0.7
        assert trace.metadata["zipf_skew"] == 0.9
        assert trace.metadata["burst_probability"] == 0.05
        
        # Check timestamps are sorted
        assert trace.timestamps == sorted(trace.timestamps)
    
    def test_object_sizes(self):
        """Test object size generation."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=100,
            min_object_size=1024,
            max_object_size=1024*1024  # 1MB
        )
        
        sizes = list(trace.object_sizes.values())
        
        # All sizes should be within bounds
        assert all(1024 <= size <= 1024*1024 for size in sizes)
        
        # Should have variation in sizes
        assert min(sizes) < max(sizes)
        
        # Most objects should be small (log-normal distribution)
        median_size = np.median(sizes)
        assert median_size < max(sizes) * 0.5
    
    def test_trace_duration(self):
        """Test trace duration calculation."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=100,
            duration_seconds=60.0
        )
        
        duration = trace.get_duration()
        assert abs(duration - 60.0) < 1.0  # Within 1 second
        
        request_rate = trace.get_request_rate()
        assert abs(request_rate - 1000/60.0) < 1.0  # Within 1 req/s


class TestRequestTrace:
    """Test RequestTrace class."""
    
    def test_trace_creation(self):
        """Test trace creation."""
        requests = ["obj1", "obj2", "obj1", "obj3"]
        timestamps = [0.0, 1.0, 2.0, 3.0]
        object_sizes = {"obj1": 100, "obj2": 200, "obj3": 300}
        metadata = {"test": "data"}
        
        trace = RequestTrace(
            requests=requests,
            timestamps=timestamps,
            object_sizes=object_sizes,
            metadata=metadata
        )
        
        assert len(trace) == 4
        assert trace.requests == requests
        assert trace.timestamps == timestamps
        assert trace.object_sizes == object_sizes
        assert trace.metadata == metadata
    
    def test_trace_duration(self):
        """Test duration calculation."""
        trace = RequestTrace(
            requests=["obj1", "obj2"],
            timestamps=[0.0, 5.0],
            object_sizes={"obj1": 100, "obj2": 200},
            metadata={}
        )
        
        assert trace.get_duration() == 5.0
    
    def test_trace_request_rate(self):
        """Test request rate calculation."""
        trace = RequestTrace(
            requests=["obj1", "obj2", "obj3"],
            timestamps=[0.0, 1.0, 2.0],
            object_sizes={"obj1": 100, "obj2": 200, "obj3": 300},
            metadata={}
        )
        
        assert trace.get_request_rate() == 1.5  # 3 requests / 2 seconds
    
    def test_empty_trace(self):
        """Test empty trace."""
        trace = RequestTrace(
            requests=[],
            timestamps=[],
            object_sizes={},
            metadata={}
        )
        
        assert len(trace) == 0
        assert trace.get_duration() == 0.0
        assert trace.get_request_rate() == 0.0


class TestTraceAnalysis:
    """Test trace analysis functionality."""
    
    def test_trace_analysis(self):
        """Test trace analysis."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=5000,
            catalog_size=100,
            skew=0.9
        )
        
        analysis = generator.analyze_trace(trace)
        
        assert analysis["total_requests"] == 5000
        assert analysis["unique_objects"] > 0
        assert analysis["duration_seconds"] > 0
        assert analysis["avg_request_rate"] > 0
        assert 0 <= analysis["gini_coefficient"] <= 1
        assert analysis["catalog_coverage"] > 0
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        generator = TraceGenerator(seed=42)
        
        # Create trace with highly skewed distribution
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=10,
            skew=1.5  # High skew
        )
        
        analysis = generator.analyze_trace(trace)
        
        # High skew should result in high Gini coefficient
        assert analysis["gini_coefficient"] > 0.5
    
    def test_popularity_analysis(self):
        """Test popularity analysis."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=50,
            skew=0.9
        )
        
        analysis = generator.analyze_trace(trace)
        
        assert analysis["most_popular_object"] is not None
        assert analysis["most_popular_count"] > 0
        assert analysis["catalog_coverage"] <= 1.0


class TestTraceEdgeCases:
    """Test trace generation edge cases."""
    
    def test_zero_requests(self):
        """Test trace with zero requests."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=0,
            catalog_size=100
        )
        
        assert len(trace) == 0
        assert trace.get_duration() == 0.0
        assert trace.get_request_rate() == 0.0
    
    def test_single_request(self):
        """Test trace with single request."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1,
            catalog_size=100
        )
        
        assert len(trace) == 1
        assert len(trace.requests) == 1
        assert len(trace.timestamps) == 1
        assert trace.get_duration() >= 0.0
    
    def test_single_object_catalog(self):
        """Test trace with single object in catalog."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=100,
            catalog_size=1
        )
        
        assert len(trace) == 100
        assert len(set(trace.requests)) == 1  # All requests for same object
        assert len(trace.object_sizes) == 1
    
    def test_invalid_skew(self):
        """Test invalid skew parameter."""
        generator = TraceGenerator(seed=42)
        
        with pytest.raises(ValueError):
            generator.generate_zipf_trace(
                num_requests=100,
                catalog_size=100,
                skew=0.0  # Invalid skew
            )
        
        with pytest.raises(ValueError):
            generator.generate_zipf_trace(
                num_requests=100,
                catalog_size=100,
                skew=-1.0  # Invalid skew
            )
    
    def test_large_catalog(self):
        """Test trace with large catalog."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=10000
        )
        
        assert len(trace) == 1000
        assert len(trace.object_sizes) == 10000
        
        # Should have reasonable number of unique objects requested
        unique_objects = len(set(trace.requests))
        assert unique_objects > 0
        assert unique_objects <= 1000  # Can't request more than we have requests
    
    def test_very_short_duration(self):
        """Test trace with very short duration."""
        generator = TraceGenerator(seed=42)
        
        trace = generator.generate_zipf_trace(
            num_requests=1000,
            catalog_size=100,
            duration_seconds=0.1  # Very short duration
        )
        
        assert len(trace) == 1000
        assert trace.get_duration() <= 0.1
        assert trace.get_request_rate() >= 10000  # Very high rate


class TestTraceReproducibility:
    """Test trace generation reproducibility."""
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same trace."""
        generator1 = TraceGenerator(seed=42)
        generator2 = TraceGenerator(seed=42)
        
        trace1 = generator1.generate_zipf_trace(
            num_requests=100,
            catalog_size=50,
            skew=0.9
        )
        
        trace2 = generator2.generate_zipf_trace(
            num_requests=100,
            catalog_size=50,
            skew=0.9
        )
        
        assert trace1.requests == trace2.requests
        assert trace1.timestamps == trace2.timestamps
        assert trace1.object_sizes == trace2.object_sizes
    
    def test_different_seeds(self):
        """Test that different seeds produce different traces."""
        generator1 = TraceGenerator(seed=42)
        generator2 = TraceGenerator(seed=123)
        
        trace1 = generator1.generate_zipf_trace(
            num_requests=100,
            catalog_size=50,
            skew=0.9
        )
        
        trace2 = generator2.generate_zipf_trace(
            num_requests=100,
            catalog_size=50,
            skew=0.9
        )
        
        # Should be different (very high probability)
        assert trace1.requests != trace2.requests
        assert trace1.timestamps != trace2.timestamps
