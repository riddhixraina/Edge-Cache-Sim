"""
Comprehensive metrics analysis and reporting system.

This module provides tools for analyzing simulation results, generating
statistics, and creating detailed reports.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import time

from .simulator import SimulationResult, RequestResult


@dataclass
class MetricsSummary:
    """Summary of key performance metrics."""
    
    hit_ratio: float
    bandwidth_saved_pct: float
    avg_latency_ms: float
    total_requests: int
    execution_time_seconds: float
    load_balance_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hit_ratio": self.hit_ratio,
            "bandwidth_saved_pct": self.bandwidth_saved_pct,
            "avg_latency_ms": self.avg_latency_ms,
            "total_requests": self.total_requests,
            "execution_time_seconds": self.execution_time_seconds,
            "load_balance_ratio": self.load_balance_ratio,
        }


class MetricsAnalyzer:
    """
    Comprehensive metrics analyzer for CDN simulation results.
    
    Provides detailed analysis of simulation results including performance
    metrics, statistical analysis, and comparative studies.
    """
    
    def __init__(self) -> None:
        """Initialize the metrics analyzer."""
        self.results_history: List[SimulationResult] = []
    
    def add_result(self, result: SimulationResult) -> None:
        """
        Add a simulation result for analysis.
        
        Args:
            result: SimulationResult to analyze
        """
        self.results_history.append(result)
    
    def analyze_result(self, result: SimulationResult) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single simulation result.
        
        Args:
            result: SimulationResult to analyze
            
        Returns:
            Dictionary containing detailed analysis
        """
        analysis = {
            "basic_metrics": self._analyze_basic_metrics(result),
            "latency_analysis": self._analyze_latency(result),
            "hit_pattern_analysis": self._analyze_hit_patterns(result),
            "node_performance": self._analyze_node_performance(result),
            "temporal_analysis": self._analyze_temporal_patterns(result),
            "object_popularity": self._analyze_object_popularity(result),
        }
        
        return analysis
    
    def _analyze_basic_metrics(self, result: SimulationResult) -> Dict[str, Any]:
        """Analyze basic performance metrics."""
        metrics = result.aggregate_metrics
        
        return {
            "hit_ratio": metrics.get("avg_hit_ratio", 0.0),
            "bandwidth_saved_pct": metrics.get("bandwidth_saved_pct", 0.0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0.0),
            "total_requests": metrics.get("total_requests", 0),
            "execution_time_seconds": result.execution_time_seconds,
            "requests_per_second": metrics.get("total_requests", 0) / result.execution_time_seconds,
            "load_balance_ratio": metrics.get("load_balance_ratio", 1.0),
        }
    
    def _analyze_latency(self, result: SimulationResult) -> Dict[str, Any]:
        """Analyze latency patterns and statistics."""
        results = result.results
        
        if not results:
            return {}
        
        hit_latencies = [r.latency_ms for r in results if r.hit]
        miss_latencies = [r.latency_ms for r in results if not r.hit]
        all_latencies = [r.latency_ms for r in results]
        
        def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
            if not latencies:
                return {}
            return {
                "p50": np.percentile(latencies, 50),
                "p90": np.percentile(latencies, 90),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "p99.9": np.percentile(latencies, 99.9),
            }
        
        return {
            "hit_latency_stats": {
                "mean": np.mean(hit_latencies) if hit_latencies else 0.0,
                "std": np.std(hit_latencies) if hit_latencies else 0.0,
                "min": np.min(hit_latencies) if hit_latencies else 0.0,
                "max": np.max(hit_latencies) if hit_latencies else 0.0,
                "percentiles": calculate_percentiles(hit_latencies),
            },
            "miss_latency_stats": {
                "mean": np.mean(miss_latencies) if miss_latencies else 0.0,
                "std": np.std(miss_latencies) if miss_latencies else 0.0,
                "min": np.min(miss_latencies) if miss_latencies else 0.0,
                "max": np.max(miss_latencies) if miss_latencies else 0.0,
                "percentiles": calculate_percentiles(miss_latencies),
            },
            "overall_latency_stats": {
                "mean": np.mean(all_latencies),
                "std": np.std(all_latencies),
                "min": np.min(all_latencies),
                "max": np.max(all_latencies),
                "percentiles": calculate_percentiles(all_latencies),
            },
        }
    
    def _analyze_hit_patterns(self, result: SimulationResult) -> Dict[str, Any]:
        """Analyze cache hit patterns and efficiency."""
        results = result.results
        
        if not results:
            return {}
        
        # Calculate hit ratio over time windows
        window_size = max(100, len(results) // 20)  # 20 windows
        hit_ratios_over_time = []
        
        for i in range(0, len(results), window_size):
            window_results = results[i:i + window_size]
            if window_results:
                hits = sum(1 for r in window_results if r.hit)
                hit_ratio = hits / len(window_results)
                hit_ratios_over_time.append(hit_ratio)
        
        # Analyze hit streaks
        hit_streaks = []
        current_streak = 0
        
        for r in results:
            if r.hit:
                current_streak += 1
            else:
                if current_streak > 0:
                    hit_streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            hit_streaks.append(current_streak)
        
        return {
            "hit_ratio_over_time": hit_ratios_over_time,
            "hit_streak_stats": {
                "mean": np.mean(hit_streaks) if hit_streaks else 0.0,
                "max": np.max(hit_streaks) if hit_streaks else 0,
                "total_streaks": len(hit_streaks),
            },
            "hit_ratio_stability": {
                "std": np.std(hit_ratios_over_time) if hit_ratios_over_time else 0.0,
                "min": np.min(hit_ratios_over_time) if hit_ratios_over_time else 0.0,
                "max": np.max(hit_ratios_over_time) if hit_ratios_over_time else 0.0,
            },
        }
    
    def _analyze_node_performance(self, result: SimulationResult) -> Dict[str, Any]:
        """Analyze performance across different nodes."""
        node_metrics = result.node_metrics
        
        if not node_metrics:
            return {}
        
        # Extract metrics for analysis
        hit_ratios = [m["hit_ratio"] for m in node_metrics]
        utilizations = [m["utilization"] for m in node_metrics]
        request_counts = [m["total_requests"] for m in node_metrics]
        
        return {
            "hit_ratio_distribution": {
                "mean": np.mean(hit_ratios),
                "std": np.std(hit_ratios),
                "min": np.min(hit_ratios),
                "max": np.max(hit_ratios),
                "coefficient_of_variation": np.std(hit_ratios) / np.mean(hit_ratios) if np.mean(hit_ratios) > 0 else 0.0,
            },
            "utilization_distribution": {
                "mean": np.mean(utilizations),
                "std": np.std(utilizations),
                "min": np.min(utilizations),
                "max": np.max(utilizations),
            },
            "load_distribution": {
                "mean": np.mean(request_counts),
                "std": np.std(request_counts),
                "min": np.min(request_counts),
                "max": np.max(request_counts),
                "load_balance_ratio": np.min(request_counts) / np.max(request_counts) if np.max(request_counts) > 0 else 1.0,
            },
            "node_details": [
                {
                    "node_id": m["node_id"],
                    "hit_ratio": m["hit_ratio"],
                    "utilization": m["utilization"],
                    "total_requests": m["total_requests"],
                    "bandwidth_saved_pct": m["bandwidth_saved_pct"],
                }
                for m in node_metrics
            ],
        }
    
    def _analyze_temporal_patterns(self, result: SimulationResult) -> Dict[str, Any]:
        """Analyze temporal patterns in the simulation."""
        results = result.results
        
        if not results:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([r.to_dict() for r in results])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Analyze patterns over time
        df['time_window'] = df['timestamp'].dt.floor('1min')  # 1-minute windows
        
        temporal_stats = df.groupby('time_window').agg({
            'hit': ['mean', 'count'],
            'latency_ms': ['mean', 'std'],
            'object_size': ['sum', 'mean'],
        }).round(4)
        
        # Flatten column names
        temporal_stats.columns = ['_'.join(col).strip() for col in temporal_stats.columns]
        
        return {
            "temporal_data": temporal_stats.to_dict('index'),
            "request_rate_over_time": df.groupby('time_window').size().to_dict(),
            "hit_ratio_over_time": df.groupby('time_window')['hit'].mean().to_dict(),
        }
    
    def _analyze_object_popularity(self, result: SimulationResult) -> Dict[str, Any]:
        """Analyze object popularity patterns."""
        results = result.results
        
        if not results:
            return {}
        
        # Count requests per object
        from collections import Counter
        object_counts = Counter(r.object_id for r in results)
        
        # Calculate popularity statistics
        counts = list(object_counts.values())
        
        if counts:
            # Calculate Gini coefficient for inequality
            sorted_counts = sorted(counts)
            n = len(sorted_counts)
            cumsum = np.cumsum(sorted_counts)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
            
            # Calculate concentration ratios
            total_requests = sum(counts)
            top_10_pct = int(0.1 * len(counts))
            top_10_requests = sum(sorted_counts[-top_10_pct:]) if top_10_pct > 0 else 0
            
            return {
                "total_unique_objects": len(object_counts),
                "most_popular_object": object_counts.most_common(1)[0] if object_counts else None,
                "top_10_objects": object_counts.most_common(10),
                "gini_coefficient": gini,
                "top_10_percent_share": top_10_requests / total_requests if total_requests > 0 else 0.0,
                "popularity_stats": {
                    "mean": np.mean(counts),
                    "std": np.std(counts),
                    "min": np.min(counts),
                    "max": np.max(counts),
                },
            }
        
        return {}
    
    def compare_results(self, result_ids: List[int]) -> Dict[str, Any]:
        """
        Compare multiple simulation results.
        
        Args:
            result_ids: List of result indices to compare
            
        Returns:
            Dictionary containing comparative analysis
        """
        if not result_ids or any(i >= len(self.results_history) for i in result_ids):
            return {}
        
        results = [self.results_history[i] for i in result_ids]
        
        comparison = {
            "configurations": [r.config.to_dict() for r in results],
            "basic_metrics": [self._analyze_basic_metrics(r) for r in results],
            "hit_ratios": [r.aggregate_metrics.get("avg_hit_ratio", 0.0) for r in results],
            "bandwidth_saved": [r.aggregate_metrics.get("bandwidth_saved_pct", 0.0) for r in results],
            "avg_latencies": [r.aggregate_metrics.get("avg_latency_ms", 0.0) for r in results],
        }
        
        return comparison
    
    def generate_summary_report(self, result: SimulationResult) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            result: SimulationResult to summarize
            
        Returns:
            Formatted summary report
        """
        analysis = self.analyze_result(result)
        basic = analysis["basic_metrics"]
        latency = analysis["latency_analysis"]
        node_perf = analysis["node_performance"]
        
        report = f"""
CDN Simulation Summary Report
=============================

Configuration:
  Nodes: {result.config.num_nodes}
  Policy: {result.config.policy}
  Cache Capacity: {result.config.cache_capacity} objects/node
  Total Cache Capacity: {result.config.cache_capacity * result.config.num_nodes} objects

Performance Metrics:
  Hit Ratio: {basic['hit_ratio']:.2%}
  Bandwidth Saved: {basic['bandwidth_saved_pct']:.1f}%
  Average Latency: {basic['avg_latency_ms']:.1f}ms
  Load Balance Ratio: {basic['load_balance_ratio']:.3f}

Latency Analysis:
  Hit Latency (P50/P95/P99): {latency['hit_latency_stats']['percentiles'].get('p50', 0):.1f}/{latency['hit_latency_stats']['percentiles'].get('p95', 0):.1f}/{latency['hit_latency_stats']['percentiles'].get('p99', 0):.1f}ms
  Miss Latency (P50/P95/P99): {latency['miss_latency_stats']['percentiles'].get('p50', 0):.1f}/{latency['miss_latency_stats']['percentiles'].get('p95', 0):.1f}/{latency['miss_latency_stats']['percentiles'].get('p99', 0):.1f}ms

Node Performance:
  Hit Ratio CV: {node_perf['hit_ratio_distribution']['coefficient_of_variation']:.3f}
  Utilization Range: {node_perf['utilization_distribution']['min']:.2%} - {node_perf['utilization_distribution']['max']:.2%}
  Load Balance: {node_perf['load_distribution']['load_balance_ratio']:.3f}

Execution:
  Total Requests: {basic['total_requests']:,}
  Execution Time: {basic['execution_time_seconds']:.2f}s
  Throughput: {basic['requests_per_second']:.0f} req/s
"""
        
        return report
    
    def export_to_csv(self, result: SimulationResult, filepath: str) -> None:
        """
        Export simulation results to CSV file.
        
        Args:
            result: SimulationResult to export
            filepath: Path to save CSV file
        """
        df = pd.DataFrame([r.to_dict() for r in result.results])
        df.to_csv(filepath, index=False)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all stored results.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results_history:
            return {}
        
        hit_ratios = [r.aggregate_metrics.get("avg_hit_ratio", 0.0) for r in self.results_history]
        bandwidth_saved = [r.aggregate_metrics.get("bandwidth_saved_pct", 0.0) for r in self.results_history]
        latencies = [r.aggregate_metrics.get("avg_latency_ms", 0.0) for r in self.results_history]
        
        return {
            "num_simulations": len(self.results_history),
            "hit_ratio_stats": {
                "mean": np.mean(hit_ratios),
                "std": np.std(hit_ratios),
                "min": np.min(hit_ratios),
                "max": np.max(hit_ratios),
            },
            "bandwidth_saved_stats": {
                "mean": np.mean(bandwidth_saved),
                "std": np.std(bandwidth_saved),
                "min": np.min(bandwidth_saved),
                "max": np.max(bandwidth_saved),
            },
            "latency_stats": {
                "mean": np.mean(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
            },
        }
