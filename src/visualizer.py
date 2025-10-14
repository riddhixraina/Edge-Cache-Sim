"""
Comprehensive visualization utilities for CDN simulation results.

This module provides plotting functions for analyzing simulation results,
creating publication-ready figures, and generating interactive visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

from .simulator import SimulationResult
from .metrics import MetricsAnalyzer


class CDNVisualizer:
    """
    Comprehensive visualization system for CDN simulation results.
    
    Provides static plots, interactive visualizations, and publication-ready
    figures for analyzing cache performance and system behavior.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # Create results directory if it doesn't exist
        self.results_dir = Path("results/plots")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_hit_ratio_vs_cache_size(
        self,
        results: List[SimulationResult],
        policies: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot hit ratio vs cache size for different policies.
        
        Args:
            results: List of simulation results
            policies: List of policies to include (None = all)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Filter results by policy if specified
        if policies:
            results = [r for r in results if r.config.policy in policies]
        
        # Group results by policy
        policy_groups = {}
        for result in results:
            policy = result.config.policy
            if policy not in policy_groups:
                policy_groups[policy] = []
            policy_groups[policy].append(result)
        
        # Plot each policy
        for i, (policy, policy_results) in enumerate(policy_groups.items()):
            cache_sizes = [r.config.cache_capacity for r in policy_results]
            hit_ratios = [r.aggregate_metrics["avg_hit_ratio"] for r in policy_results]
            
            # Sort by cache size
            sorted_data = sorted(zip(cache_sizes, hit_ratios))
            cache_sizes, hit_ratios = zip(*sorted_data)
            
            ax.plot(cache_sizes, hit_ratios, 
                   marker='o', linewidth=2, markersize=6,
                   label=policy, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel("Cache Size (objects)", fontsize=12)
        ax.set_ylabel("Hit Ratio", fontsize=12)
        ax.set_title("Hit Ratio vs Cache Size", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_zipf_skew_sensitivity(
        self,
        results: List[SimulationResult],
        policies: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot hit ratio vs Zipf skew parameter.
        
        Args:
            results: List of simulation results
            policies: List of policies to include (None = all)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Filter results by policy if specified
        if policies:
            results = [r for r in results if r.config.policy in policies]
        
        # Group results by policy
        policy_groups = {}
        for result in results:
            policy = result.config.policy
            if policy not in policy_groups:
                policy_groups[policy] = []
            policy_groups[policy].append(result)
        
        # Plot each policy
        for i, (policy, policy_results) in enumerate(policy_groups.items()):
            skews = []
            hit_ratios = []
            
            for result in policy_results:
                # Extract skew from trace metadata
                skew = result.trace.metadata.get("skew", 0.9)
                skews.append(skew)
                hit_ratios.append(result.aggregate_metrics["avg_hit_ratio"])
            
            # Sort by skew
            sorted_data = sorted(zip(skews, hit_ratios))
            skews, hit_ratios = zip(*sorted_data)
            
            ax.plot(skews, hit_ratios, 
                   marker='s', linewidth=2, markersize=6,
                   label=policy, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel("Zipf Skew Parameter (α)", fontsize=12)
        ax.set_ylabel("Hit Ratio", fontsize=12)
        ax.set_title("Hit Ratio vs Zipf Skew Parameter", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_series_performance(
        self,
        result: SimulationResult,
        window_size: int = 1000,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series of hit ratio and latency over time.
        
        Args:
            result: Single simulation result
            window_size: Size of sliding window for smoothing
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        results = result.results
        
        if not results:
            return fig
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Calculate sliding window metrics
        df['hit_ratio_window'] = df['hit'].rolling(window=window_size, min_periods=1).mean()
        df['latency_window'] = df['latency_ms'].rolling(window=window_size, min_periods=1).mean()
        
        # Plot hit ratio over time
        ax1.plot(df.index, df['hit_ratio_window'], 
                color='blue', linewidth=2, alpha=0.8)
        ax1.set_ylabel("Hit Ratio", fontsize=12)
        ax1.set_title("Hit Ratio Over Time", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Plot latency over time
        ax2.plot(df.index, df['latency_window'], 
                color='red', linewidth=2, alpha=0.8)
        ax2.set_xlabel("Request Number", fontsize=12)
        ax2.set_ylabel("Average Latency (ms)", fontsize=12)
        ax2.set_title("Average Latency Over Time", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_node_load_distribution(
        self,
        result: SimulationResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of node performance metrics.
        
        Args:
            result: Single simulation result
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        
        node_metrics = result.node_metrics
        
        if not node_metrics:
            return fig
        
        # Extract metrics
        node_ids = [m["node_id"] for m in node_metrics]
        hit_ratios = [m["hit_ratio"] for m in node_metrics]
        utilizations = [m["utilization"] for m in node_metrics]
        request_counts = [m["total_requests"] for m in node_metrics]
        bandwidth_saved = [m["bandwidth_saved_pct"] for m in node_metrics]
        
        # Plot hit ratios
        bars1 = ax1.bar(node_ids, hit_ratios, color=self.colors[0], alpha=0.7)
        ax1.set_ylabel("Hit Ratio", fontsize=12)
        ax1.set_title("Hit Ratio by Node", fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        # Plot utilizations
        bars2 = ax2.bar(node_ids, utilizations, color=self.colors[1], alpha=0.7)
        ax2.set_ylabel("Cache Utilization", fontsize=12)
        ax2.set_title("Cache Utilization by Node", fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Plot request counts
        bars3 = ax3.bar(node_ids, request_counts, color=self.colors[2], alpha=0.7)
        ax3.set_xlabel("Node ID", fontsize=12)
        ax3.set_ylabel("Total Requests", fontsize=12)
        ax3.set_title("Request Load by Node", fontsize=14, fontweight='bold')
        
        # Plot bandwidth saved
        bars4 = ax4.bar(node_ids, bandwidth_saved, color=self.colors[3], alpha=0.7)
        ax4.set_xlabel("Node ID", fontsize=12)
        ax4.set_ylabel("Bandwidth Saved (%)", fontsize=12)
        ax4.set_title("Bandwidth Savings by Node", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_latency_distribution(
        self,
        result: SimulationResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot latency distribution as CDF and histogram.
        
        Args:
            result: Single simulation result
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        
        results = result.results
        
        if not results:
            return fig
        
        # Separate hit and miss latencies
        hit_latencies = [r.latency_ms for r in results if r.hit]
        miss_latencies = [r.latency_ms for r in results if not r.hit]
        
        # Plot CDF
        if hit_latencies:
            sorted_hits = np.sort(hit_latencies)
            y_hits = np.arange(1, len(sorted_hits) + 1) / len(sorted_hits)
            ax1.plot(sorted_hits, y_hits, label='Cache Hits', linewidth=2, color='blue')
        
        if miss_latencies:
            sorted_misses = np.sort(miss_latencies)
            y_misses = np.arange(1, len(sorted_misses) + 1) / len(sorted_misses)
            ax1.plot(sorted_misses, y_misses, label='Cache Misses', linewidth=2, color='red')
        
        ax1.set_xlabel("Latency (ms)", fontsize=12)
        ax1.set_ylabel("Cumulative Probability", fontsize=12)
        ax1.set_title("Latency CDF", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot histogram
        ax2.hist(hit_latencies, bins=50, alpha=0.7, label='Cache Hits', 
                color='blue', density=True)
        ax2.hist(miss_latencies, bins=50, alpha=0.7, label='Cache Misses', 
                color='red', density=True)
        
        ax2.set_xlabel("Latency (ms)", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Latency Distribution", fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_object_popularity(
        self,
        result: SimulationResult,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot object popularity distribution.
        
        Args:
            result: Single simulation result
            top_n: Number of top objects to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        
        results = result.results
        
        if not results:
            return fig
        
        # Count object requests
        from collections import Counter
        object_counts = Counter(r.object_id for r in results)
        
        # Get top N objects
        top_objects = object_counts.most_common(top_n)
        objects, counts = zip(*top_objects)
        
        # Plot top objects
        bars = ax1.bar(range(len(objects)), counts, color=self.colors[0], alpha=0.7)
        ax1.set_xlabel("Object Rank", fontsize=12)
        ax1.set_ylabel("Request Count", fontsize=12)
        ax1.set_title(f"Top {top_n} Most Popular Objects", fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(objects)))
        ax1.set_xticklabels([f"#{i+1}" for i in range(len(objects))], rotation=45)
        
        # Plot popularity distribution (log scale)
        all_counts = list(object_counts.values())
        ax2.hist(all_counts, bins=50, alpha=0.7, color=self.colors[1])
        ax2.set_xlabel("Request Count", fontsize=12)
        ax2.set_ylabel("Number of Objects", fontsize=12)
        ax2.set_title("Object Popularity Distribution", fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_policy_comparison(
        self,
        results: List[SimulationResult],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of different policies across multiple metrics.
        
        Args:
            results: List of simulation results
            metrics: List of metrics to compare (None = default set)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["avg_hit_ratio", "bandwidth_saved_pct", "avg_latency_ms", "load_balance_ratio"]
        
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        axes = axes.flatten()
        
        # Group results by policy
        policy_groups = {}
        for result in results:
            policy = result.config.policy
            if policy not in policy_groups:
                policy_groups[policy] = []
            policy_groups[policy].append(result)
        
        # Plot each metric
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics
            ax = axes[i]
            
            policies = list(policy_groups.keys())
            values = []
            
            for policy in policies:
                policy_results = policy_groups[policy]
                metric_values = [r.aggregate_metrics.get(metric, 0) for r in policy_results]
                values.append(metric_values)
            
            # Create box plot
            bp = ax.boxplot(values, labels=policies, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis for percentages
            if 'ratio' in metric or 'pct' in metric:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_dashboard(
        self,
        result: SimulationResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            result: Single simulation result
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main metrics (top row, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_main_metrics(ax1, result)
        
        # Hit ratio over time (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_hit_ratio_trend(ax2, result)
        
        # Node performance (middle row)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_node_performance(ax3, result)
        
        # Latency distribution (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_latency_summary(ax4, result)
        
        # Object popularity (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_popularity_summary(ax5, result)
        
        # Configuration info (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_config_info(ax6, result)
        
        plt.suptitle("CDN Simulation Summary Dashboard", fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_main_metrics(self, ax, result: SimulationResult):
        """Plot main performance metrics."""
        metrics = result.aggregate_metrics
        
        metrics_data = [
            ("Hit Ratio", metrics.get("avg_hit_ratio", 0)),
            ("Bandwidth Saved", metrics.get("bandwidth_saved_pct", 0)),
            ("Avg Latency", metrics.get("avg_latency_ms", 0)),
            ("Load Balance", metrics.get("load_balance_ratio", 1)),
        ]
        
        labels, values = zip(*metrics_data)
        
        bars = ax.bar(labels, values, color=self.colors[:4], alpha=0.7)
        ax.set_title("Key Performance Metrics", fontsize=12, fontweight='bold')
        ax.set_ylabel("Value", fontsize=10)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if 'ratio' in bar.get_label().lower() or 'balance' in bar.get_label().lower():
                label = f'{value:.2%}'
            elif 'latency' in bar.get_label().lower():
                label = f'{value:.1f}ms'
            else:
                label = f'{value:.1f}%'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   label, ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, max(values) * 1.2)
    
    def _plot_hit_ratio_trend(self, ax, result: SimulationResult):
        """Plot hit ratio trend over time."""
        results = result.results
        if not results:
            return
        
        window_size = max(100, len(results) // 20)
        hit_ratios = []
        
        for i in range(0, len(results), window_size):
            window_results = results[i:i + window_size]
            if window_results:
                hits = sum(1 for r in window_results if r.hit)
                hit_ratio = hits / len(window_results)
                hit_ratios.append(hit_ratio)
        
        ax.plot(hit_ratios, color='blue', linewidth=2)
        ax.set_title("Hit Ratio Trend", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time Window", fontsize=10)
        ax.set_ylabel("Hit Ratio", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def _plot_node_performance(self, ax, result: SimulationResult):
        """Plot node performance comparison."""
        node_metrics = result.node_metrics
        if not node_metrics:
            return
        
        node_ids = [m["node_id"] for m in node_metrics]
        hit_ratios = [m["hit_ratio"] for m in node_metrics]
        
        bars = ax.bar(node_ids, hit_ratios, color=self.colors[0], alpha=0.7)
        ax.set_title("Hit Ratio by Node", fontsize=12, fontweight='bold')
        ax.set_xlabel("Node ID", fontsize=10)
        ax.set_ylabel("Hit Ratio", fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def _plot_latency_summary(self, ax, result: SimulationResult):
        """Plot latency summary."""
        results = result.results
        if not results:
            return
        
        latencies = [r.latency_ms for r in results]
        
        ax.hist(latencies, bins=30, color=self.colors[1], alpha=0.7)
        ax.set_title("Latency Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Latency (ms)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_popularity_summary(self, ax, result: SimulationResult):
        """Plot object popularity summary."""
        results = result.results
        if not results:
            return
        
        from collections import Counter
        object_counts = Counter(r.object_id for r in results)
        
        # Get top 10 objects
        top_objects = object_counts.most_common(10)
        objects, counts = zip(*top_objects)
        
        ax.bar(range(len(objects)), counts, color=self.colors[2], alpha=0.7)
        ax.set_title("Top 10 Objects", fontsize=12, fontweight='bold')
        ax.set_xlabel("Object Rank", fontsize=10)
        ax.set_ylabel("Requests", fontsize=10)
        ax.set_xticks(range(len(objects)))
        ax.set_xticklabels([f"#{i+1}" for i in range(len(objects))], fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_config_info(self, ax, result: SimulationResult):
        """Plot configuration information."""
        config = result.config
        
        info_text = f"""
Configuration:
• Nodes: {config.num_nodes}
• Policy: {config.policy}
• Cache Size: {config.cache_capacity}
• TTL: {config.ttl_seconds}s
• Origin Latency: {config.origin_latency_ms}ms
• Edge Latency: {config.edge_latency_ms}ms

Trace Info:
• Requests: {len(result.results):,}
• Duration: {result.trace.get_duration():.1f}s
• Rate: {result.trace.get_request_rate():.1f} req/s
• Objects: {result.trace.metadata.get('catalog_size', 0):,}
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        ax.set_title("Configuration", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def save_all_plots(self, result: SimulationResult, prefix: str = "simulation") -> Dict[str, str]:
        """
        Generate and save all standard plots for a simulation result.
        
        Args:
            result: Simulation result to plot
            prefix: Prefix for saved files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        saved_files = {}
        
        # Generate timestamp for unique filenames
        timestamp = int(result.timestamp)
        
        plots = [
            ("time_series", lambda: self.plot_time_series_performance(result)),
            ("node_load", lambda: self.plot_node_load_distribution(result)),
            ("latency_dist", lambda: self.plot_latency_distribution(result)),
            ("object_popularity", lambda: self.plot_object_popularity(result)),
            ("summary_dashboard", lambda: self.create_summary_dashboard(result)),
        ]
        
        for plot_name, plot_func in plots:
            filename = f"{prefix}_{plot_name}_{timestamp}.png"
            filepath = self.results_dir / filename
            
            try:
                fig = plot_func()
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files[plot_name] = str(filepath)
            except Exception as e:
                print(f"Error generating {plot_name}: {e}")
        
        return saved_files
