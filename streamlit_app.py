"""
Interactive Streamlit dashboard for CDN cache simulation.

This module provides a comprehensive web-based interface for running
simulations, analyzing results, and exploring different configurations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, Any, List

from src.simulator import CDNSimulator, SimulationConfig, SimulationResult
from src.trace_generator import TraceGenerator
from src.metrics import MetricsAnalyzer
from src.visualizer import CDNVisualizer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="CDN Cache Simulator",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌐 CDN Cache Simulator Dashboard")
    st.markdown("Production-grade CDN edge cache simulation and analysis tool")
    
    # Initialize session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = []
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MetricsAnalyzer()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = CDNVisualizer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Simulation Configuration")
        
        # Basic configuration
        st.subheader("Basic Settings")
        num_nodes = st.slider("Number of Edge Nodes", 2, 32, 8)
        cache_capacity = st.slider("Cache Capacity per Node", 10, 1000, 100)
        policy = st.selectbox("Cache Policy", ["LRU", "LFU", "TTL"])
        
        # Advanced configuration
        with st.expander("Advanced Settings"):
            ttl_seconds = st.number_input("TTL (seconds)", 60, 3600, 300)
            origin_latency = st.number_input("Origin Latency (ms)", 10, 500, 100)
            edge_latency = st.number_input("Edge Latency (ms)", 0.1, 10.0, 1.0)
            use_consistent_hashing = st.checkbox("Use Consistent Hashing", True)
        
        # Trace generation
        st.subheader("Trace Generation")
        trace_type = st.selectbox("Trace Type", ["Zipf", "Bursty", "Mixed"])
        
        if trace_type == "Zipf":
            num_requests = st.slider("Number of Requests", 1000, 100000, 10000)
            catalog_size = st.slider("Catalog Size", 100, 10000, 1000)
            skew = st.slider("Zipf Skew (α)", 0.6, 1.4, 0.9)
            
        elif trace_type == "Bursty":
            num_requests = st.slider("Number of Requests", 1000, 100000, 10000)
            catalog_size = st.slider("Catalog Size", 100, 10000, 1000)
            base_rate = st.slider("Base Rate (req/s)", 10, 200, 100)
            burst_rate = st.slider("Burst Rate (req/s)", 100, 1000, 500)
            burst_probability = st.slider("Burst Probability", 0.01, 0.5, 0.1)
            
        else:  # Mixed
            num_requests = st.slider("Number of Requests", 1000, 100000, 10000)
            catalog_size = st.slider("Catalog Size", 100, 10000, 1000)
            zipf_ratio = st.slider("Zipf Ratio", 0.1, 0.9, 0.7)
            zipf_skew = st.slider("Zipf Skew (α)", 0.6, 1.4, 0.9)
            burst_probability = st.slider("Burst Probability", 0.01, 0.2, 0.05)
        
        # Run simulation button
        st.subheader("Simulation Control")
        if st.button("🚀 Run Simulation", type="primary"):
            run_simulation()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Analysis", "🔍 Comparison", "📋 Export"])
    
    with tab1:
        show_results_tab()
    
    with tab2:
        show_analysis_tab()
    
    with tab3:
        show_comparison_tab()
    
    with tab4:
        show_export_tab()


def run_simulation():
    """Run a simulation with current configuration."""
    with st.spinner("Running simulation..."):
        try:
            # Get configuration from sidebar
            config = SimulationConfig(
                num_nodes=st.session_state.get('num_nodes', 8),
                cache_capacity=st.session_state.get('cache_capacity', 100),
                policy=st.session_state.get('policy', 'LRU'),
                ttl_seconds=st.session_state.get('ttl_seconds', 300),
                origin_latency_ms=st.session_state.get('origin_latency', 100),
                edge_latency_ms=st.session_state.get('edge_latency', 1.0),
                use_consistent_hashing=st.session_state.get('use_consistent_hashing', True)
            )
            
            # Generate trace
            generator = TraceGenerator(seed=42)
            
            if st.session_state.get('trace_type', 'Zipf') == "Zipf":
                trace = generator.generate_zipf_trace(
                    num_requests=st.session_state.get('num_requests', 10000),
                    catalog_size=st.session_state.get('catalog_size', 1000),
                    skew=st.session_state.get('skew', 0.9)
                )
            elif st.session_state.get('trace_type') == "Bursty":
                trace = generator.generate_bursty_trace(
                    num_requests=st.session_state.get('num_requests', 10000),
                    catalog_size=st.session_state.get('catalog_size', 1000),
                    base_rate=st.session_state.get('base_rate', 100),
                    burst_rate=st.session_state.get('burst_rate', 500),
                    burst_probability=st.session_state.get('burst_probability', 0.1)
                )
            else:  # Mixed
                trace = generator.generate_mixed_trace(
                    num_requests=st.session_state.get('num_requests', 10000),
                    catalog_size=st.session_state.get('catalog_size', 1000),
                    zipf_ratio=st.session_state.get('zipf_ratio', 0.7),
                    zipf_skew=st.session_state.get('zipf_skew', 0.9),
                    burst_probability=st.session_state.get('burst_probability', 0.05)
                )
            
            # Run simulation
            simulator = CDNSimulator(config)
            result = simulator.run(trace)
            
            # Add to session state
            st.session_state.simulation_results.append(result)
            st.session_state.analyzer.add_result(result)
            
            st.success(f"Simulation completed! Processed {len(result.results):,} requests in {result.execution_time_seconds:.2f} seconds")
            
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")


def show_results_tab():
    """Show the results tab."""
    if not st.session_state.simulation_results:
        st.info("No simulation results yet. Run a simulation from the sidebar!")
        return
    
    # Get latest result
    result = st.session_state.simulation_results[-1]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Hit Ratio",
            f"{result.aggregate_metrics['avg_hit_ratio']:.2%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Bandwidth Saved",
            f"{result.aggregate_metrics['bandwidth_saved_pct']:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Avg Latency",
            f"{result.aggregate_metrics['avg_latency_ms']:.1f}ms",
            delta=None
        )
    
    with col4:
        st.metric(
            "Load Balance",
            f"{result.aggregate_metrics['load_balance_ratio']:.3f}",
            delta=None
        )
    
    # Time series plots
    st.subheader("📈 Performance Over Time")
    
        # Hit ratio over time
    fig_hit_ratio = plot_hit_ratio_over_time(result)
    st.plotly_chart(fig_hit_ratio, use_container_width=True)
    
    # Latency over time
    fig_latency = plot_latency_over_time(result)
    st.plotly_chart(fig_latency, use_container_width=True)
    
    # Node performance
    st.subheader("🖥️ Node Performance")
    fig_nodes = plot_node_performance(result)
    st.plotly_chart(fig_nodes, use_container_width=True)


def show_analysis_tab():
    """Show the analysis tab."""
    if not st.session_state.simulation_results:
        st.info("No simulation results yet. Run a simulation from the sidebar!")
        return
    
    result = st.session_state.simulation_results[-1]
    
    # Detailed analysis
    analysis = st.session_state.analyzer.analyze_result(result)
    
    # Latency analysis
    st.subheader("⏱️ Latency Analysis")
    
    latency_data = analysis["latency_analysis"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hit Latency Statistics")
        hit_stats = latency_data["hit_latency_stats"]
        st.write(f"**Mean:** {hit_stats['mean']:.2f}ms")
        st.write(f"**Std Dev:** {hit_stats['std']:.2f}ms")
        st.write(f"**P95:** {hit_stats['percentiles'].get('p95', 0):.2f}ms")
        st.write(f"**P99:** {hit_stats['percentiles'].get('p99', 0):.2f}ms")
    
    with col2:
        st.subheader("Miss Latency Statistics")
        miss_stats = latency_data["miss_latency_stats"]
        st.write(f"**Mean:** {miss_stats['mean']:.2f}ms")
        st.write(f"**Std Dev:** {miss_stats['std']:.2f}ms")
        st.write(f"**P95:** {miss_stats['percentiles'].get('p95', 0):.2f}ms")
        st.write(f"**P99:** {miss_stats['percentiles'].get('p99', 0):.2f}ms")
    
    # Latency distribution
    fig_latency_dist = plot_latency_distribution(result)
    st.plotly_chart(fig_latency_dist, use_container_width=True)
    
    # Object popularity analysis
    st.subheader("🔥 Object Popularity Analysis")
    
    popularity_data = analysis["object_popularity"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total Unique Objects:** {popularity_data.get('total_unique_objects', 0):,}")
        st.write(f"**Gini Coefficient:** {popularity_data.get('gini_coefficient', 0):.3f}")
        st.write(f"**Top 10% Share:** {popularity_data.get('top_10_percent_share', 0):.2%}")
    
    with col2:
        if popularity_data.get('most_popular_object'):
            obj, count = popularity_data['most_popular_object']
            st.write(f"**Most Popular Object:** {obj}")
            st.write(f"**Request Count:** {count:,}")
    
    # Top objects chart
    fig_popularity = plot_object_popularity(result)
    st.plotly_chart(fig_popularity, use_container_width=True)


def show_comparison_tab():
    """Show the comparison tab."""
    if len(st.session_state.simulation_results) < 2:
        st.info("Need at least 2 simulation results to compare. Run more simulations!")
        return
    
    st.subheader("🔄 Policy Comparison")
    
    # Policy comparison chart
    fig_comparison = plot_policy_comparison(st.session_state.simulation_results)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Comparison table
    st.subheader("📊 Comparison Table")
    
    comparison_data = []
    for i, result in enumerate(st.session_state.simulation_results):
        comparison_data.append({
            "Simulation": f"#{i+1}",
            "Policy": result.config.policy,
            "Nodes": result.config.num_nodes,
            "Cache Size": result.config.cache_capacity,
            "Hit Ratio": f"{result.aggregate_metrics['avg_hit_ratio']:.2%}",
            "Bandwidth Saved": f"{result.aggregate_metrics['bandwidth_saved_pct']:.1f}%",
            "Avg Latency": f"{result.aggregate_metrics['avg_latency_ms']:.1f}ms",
            "Load Balance": f"{result.aggregate_metrics['load_balance_ratio']:.3f}",
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)


def show_export_tab():
    """Show the export tab."""
    if not st.session_state.simulation_results:
        st.info("No simulation results to export. Run a simulation first!")
        return
    
    st.subheader("📥 Export Results")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Export")
        
        if st.button("Export Latest Results to CSV"):
            result = st.session_state.simulation_results[-1]
            df = pd.DataFrame([r.to_dict() for r in result.results])
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"simulation_results_{int(result.timestamp)}.csv",
                mime="text/csv"
            )
        
        if st.button("Export Configuration to JSON"):
            result = st.session_state.simulation_results[-1]
            config_json = result.config.to_dict()
            import json
            json_str = json.dumps(config_json, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"simulation_config_{int(result.timestamp)}.json",
                mime="application/json"
            )
    
    with col2:
        st.subheader("📈 Report Generation")
        
        if st.button("Generate Summary Report"):
            result = st.session_state.simulation_results[-1]
            report = st.session_state.analyzer.generate_summary_report(result)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"simulation_report_{int(result.timestamp)}.txt",
                mime="text/plain"
            )
    
    # Clear results
    st.subheader("🗑️ Data Management")
    
    if st.button("Clear All Results", type="secondary"):
        st.session_state.simulation_results = []
        st.session_state.analyzer = MetricsAnalyzer()
        st.success("All results cleared!")


def plot_hit_ratio_over_time(result: SimulationResult) -> go.Figure:
    """Plot hit ratio over time."""
    results = result.results
    if not results:
        return go.Figure()
    
    window_size = max(100, len(results) // 20)
    hit_ratios = []
    timestamps = []
    
    for i in range(0, len(results), window_size):
        window_results = results[i:i + window_size]
        if window_results:
            hits = sum(1 for r in window_results if r.hit)
            hit_ratio = hits / len(window_results)
            hit_ratios.append(hit_ratio)
            timestamps.append(window_results[0].timestamp)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=hit_ratios,
        mode='lines+markers',
        name='Hit Ratio',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Hit Ratio Over Time",
        xaxis_title="Time",
        yaxis_title="Hit Ratio",
        yaxis=dict(tickformat='.0%'),
        hovermode='x unified'
    )
    
    return fig


def plot_latency_over_time(result: SimulationResult) -> go.Figure:
    """Plot latency over time."""
    results = result.results
    if not results:
        return go.Figure()
    
    window_size = max(100, len(results) // 20)
    avg_latencies = []
    timestamps = []
    
    for i in range(0, len(results), window_size):
        window_results = results[i:i + window_size]
        if window_results:
            avg_latency = sum(r.latency_ms for r in window_results) / len(window_results)
            avg_latencies.append(avg_latency)
            timestamps.append(window_results[0].timestamp)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=avg_latencies,
        mode='lines+markers',
        name='Average Latency',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Average Latency Over Time",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        hovermode='x unified'
    )
    
    return fig


def plot_node_performance(result: SimulationResult) -> go.Figure:
    """Plot node performance metrics."""
    node_metrics = result.node_metrics
    if not node_metrics:
        return go.Figure()
    
    node_ids = [m["node_id"] for m in node_metrics]
    hit_ratios = [m["hit_ratio"] for m in node_metrics]
    utilizations = [m["utilization"] for m in node_metrics]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Hit Ratio by Node", "Cache Utilization by Node"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=node_ids, y=hit_ratios, name="Hit Ratio", marker_color='blue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=node_ids, y=utilizations, name="Utilization", marker_color='green'),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Node Performance Metrics",
        showlegend=False,
        height=400
    )
    
    fig.update_yaxes(title_text="Hit Ratio", tickformat='.0%', row=1, col=1)
    fig.update_yaxes(title_text="Utilization", tickformat='.0%', row=1, col=2)
    
    return fig


def plot_latency_distribution(result: SimulationResult) -> go.Figure:
    """Plot latency distribution."""
    results = result.results
    if not results:
        return go.Figure()
    
    hit_latencies = [r.latency_ms for r in results if r.hit]
    miss_latencies = [r.latency_ms for r in results if not r.hit]
    
    fig = go.Figure()
    
    if hit_latencies:
        fig.add_trace(go.Histogram(
            x=hit_latencies,
            name='Cache Hits',
            opacity=0.7,
            marker_color='blue'
        ))
    
    if miss_latencies:
        fig.add_trace(go.Histogram(
            x=miss_latencies,
            name='Cache Misses',
            opacity=0.7,
            marker_color='red'
        ))
    
    fig.update_layout(
        title="Latency Distribution",
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        barmode='overlay'
    )
    
    return fig


def plot_object_popularity(result: SimulationResult) -> go.Figure:
    """Plot object popularity."""
    results = result.results
    if not results:
        return go.Figure()
    
    from collections import Counter
    object_counts = Counter(r.object_id for r in results)
    
    # Get top 20 objects
    top_objects = object_counts.most_common(20)
    objects, counts = zip(*top_objects)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"#{i+1}" for i in range(len(objects))],
        y=counts,
        marker_color='purple',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Top 20 Most Popular Objects",
        xaxis_title="Object Rank",
        yaxis_title="Request Count"
    )
    
    return fig


def plot_policy_comparison(results: List[SimulationResult]) -> go.Figure:
    """Plot policy comparison."""
    policies = [r.config.policy for r in results]
    hit_ratios = [r.aggregate_metrics['avg_hit_ratio'] for r in results]
    bandwidth_saved = [r.aggregate_metrics['bandwidth_saved_pct'] for r in results]
    avg_latencies = [r.aggregate_metrics['avg_latency_ms'] for r in results]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Hit Ratio", "Bandwidth Saved", "Average Latency"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=policies, y=hit_ratios, name="Hit Ratio", marker_color='blue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=policies, y=bandwidth_saved, name="Bandwidth Saved", marker_color='green'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=policies, y=avg_latencies, name="Avg Latency", marker_color='red'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Policy Comparison",
        showlegend=False,
        height=400
    )
    
    fig.update_yaxes(title_text="Hit Ratio", tickformat='.0%', row=1, col=1)
    fig.update_yaxes(title_text="Bandwidth Saved (%)", row=1, col=2)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=3)
    
    return fig


if __name__ == "__main__":
    main()
