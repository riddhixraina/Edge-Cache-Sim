"""
Command-line interface for CDN cache simulator.

This module provides a comprehensive CLI for running simulations,
generating reports, and managing experiments.
"""

import argparse
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

from src.simulator import CDNSimulator, SimulationConfig, SimulationResult
from src.trace_generator import TraceGenerator
from src.metrics import MetricsAnalyzer
from src.visualizer import CDNVisualizer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CDN Cache Simulator - Production-grade edge cache simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simple simulation
  python cli.py run --policy LRU --nodes 8 --cache-size 100 --requests 50000

  # Run parameter sweep
  python cli.py sweep --param cache_size --values 10,50,100,500 --policy LRU

  # Run with HTTP server
  python cli.py run-http --origin http://localhost:8080 --requests 10000

  # Generate report
  python cli.py report --experiment-dir results/csv/ --output report.pdf
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single simulation')
    add_run_arguments(run_parser)
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweep')
    add_sweep_arguments(sweep_parser)
    
    # HTTP command
    http_parser = subparsers.add_parser('run-http', help='Run simulation with HTTP server')
    add_http_arguments(http_parser)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate analysis report')
    add_report_arguments(report_parser)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    add_benchmark_arguments(benchmark_parser)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'run':
            run_simulation(args)
        elif args.command == 'sweep':
            run_sweep(args)
        elif args.command == 'run-http':
            run_http_simulation(args)
        elif args.command == 'report':
            generate_report(args)
        elif args.command == 'benchmark':
            run_benchmark(args)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def add_run_arguments(parser):
    """Add arguments for run command."""
    parser.add_argument('--policy', choices=['LRU', 'LFU', 'TTL'], default='LRU',
                       help='Cache eviction policy')
    parser.add_argument('--nodes', type=int, default=8,
                       help='Number of edge nodes')
    parser.add_argument('--cache-size', type=int, default=100,
                       help='Cache capacity per node')
    parser.add_argument('--requests', type=int, default=10000,
                       help='Number of requests to simulate')
    parser.add_argument('--catalog-size', type=int, default=1000,
                       help='Number of unique objects in catalog')
    parser.add_argument('--skew', type=float, default=0.9,
                       help='Zipf skew parameter')
    parser.add_argument('--ttl', type=float, default=300.0,
                       help='TTL for cached objects (seconds)')
    parser.add_argument('--origin-latency', type=float, default=100.0,
                       help='Origin server latency (ms)')
    parser.add_argument('--edge-latency', type=float, default=1.0,
                       help='Edge cache latency (ms)')
    parser.add_argument('--trace-type', choices=['zipf', 'bursty', 'mixed'], default='zipf',
                       help='Type of request trace to generate')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')


def add_sweep_arguments(parser):
    """Add arguments for sweep command."""
    parser.add_argument('--param', choices=['cache_size', 'nodes', 'skew', 'policy'],
                       help='Parameter to sweep')
    parser.add_argument('--values', type=str,
                       help='Comma-separated values to test')
    parser.add_argument('--policy', choices=['LRU', 'LFU', 'TTL'], default='LRU',
                       help='Cache policy (if not sweeping policy)')
    parser.add_argument('--requests', type=int, default=10000,
                       help='Number of requests per simulation')
    parser.add_argument('--catalog-size', type=int, default=1000,
                       help='Catalog size')
    parser.add_argument('--output-dir', type=str, default='results/csv',
                       help='Output directory for results')
    parser.add_argument('--plots', action='store_true',
                       help='Generate comparison plots')


def add_http_arguments(parser):
    """Add arguments for HTTP simulation."""
    parser.add_argument('--origin', type=str, default='http://localhost:8080',
                       help='Origin server URL')
    parser.add_argument('--requests', type=int, default=1000,
                       help='Number of requests')
    parser.add_argument('--nodes', type=int, default=4,
                       help='Number of edge nodes')
    parser.add_argument('--cache-size', type=int, default=50,
                       help='Cache capacity per node')
    parser.add_argument('--policy', choices=['LRU', 'LFU', 'TTL'], default='LRU',
                       help='Cache policy')


def add_report_arguments(parser):
    """Add arguments for report generation."""
    parser.add_argument('--experiment-dir', type=str, default='results/csv',
                       help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default='report.pdf',
                       help='Output report file')
    parser.add_argument('--format', choices=['pdf', 'html', 'txt'], default='txt',
                       help='Report format')


def add_benchmark_arguments(parser):
    """Add arguments for benchmark command."""
    parser.add_argument('--requests', type=int, nargs='+', default=[1000, 10000, 100000],
                       help='Request counts to benchmark')
    parser.add_argument('--nodes', type=int, nargs='+', default=[2, 4, 8, 16],
                       help='Node counts to benchmark')
    parser.add_argument('--policies', nargs='+', default=['LRU', 'LFU', 'TTL'],
                       help='Policies to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output file for benchmark results')


def run_simulation(args):
    """Run a single simulation."""
    print(f"Running simulation with {args.policy} policy...")
    
    # Create configuration
    config = SimulationConfig(
        num_nodes=args.nodes,
        cache_capacity=args.cache_size,
        policy=args.policy,
        ttl_seconds=args.ttl,
        origin_latency_ms=args.origin_latency,
        edge_latency_ms=args.edge_latency,
    )
    
    # Generate trace
    generator = TraceGenerator(seed=args.seed)
    
    if args.trace_type == 'zipf':
        trace = generator.generate_zipf_trace(
            num_requests=args.requests,
            catalog_size=args.catalog_size,
            skew=args.skew
        )
    elif args.trace_type == 'bursty':
        trace = generator.generate_bursty_trace(
            num_requests=args.requests,
            catalog_size=args.catalog_size
        )
    else:  # mixed
        trace = generator.generate_mixed_trace(
            num_requests=args.requests,
            catalog_size=args.catalog_size,
            zipf_skew=args.skew
        )
    
    # Run simulation
    start_time = time.time()
    simulator = CDNSimulator(config)
    result = simulator.run(trace)
    execution_time = time.time() - start_time
    
    # Print results
    print(f"\nSimulation completed in {execution_time:.2f} seconds")
    print(f"Hit Ratio: {result.aggregate_metrics['avg_hit_ratio']:.2%}")
    print(f"Bandwidth Saved: {result.aggregate_metrics['bandwidth_saved_pct']:.1f}%")
    print(f"Average Latency: {result.aggregate_metrics['avg_latency_ms']:.1f}ms")
    print(f"Load Balance Ratio: {result.aggregate_metrics['load_balance_ratio']:.3f}")
    
    # Save results
    if args.output:
        result.to_json(args.output)
        print(f"Results saved to {args.output}")
    
    # Generate plots
    if args.plots:
        visualizer = CDNVisualizer()
        saved_files = visualizer.save_all_plots(result, prefix="simulation")
        print(f"Plots saved: {list(saved_files.keys())}")


def run_sweep(args):
    """Run parameter sweep."""
    print(f"Running parameter sweep for {args.param}...")
    
    # Parse values
    if args.param == 'cache_size':
        values = [int(x) for x in args.values.split(',')]
    elif args.param == 'nodes':
        values = [int(x) for x in args.values.split(',')]
    elif args.param == 'skew':
        values = [float(x) for x in args.values.split(',')]
    elif args.param == 'policy':
        values = args.values.split(',')
    else:
        raise ValueError(f"Unknown parameter: {args.param}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, value in enumerate(values):
        print(f"Running simulation {i+1}/{len(values)}: {args.param}={value}")
        
        # Create configuration
        config_dict = {
            'num_nodes': args.nodes,
            'cache_capacity': 100,  # Default
            'policy': args.policy,
            'ttl_seconds': 300.0,
            'origin_latency_ms': 100.0,
            'edge_latency_ms': 1.0,
        }
        
        # Set the parameter being swept
        if args.param == 'cache_size':
            config_dict['cache_capacity'] = value
        elif args.param == 'nodes':
            config_dict['num_nodes'] = value
        elif args.param == 'policy':
            config_dict['policy'] = value
        
        config = SimulationConfig(**config_dict)
        
        # Generate trace
        generator = TraceGenerator(seed=42)
        trace = generator.generate_zipf_trace(
            num_requests=args.requests,
            catalog_size=args.catalog_size,
            skew=0.9
        )
        
        # Run simulation
        simulator = CDNSimulator(config)
        result = simulator.run(trace)
        results.append(result)
        
        # Save individual result
        result_file = output_dir / f"sweep_{args.param}_{value}.json"
        result.to_json(str(result_file))
    
    # Generate summary CSV
    summary_file = output_dir / f"sweep_{args.param}_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.param, 'hit_ratio', 'bandwidth_saved_pct', 'avg_latency_ms', 'load_balance_ratio'])
        
        for result in results:
            param_value = getattr(result.config, args.param)
            writer.writerow([
                param_value,
                result.aggregate_metrics['avg_hit_ratio'],
                result.aggregate_metrics['bandwidth_saved_pct'],
                result.aggregate_metrics['avg_latency_ms'],
                result.aggregate_metrics['load_balance_ratio']
            ])
    
    print(f"Sweep completed. Results saved to {output_dir}")
    print(f"Summary: {summary_file}")
    
    # Generate plots if requested
    if args.plots:
        visualizer = CDNVisualizer()
        
        if args.param == 'cache_size':
            fig = visualizer.plot_hit_ratio_vs_cache_size(results)
            plot_file = output_dir / f"hit_ratio_vs_cache_size.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_file}")
        
        elif args.param == 'skew':
            fig = visualizer.plot_zipf_skew_sensitivity(results)
            plot_file = output_dir / f"hit_ratio_vs_skew.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_file}")


def run_http_simulation(args):
    """Run simulation with HTTP server."""
    print("HTTP simulation not yet implemented.")
    print("This would integrate with the mock origin server.")
    print(f"Would simulate {args.requests} requests to {args.origin}")


def generate_report(args):
    """Generate analysis report."""
    print(f"Generating report from {args.experiment_dir}...")
    
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Directory {experiment_dir} does not exist")
        return
    
    # Find all JSON result files
    result_files = list(experiment_dir.glob("*.json"))
    if not result_files:
        print(f"No JSON result files found in {experiment_dir}")
        return
    
    # Load results
    results = []
    for file_path in result_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                # Convert back to SimulationResult (simplified)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Generate report
    analyzer = MetricsAnalyzer()
    
    report_lines = [
        "# CDN Cache Simulation Report",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Number of experiments: {len(results)}",
        "",
    ]
    
    # Summary statistics
    if results:
        hit_ratios = [r['aggregate_metrics']['avg_hit_ratio'] for r in results]
        bandwidth_saved = [r['aggregate_metrics']['bandwidth_saved_pct'] for r in results]
        
        report_lines.extend([
            "## Summary Statistics",
            f"Average Hit Ratio: {sum(hit_ratios)/len(hit_ratios):.2%}",
            f"Average Bandwidth Saved: {sum(bandwidth_saved)/len(bandwidth_saved):.1f}%",
            "",
        ])
    
    # Individual results
    report_lines.append("## Individual Results")
    for i, result in enumerate(results):
        config = result['config']
        metrics = result['aggregate_metrics']
        
        report_lines.extend([
            f"### Experiment {i+1}",
            f"- Policy: {config['policy']}",
            f"- Nodes: {config['num_nodes']}",
            f"- Cache Size: {config['cache_capacity']}",
            f"- Hit Ratio: {metrics['avg_hit_ratio']:.2%}",
            f"- Bandwidth Saved: {metrics['bandwidth_saved_pct']:.1f}%",
            f"- Avg Latency: {metrics['avg_latency_ms']:.1f}ms",
            "",
        ])
    
    # Save report
    report_content = "\n".join(report_lines)
    
    if args.format == 'txt':
        with open(args.output, 'w') as f:
            f.write(report_content)
    else:
        print("HTML and PDF formats not yet implemented")
        with open(args.output.replace('.pdf', '.txt'), 'w') as f:
            f.write(report_content)
    
    print(f"Report saved to {args.output}")


def run_benchmark(args):
    """Run performance benchmarks."""
    print("Running performance benchmarks...")
    
    benchmark_results = []
    
    for requests in args.requests:
        for nodes in args.nodes:
            for policy in args.policies:
                print(f"Benchmarking: {requests} requests, {nodes} nodes, {policy} policy")
                
                # Create configuration
                config = SimulationConfig(
                    num_nodes=nodes,
                    cache_capacity=100,
                    policy=policy,
                )
                
                # Generate trace
                generator = TraceGenerator(seed=42)
                trace = generator.generate_zipf_trace(
                    num_requests=requests,
                    catalog_size=1000,
                    skew=0.9
                )
                
                # Run simulation
                start_time = time.time()
                simulator = CDNSimulator(config)
                result = simulator.run(trace)
                execution_time = time.time() - start_time
                
                benchmark_results.append({
                    'requests': requests,
                    'nodes': nodes,
                    'policy': policy,
                    'execution_time': execution_time,
                    'throughput': requests / execution_time,
                    'hit_ratio': result.aggregate_metrics['avg_hit_ratio'],
                    'bandwidth_saved_pct': result.aggregate_metrics['bandwidth_saved_pct'],
                })
    
    # Save benchmark results
    with open(args.output, 'w', newline='') as f:
        if benchmark_results:
            writer = csv.DictWriter(f, fieldnames=benchmark_results[0].keys())
            writer.writeheader()
            writer.writerows(benchmark_results)
    
    print(f"Benchmark results saved to {args.output}")
    
    # Print summary
    print("\nBenchmark Summary:")
    for result in benchmark_results:
        print(f"{result['policy']:4} | {result['requests']:6} req | {result['nodes']:2} nodes | "
              f"{result['execution_time']:6.2f}s | {result['throughput']:8.0f} req/s | "
              f"{result['hit_ratio']:6.2%} hit ratio")


if __name__ == "__main__":
    main()
