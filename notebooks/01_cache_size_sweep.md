# Cache Size Sweep Analysis

This notebook demonstrates how to analyze cache performance across different sizes and policies.

## Quick Start

```python
import sys
sys.path.append('src')

from simulator import CDNSimulator, SimulationConfig
from trace_generator import TraceGenerator

# Generate trace
generator = TraceGenerator(seed=42)
trace = generator.generate_zipf_trace(
    num_requests=10000,
    catalog_size=1000,
    skew=0.9
)

# Run simulation
config = SimulationConfig(
    num_nodes=8,
    cache_capacity=100,
    policy='LRU'
)

simulator = CDNSimulator(config)
result = simulator.run(trace)

print(f"Hit Ratio: {result.aggregate_metrics['avg_hit_ratio']:.2%}")
print(f"Bandwidth Saved: {result.aggregate_metrics['bandwidth_saved_pct']:.1f}%")
```

## Analysis Examples

### Cache Size Sweep
```python
cache_sizes = [10, 25, 50, 100, 200, 500]
policies = ['LRU', 'LFU', 'TTL']
results = []

for policy in policies:
    for cache_size in cache_sizes:
        config = SimulationConfig(
            num_nodes=8,
            cache_capacity=cache_size,
            policy=policy
        )
        simulator = CDNSimulator(config)
        result = simulator.run(trace)
        results.append(result)
```

### Policy Comparison
```python
from visualizer import CDNVisualizer

visualizer = CDNVisualizer()
fig = visualizer.plot_hit_ratio_vs_cache_size(results)
plt.show()
```

## Key Findings

1. **LRU** typically performs best for temporal locality workloads
2. **LFU** excels with static popularity distributions  
3. **TTL** provides consistent performance with content freshness
4. Diminishing returns start around 200-500 cache objects
5. Optimal cache size depends on workload characteristics

## Next Steps

- Run parameter sweeps with different Zipf skew values
- Analyze bursty traffic patterns
- Compare multi-node vs single-node performance
- Evaluate prefetching strategies
