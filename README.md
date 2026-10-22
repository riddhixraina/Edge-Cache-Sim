# Edge Cache Simulator

A CDN edge cache simulator for answering one question: which caching and routing choices keep the most traffic off the origin server? It replays a request trace through a set of edge nodes, routes each request with consistent hashing, and measures how often each node can answer from its own cache.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)](#testing)
[![Core coverage](https://img.shields.io/badge/core%20coverage-80%25-green.svg)](#testing)

## What it does

- **Edge nodes with pluggable caches**: LRU, LFU and TTL policies behind one `CachePolicy` interface
- **Consistent hashing** with virtual nodes to spread objects across nodes and limit reshuffling when a node fails
- **Synthetic workloads**: Zipf-distributed popularity, plus bursty Poisson arrivals
- **Metrics**: hit ratio, bytes served from the edge, per-node load, cache occupancy
- **Interfaces**: a CLI, a Streamlit dashboard and Jupyter notebooks

## Results

Measured with `python benchmarks/run_benchmarks.py`. Setup: 100,000 requests with Zipf skew 0.9 over 10,000 objects, 8 edge nodes holding 100 objects each, so the edge can hold 8% of the catalog. Averaged over seeds 1 to 3.

### Caching policy

| Policy | Edge hit ratio | Bytes served from edge |
|---|---|---|
| LRU | 52.1% | 52.7% |
| LFU | **59.5%** | **60.0%** |
| TTL (300 s) | 47.5% | 48.1% |

LFU wins by 7 points because this workload's popularity never changes, so counting past requests predicts future ones. Real traffic shifts, and pure LFU then keeps stale content, which is why production caches use hybrids such as TinyLFU. TTL behaves like FIFO here: it ignores how often an item is read, so it loses to LRU.

### Cache size (LRU)

| Objects per node | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| Edge hit ratio | 33.9% | 42.6% | 52.2% | 62.8% | 74.6% |

Each doubling of cache size adds roughly 10 points of hit ratio.

### TTL: freshness against hit ratio

| TTL (seconds) | 1 | 5 | 10 | 20 | 60 |
|---|---|---|---|---|---|
| Edge hit ratio | 20.3% | 35.4% | 42.7% | 47.5% | 47.5% |
| Removals caused by expiry | 100% | 100% | 100% | 1% | 0% |

Short TTLs keep content fresh at the cost of hit ratio. Past about 20 seconds the TTL stops mattering at this cache size: a full cache pushes items out before they get old enough to expire.

### Routing when a node fails

| One of 8 nodes removed | Objects that move to another node |
|---|---|
| Consistent hashing, 150 virtual nodes | **12.8%** |
| Modulo hashing (`hash % nodes`) | 87.9% |
| Ideal (only the failed node's share) | 12.5% |

With modulo hashing, losing one node reshuffles almost every object and empties nearly every cache at once. Virtual nodes also even out load: the busiest node carries 1.98x the average load with 1 virtual node per server, and 1.07x with 150.

### What these numbers are not

- **Not production hit ratios.** Real CDNs often serve 90% or more of static traffic from the edge, with much larger caches. Here the cache size is chosen to make the policies easy to compare.
- **No latency figures.** Hit and miss latency are fixed inputs (1 ms and 100 ms), so average latency only restates the hit ratio.
- **Simulator speed** (about 200,000 requests per second on one core) measures this Python code, not a CDN.

## Quick start

```bash
pip install -r requirements.txt

# One simulation (defaults: 8 nodes, 100 objects per node, 1,000-object catalog)
python cli.py run --policy LRU --nodes 8 --cache-size 100 --requests 50000

# Sweep one parameter: cache_size, nodes, skew or policy
python cli.py sweep --param cache_size --values 10,50,100,500 --policy LRU
python cli.py sweep --param skew --values 0.6,0.9,1.2

# Compare policies across request counts and node counts
python cli.py benchmark --requests 1000 10000 100000 --nodes 2 4 8

# Reproduce the Results section
python benchmarks/run_benchmarks.py

# Dashboard
streamlit run streamlit_app.py
```

With Docker: `docker-compose up simulator`, then open http://localhost:8501. See [docs/docker_deployment.md](docs/docker_deployment.md).

## Programmatic use

```python
from src.simulator import CDNSimulator, SimulationConfig
from src.trace_generator import TraceGenerator

trace = TraceGenerator(seed=42).generate_zipf_trace(
    num_requests=100_000, catalog_size=10_000, skew=0.9
)
result = CDNSimulator(SimulationConfig(num_nodes=8, cache_capacity=100, policy="LFU")).run(trace)

print(f"Hit ratio: {result.aggregate_metrics['avg_hit_ratio']:.2%}")
print(f"Bytes served from edge: {result.aggregate_metrics['bandwidth_saved_pct']:.1f}%")
```

Caches run on the trace's simulated clock: each node passes the request's timestamp to `get` and `put`, so a TTL means simulated seconds, not wall-clock seconds.

## Architecture

```
Trace generator (Zipf / bursty)
        |
        v
Simulator --consistent hash--> Edge node 0 ... Edge node N
                                (LRU / LFU / TTL cache each)
        |
        v
Metrics: hit ratio, bytes from edge, per-node load
        |
        v
CLI / Streamlit dashboard / notebooks
```

## Testing

```bash
python -m pytest tests            # 88 tests
python -m pytest tests --cov=src  # with coverage
```

Coverage is 80% for the core engine (cache policies, routing, nodes, simulator, trace generation and metrics). The plotting code in `visualizer.py` has no tests, which brings the repository-wide figure down to 54%.

| Module | Coverage |
|---|---|
| `cache/ttl.py` | 97% |
| `node.py` | 96% |
| `trace_generator.py` | 96% |
| `cache/lru.py` | 95% |
| `simulator.py` | 92% |
| `cache/lfu.py` | 82% |
| `utils/hashing.py` | 59% |
| `metrics.py` | 24% |

## Project structure

```
src/
  cache/             LRU, LFU and TTL policies on a shared base class
  simulator.py       Orchestrates nodes and routing
  node.py            Edge node: one cache plus metrics
  trace_generator.py Zipf and bursty workloads
  utils/hashing.py   Consistent hash ring with virtual nodes
  metrics.py         Analysis helpers
  visualizer.py      Plots
benchmarks/          Script that reproduces the Results section
tests/               pytest suite
notebooks/           Cache size sweep notebook
cli.py               Command-line interface
streamlit_app.py     Dashboard
```

## Known limitations

- Workloads are synthetic. Popularity is fixed for the whole trace, so results favour LFU more than real traffic would.
- `enable_prefetch` and `partition_strategy` are placeholders in the config: `src/prefetch` and `src/partition` define interfaces that the simulator does not use yet.
- Caches count objects, not bytes, so object size does not affect eviction.
- The TTL is one value for the whole cache, not per object as with HTTP `Cache-Control` headers.

## Changelog

- **TTL fixed.** It used to check expiry against the wall clock, so in a run lasting about a second nothing ever expired and it returned exactly the same results as LRU. Reads also reset an item's timer, so popular content never expired. TTL now uses simulated time, counts from when an item was cached, and finds expired items in O(1) amortised time instead of scanning the cache.
- **Reproducible traces.** Generators seeded the global NumPy random state, so two generators with the same seed could produce different traces. Each generator now has its own random stream.
- **Fixed** a crash on zero-request traces, a wrong request count in the run summary, and `cli.py sweep`, which crashed and ignored skew values.
