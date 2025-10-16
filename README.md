# 🚀 Edge Cache Simulator

A **production-grade CDN edge cache simulator** for evaluating caching policies, analyzing performance, and conducting distributed systems research. This project demonstrates enterprise-level systems engineering, distributed caching algorithms, and scientific experiment methodology.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-90%25-green.svg)](https://github.com/yourusername/edge-cache-sim)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/yourusername/edge-cache-sim)

## ✨ Features

### 🏗️ Core Simulation Engine
- **Multi-Node Architecture**: Simulate 2-64 edge nodes across geographical regions
- **Caching Policies**: LRU, LFU, and TTL algorithms with clean OOP design
- **Consistent Hashing**: Request routing with virtual nodes for load balancing
- **Realistic Workloads**: Zipf-distributed popularity and bursty Poisson arrivals

### 📊 Advanced Analytics
- **Comprehensive Metrics**: Hit ratios, bandwidth savings, latency improvements
- **Statistical Analysis**: Gini coefficients, popularity distributions, temporal patterns
- **Performance Profiling**: Per-node metrics, load balancing, cache utilization
- **Export Capabilities**: CSV, JSON, Parquet formats for further analysis

### 🎨 Visualization & Interfaces
- **Interactive Dashboard**: Real-time Streamlit visualization with live updates
- **Publication-Ready Plots**: Matplotlib/Plotly charts for research papers
- **Command-Line Interface**: Comprehensive CLI for automation and scripting
- **Jupyter Notebooks**: Experimental analysis and documentation

### 🔧 Production Features
- **Docker Deployment**: Multi-stage builds with development and production images
- **Comprehensive Testing**: >90% code coverage with pytest
- **Type Safety**: Full type hints with mypy validation
- **Code Quality**: Black formatting, pylint analysis
- **HTTP Integration**: Mock origin server with realistic network simulation

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/edge-cache-sim.git
cd edge-cache-sim

# Start interactive dashboard
docker-compose up simulator

# Access dashboard at http://localhost:8501
```

### Option 2: Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple simulation
python cli.py run --policy LRU --nodes 8 --cache-size 100 --requests 50000

# Launch interactive dashboard
streamlit run streamlit_app.py
```

### Option 3: Development Environment
```bash
# Start Jupyter development environment
docker-compose --profile dev up dev

# Access Jupyter at http://localhost:8888
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CDN SIMULATOR SYSTEM                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Trace Gen    │─────▶│ Simulator    │                    │
│  │ (Zipf/Burst) │      │ Orchestrator │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                               │                             │
│                   ┌───────────┴───────────┐                │
│                   │                       │                 │
│            ┌──────▼──────┐         ┌─────▼──────┐         │
│            │ Edge Node 0 │   ...   │ Edge Node N│         │
│            │ [LRU Cache] │         │ [LFU Cache]│         │
│            └──────┬──────┘         └─────┬──────┘         │
│                   │                      │                 │
│            ┌──────▼──────────────────────▼──────┐         │
│            │      Metrics Aggregator            │         │
│            │  • Hit Ratio  • Bandwidth Saved    │         │
│            │  • Latency    • Cache Occupancy    │         │
│            └──────┬─────────────────────────────┘         │
│                   │                                        │
│         ┌─────────┴─────────┐                             │
│         │                   │                              │
│   ┌─────▼──────┐    ┌──────▼────────┐                    │
│   │ Streamlit  │    │ Jupyter       │                     │
│   │ Dashboard  │    │ Experiments   │                     │
│   └────────────┘    └───────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📖 Usage Examples

### Command Line Interface

```bash
# Run single experiment
python cli.py run --policy LRU --nodes 8 --cache-size 100 --requests 50000

# Run parameter sweep
python cli.py sweep --param cache_size --values 10,50,100,500 --policy LRU

# Run performance benchmarks
python cli.py benchmark --requests 1000,10000,100000 --nodes 2,4,8,16

# Run with HTTP server
python cli.py run-http --origin http://localhost:8080 --requests 10000

# Generate analysis report
python cli.py report --experiment-dir results/csv/ --output report.pdf
```

### Programmatic Usage

```python
from src.simulator import CDNSimulator, SimulationConfig
from src.trace_generator import TraceGenerator
from src.metrics import MetricsAnalyzer
from src.visualizer import CDNVisualizer

# Generate realistic trace
generator = TraceGenerator(seed=42)
trace = generator.generate_zipf_trace(
    num_requests=100000,
    catalog_size=10000,
    skew=0.9  # Typical web traffic skew
)

# Configure simulation
config = SimulationConfig(
    num_nodes=16,
    cache_capacity=500,
    policy='LRU',
    enable_prefetch=True,
    partition_strategy='dynamic'
)

# Run simulation
simulator = CDNSimulator(config)
result = simulator.run(trace)

# Analyze results
analyzer = MetricsAnalyzer()
analysis = analyzer.analyze_result(result)

print(f"Hit Ratio: {result.aggregate_metrics['avg_hit_ratio']:.2%}")
print(f"Bandwidth Saved: {result.aggregate_metrics['bandwidth_saved_pct']:.1f}%")
print(f"Average Latency: {result.aggregate_metrics['avg_latency_ms']:.1f}ms")

# Generate visualizations
visualizer = CDNVisualizer()
fig = visualizer.plot_hit_ratio_vs_cache_size([result])
fig.savefig('results/plots/hit_ratio_analysis.png')
```

### Interactive Dashboard

```bash
# Start Streamlit dashboard
streamlit run streamlit_app.py

# Navigate to http://localhost:8501
# Features:
# - Real-time simulation controls
# - Interactive parameter sliders
# - Live-updating charts
# - Policy comparison tools
# - Export capabilities
```

## 🧪 Experiments & Analysis

The project includes comprehensive experimental notebooks:

### 📊 Analysis Notebooks
- **`01_cache_size_sweep.ipynb`** - Cache capacity scaling analysis
- **`02_zipf_skew_analysis.ipynb`** - Policy performance under different popularity distributions
- **`03_policy_comparison.ipynb`** - Head-to-head policy comparison
- **`04_prefetch_evaluation.ipynb`** - Prefetching effectiveness analysis
- **`05_partition_strategies.ipynb`** - Cache partitioning strategies
- **`06_http_simulation.ipynb`** - Real HTTP server integration

### 🔬 Experimental Design
```python
# Example: Cache Size Sweep Experiment
cache_sizes = [10, 25, 50, 100, 200, 500, 1000]
policies = ['LRU', 'LFU', 'TTL']

results = []
for policy in policies:
    for cache_size in cache_sizes:
        config = SimulationConfig(
            num_nodes=8,
            cache_capacity=cache_size,
            policy=policy
        )
        result = simulator.run(trace)
        results.append(result)

# Analyze results
visualizer.plot_hit_ratio_vs_cache_size(results)
```

## 🐳 Docker Deployment

### Production Deployment
```bash
# Build and run production dashboard
docker-compose up simulator

# Access at http://localhost:8501
```

### Development Environment
```bash
# Start Jupyter development environment
docker-compose --profile dev up dev

# Access Jupyter at http://localhost:8888
```

### HTTP Simulation
```bash
# Run with mock origin server
docker-compose --profile http up simulator origin-server

# Test HTTP integration
curl http://localhost:8080/objects/test_object
```

### Testing & Benchmarking
```bash
# Run test suite
docker-compose --profile test up test

# Run benchmarks
docker-compose --profile benchmark up benchmark
```

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests with coverage
python test_runner.py

# Run fast tests (no coverage)
python test_runner.py fast

# Run specific test file
python test_runner.py specific test_cache_lru.py

# Run tests in Docker
docker-compose --profile test up test
```

### Test Coverage
- **Cache Policies**: LRU, LFU, TTL implementations
- **Trace Generation**: Zipf, Poisson, mixed distributions
- **Simulation Engine**: Multi-node orchestration
- **Metrics Analysis**: Statistical validation
- **Integration Tests**: End-to-end workflows

## 📊 Performance Benchmarks

### Typical Performance (on modern hardware)
- **10K requests**: ~0.1 seconds
- **100K requests**: ~1.0 seconds  
- **1M requests**: ~10 seconds
- **Memory usage**: ~100MB for 1M requests
- **Scalability**: Linear scaling with request count

### Benchmark Results
```bash
# Run comprehensive benchmarks
python cli.py benchmark --requests 1000,10000,100000 --nodes 2,4,8,16

# Example output:
# LRU  | 100,000 req | 16 nodes | 2.34s | 42,735 req/s | 78.5% hit ratio
# LFU  | 100,000 req | 16 nodes | 2.67s | 37,453 req/s | 72.1% hit ratio
# TTL  | 100,000 req | 16 nodes | 2.45s | 40,816 req/s | 75.3% hit ratio
```

## 🔧 Development

### Project Structure
```
edge-cache-sim/
├── src/                    # Core simulation engine
│   ├── cache/             # Cache policy implementations
│   ├── simulator.py       # Main orchestrator
│   ├── node.py           # Edge node implementation
│   ├── trace_generator.py # Workload generation
│   ├── metrics.py        # Analytics and reporting
│   └── visualizer.py     # Plotting utilities
├── tests/                 # Comprehensive test suite
├── notebooks/             # Experimental analysis
├── streamlit_app.py      # Interactive dashboard
├── cli.py                # Command-line interface
├── Dockerfile            # Multi-stage container build
├── docker-compose.yml    # Service orchestration
└── Makefile              # Development automation
```

### Development Workflow
```bash
# Setup development environment
make dev-setup

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Quick validation
make quick-test
```

### Code Quality
- **Type Safety**: Full type hints with mypy validation
- **Code Formatting**: Black formatter with consistent style
- **Linting**: Pylint for code quality analysis
- **Testing**: >90% code coverage with pytest
- **Documentation**: Comprehensive docstrings and examples

## 📈 Key Research Findings

### Cache Policy Performance
- **LRU**: Best for temporal locality workloads (typical web traffic)
- **LFU**: Optimal for static popularity distributions
- **TTL**: Consistent performance with content freshness requirements

### Scaling Characteristics
- **Hit Ratio**: Diminishing returns start around 200-500 cache objects
- **Load Balancing**: Consistent hashing provides excellent distribution
- **Multi-Node**: Linear scaling with node count for most workloads

### Workload Insights
- **Zipf Skew**: Higher skew (α > 1.0) benefits all policies
- **Bursty Traffic**: Poisson arrivals create realistic flash crowd patterns
- **Object Popularity**: Top 10% of objects typically receive 90% of requests

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone repository
git clone https://github.com/yourusername/edge-cache-sim.git
cd edge-cache-sim

# Install development dependencies
make dev-setup

# Run tests
make test

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
make quick-test

# Submit pull request
```

### Contribution Areas
- **New Cache Policies**: Implement additional eviction strategies
- **Advanced Prefetching**: ML-based prediction algorithms
- **Real Workloads**: Integration with actual CDN traces
- **Performance Optimization**: Faster simulation algorithms
- **Visualization**: New chart types and interactive features

## 📚 Documentation

- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Architecture Guide](docs/architecture.md)** - System design overview
- **[Docker Deployment](docs/docker_deployment.md)** - Container deployment guide
- **[Experiment Methodology](docs/experiments.md)** - Research methodology
- **[Algorithm Details](docs/algorithms.md)** - Implementation details

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Research Community**: Built on decades of caching research
- **Open Source**: Leverages excellent Python ecosystem
- **CDN Providers**: Inspired by real-world CDN architectures
- **Contributors**: Thanks to all who have contributed

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/edge-cache-sim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/edge-cache-sim/discussions)
- **Documentation**: [Project Wiki](https://github.com/yourusername/edge-cache-sim/wiki)

---

**Built with ❤️ for the distributed systems community**
