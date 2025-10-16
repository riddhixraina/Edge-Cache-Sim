# CDN Cache Simulator - Docker Deployment Guide

This guide explains how to deploy and run the CDN Cache Simulator using Docker.

## Quick Start

### Production Deployment
```bash
# Build and run the production dashboard
docker-compose up simulator

# Access the dashboard at http://localhost:8501
```

### Development Environment
```bash
# Run development environment with Jupyter
docker-compose --profile dev up dev

# Access Jupyter at http://localhost:8888
```

## Service Descriptions

### 1. Simulator (Production)
- **Port**: 8501
- **Purpose**: Main Streamlit dashboard
- **Features**: Interactive simulation, real-time analysis
- **Volumes**: Results and data directories mounted

### 2. Dev (Development)
- **Port**: 8888
- **Purpose**: Jupyter notebook environment
- **Features**: Interactive development, analysis notebooks
- **Volumes**: Full source code mounted for live editing

### 3. Origin Server (HTTP Simulation)
- **Port**: 8080
- **Purpose**: Mock origin server for realistic HTTP simulation
- **Features**: Generates synthetic objects, simulates network latency
- **Usage**: Enable with `--profile http`

### 4. Test (Testing)
- **Purpose**: Run comprehensive test suite
- **Features**: Unit tests, coverage reporting
- **Usage**: Enable with `--profile test`

### 5. Benchmark (Performance Testing)
- **Purpose**: Run performance benchmarks
- **Features**: Automated benchmarking across configurations
- **Usage**: Enable with `--profile benchmark`

## Docker Commands

### Build Images
```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build simulator

# Build with no cache
docker-compose build --no-cache
```

### Run Services
```bash
# Run production dashboard
docker-compose up simulator

# Run development environment
docker-compose --profile dev up dev

# Run with HTTP server
docker-compose --profile http up simulator origin-server

# Run tests
docker-compose --profile test up test

# Run benchmarks
docker-compose --profile benchmark up benchmark
```

### Development Workflow
```bash
# Start development environment
docker-compose --profile dev up dev

# In another terminal, run tests
docker-compose --profile test up test

# Run specific CLI commands
docker-compose run --rm simulator python cli.py run --policy LRU --nodes 8 --cache-size 100
```

## Environment Variables

### Simulator Service
- `PYTHONPATH=/app/src`: Python path configuration
- `STREAMLIT_SERVER_PORT=8501`: Streamlit port
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`: Streamlit address

### Development Service
- `PYTHONPATH=/app/src`: Python path configuration
- `JUPYTER_ENABLE_LAB=yes`: Enable JupyterLab

## Volume Mounts

### Results Directory
- **Host**: `./results`
- **Container**: `/app/results`
- **Purpose**: Store simulation results, plots, reports

### Data Directory
- **Host**: `./data`
- **Container**: `/app/data`
- **Purpose**: Store traces, objects, real workloads

### Development Mounts
- **Host**: `.` (project root)
- **Container**: `/app`
- **Purpose**: Live code editing, hot reload

## Health Checks

The simulator service includes health checks:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Command**: Checks Streamlit health endpoint

## Networking

All services run on the `cdn-simulator-network`:
- Services can communicate using service names
- Ports are exposed to host for external access
- Internal communication uses Docker networking

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check if ports are in use
   netstat -tulpn | grep :8501
   
   # Use different ports
   docker-compose up -p 8502:8501 simulator
   ```

2. **Permission Issues**
   ```bash
   # Fix volume permissions
   sudo chown -R $USER:$USER results/ data/
   ```

3. **Build Failures**
   ```bash
   # Clean build
   docker-compose build --no-cache
   
   # Check logs
   docker-compose logs simulator
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # In Docker Desktop: Settings > Resources > Memory
   ```

### Logs and Debugging

```bash
# View logs
docker-compose logs simulator
docker-compose logs -f simulator  # Follow logs

# Debug container
docker-compose exec simulator bash

# Check container status
docker-compose ps
```

## Production Deployment

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml cdn-simulator

# Scale services
docker service scale cdn-simulator_simulator=3
```

### Using Kubernetes
```bash
# Convert to Kubernetes manifests
kompose convert

# Deploy to Kubernetes
kubectl apply -f .
```

## Security Considerations

1. **Non-root User**: Production image runs as non-root user
2. **Minimal Base Image**: Uses slim Python image
3. **No Secrets**: No sensitive data in images
4. **Network Isolation**: Services run in isolated network

## Performance Optimization

1. **Multi-stage Build**: Reduces final image size
2. **Layer Caching**: Optimized Dockerfile for caching
3. **Resource Limits**: Configure memory/CPU limits
4. **Volume Optimization**: Use named volumes for data

## Monitoring

### Health Monitoring
```bash
# Check service health
docker-compose ps

# Monitor resource usage
docker stats
```

### Application Monitoring
- Streamlit dashboard includes built-in metrics
- Results are saved to mounted volumes
- Logs available via Docker logging

## Backup and Recovery

### Backup Results
```bash
# Backup results directory
tar -czf results-backup.tar.gz results/

# Restore results
tar -xzf results-backup.tar.gz
```

### Container Backup
```bash
# Save container state
docker-compose down
docker-compose up -d
```
