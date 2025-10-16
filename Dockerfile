# Multi-stage Dockerfile for CDN Cache Simulator

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    black \
    mypy \
    pylint

# Copy source code
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY test_runner.py ./
COPY pyproject.toml ./

# Create results directory
RUN mkdir -p results/csv results/plots results/gifs results/reports

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 3: Testing image
FROM development as testing

# Run tests
RUN python test_runner.py

# Stage 4: Production image
FROM base as production

# Copy application code
COPY src/ ./src/
COPY streamlit_app.py ./
COPY cli.py ./

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Create results directory
RUN mkdir -p results/csv results/plots results/gifs results/reports

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

# Default command for production
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
