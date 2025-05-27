# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data models logs

# Download dataset and copy source code
COPY src/ ./src/
COPY data/ ./data/

# Download dataset if not present
RUN if [ ! -f "data/german.data" ]; then \
        wget -O data/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data; \
    fi

# Train model during build
RUN cd src && \
    python data_preprocessing.py && \
    python model_training.py

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application (stay in /app directory)
WORKDIR /app
CMD ["python", "src/api.py"] 