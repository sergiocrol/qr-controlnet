# apps/controlnet/Dockerfile

# Base stage with common dependencies
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Development stage with hot reloading
FROM base AS development

# Set environment for development
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=server.py \
    FLASK_ENV=development \
    PYTHONPATH=/app \
    DEBUG=True \
    PREFERRED_DEVICE=cpu \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /results

# Expose port
EXPOSE 8080

# Command for development (will be overridden in docker-compose.dev.yml)
CMD ["python", "server.py"]

# Production stage
FROM base AS production

# Set environment for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=server.py \
    PYTHONPATH=/app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for results
RUN mkdir -p /results

# Expose port
EXPOSE 8080

# Set up volume for results
VOLUME ["/results"]

# Command to run the application
CMD ["python", "server.py"]