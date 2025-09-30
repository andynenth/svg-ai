# Multi-stage Production Dockerfile for SVG-AI Optimization System
# Optimized for production deployment with security and performance

# Build stage
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_ai_phase1.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_ai_phase1.txt

# Production stage
FROM python:3.9-slim as production

# Create non-root user for security
RUN groupadd -g 1001 svgai && \
    useradd -r -u 1001 -g svgai svgai

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY requirements.txt ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/cache /app/logs /app/uploads /app/models && \
    chown -R svgai:svgai /app

# Switch to non-root user
USER svgai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp/claude

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]