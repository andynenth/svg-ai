# Multi-stage build for production
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

# Install Python dependencies
FROM base as dependencies
COPY requirements.txt requirements_ai_phase1.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_ai_phase1.txt

# Production stage
FROM dependencies as production
USER app

# Copy application code
COPY --chown=app:app . .

# Set environment variables
ENV PYTHONPATH=/home/app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start application
CMD ["python", "-m", "backend.app"]