# Method 1 Deployment Guide

*Generated on 2025-09-29 11:44:02*

## Overview

This guide covers deploying Method 1 Parameter Optimization Engine in production environments, including installation, configuration, monitoring, and maintenance.

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 18.04+, CentOS 7+, RHEL 7+), macOS 10.14+, Windows 10
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: 2 cores minimum (4+ cores recommended)
- **Storage**: 5GB free space (20GB+ for production)
- **Network**: Internet access for package installation

### Production Requirements

- **RAM**: 16GB+ (for batch processing)
- **CPU**: 8+ cores (for parallel processing)
- **Storage**: 50GB+ SSD (for logs, cache, and temporary files)
- **Network**: High-speed connection for image processing
- **Load Balancer**: For high availability deployments

## Installation Methods

### Method 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/yourorg/svg-ai.git
cd svg-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install VTracer
pip install vtracer

# Verify installation
python -c "from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer; print('Installation successful')"
```

### Method 2: Docker Installation

```bash
# Build Docker image
docker build -t optimization-engine:latest .

# Run container
docker run -d \
  --name optimization-engine \
  -p 8000:8000 \
  -v /path/to/config:/app/config \
  -v /path/to/logs:/app/logs \
  -v /path/to/data:/app/data \
  optimization-engine:latest
```

## Configuration

### Environment Configuration

Create `.env` file:
```bash
# Core settings
OPTIMIZATION_LOG_LEVEL=INFO
OPTIMIZATION_LOG_DIR=/app/logs/optimization
OPTIMIZATION_CACHE_SIZE=5000
OPTIMIZATION_ENABLE_PROFILING=false

# Performance settings
OPTIMIZATION_MAX_WORKERS=8
OPTIMIZATION_BATCH_SIZE=20
OPTIMIZATION_TIMEOUT=60

# Production settings
PYTHON_ENV=production
DEBUG=false
```

## Service Setup

### Systemd Service

Create `/etc/systemd/system/optimization-engine.service`:
```ini
[Unit]
Description=Method 1 Optimization Engine
After=network.target

[Service]
Type=simple
User=optimization
Group=optimization
WorkingDirectory=/opt/optimization-engine
Environment=PATH=/opt/optimization-engine/venv/bin
Environment=PYTHON_ENV=production
ExecStart=/opt/optimization-engine/venv/bin/python -m backend.ai_modules.optimization.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable optimization-engine
sudo systemctl start optimization-engine

# Check status
sudo systemctl status optimization-engine
```

## Monitoring and Observability

### Health Checks

```python
# health.py
from fastapi import FastAPI, HTTPException
import time

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        # Test optimization engine
        optimizer = FeatureMappingOptimizer()
        test_features = {
            "edge_density": 0.1,
            "unique_colors": 5,
            "entropy": 0.5,
            "corner_density": 0.05,
            "gradient_strength": 0.2,
            "complexity_score": 0.3
        }

        start_time = time.time()
        result = optimizer.optimize(test_features)
        processing_time = time.time() - start_time

        return {
            "status": "healthy",
            "processing_time": processing_time,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")
```

## Security

### SSL/TLS Configuration

```bash
# Generate SSL certificate (for testing)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/optimization.key \
    -out /etc/ssl/certs/optimization.crt
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/optimization"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup configuration
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /opt/optimization-engine/config/

# Backup logs (last 7 days)
find /opt/optimization-engine/logs -name "*.log*" -mtime -7 \
    -exec tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" {} \;

echo "Backup completed: $DATE"
```

## Maintenance

### Log Rotation

```bash
# /etc/logrotate.d/optimization-engine
/opt/optimization-engine/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 optimization optimization
}
```

### Update Procedure

```bash
#!/bin/bash
# update.sh

# Backup current version
./backup.sh

# Stop service
sudo systemctl stop optimization-engine

# Update code
cd /opt/optimization-engine
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Start service
sudo systemctl start optimization-engine

echo "Update completed successfully"
```

This deployment guide provides comprehensive instructions for deploying Method 1 Parameter Optimization Engine in production environments.
