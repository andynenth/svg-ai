# Deployment and Configuration Guide

## Overview

This guide covers deployment, configuration, and production setup for the SVG-AI Converter system. The system supports both local development and production deployment scenarios with comprehensive caching, monitoring, and performance optimization.

## System Requirements

### Minimum Requirements

- **Operating System:** Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python:** 3.9+ (required for VTracer compatibility)
- **Memory:** 4GB RAM minimum, 8GB recommended
- **Storage:** 2GB free space for dependencies, additional space for cache and uploads
- **CPU:** Multi-core processor recommended for parallel processing

### Recommended Production Environment

- **Operating System:** Ubuntu 22.04 LTS or CentOS 8+
- **Python:** 3.9 or 3.10 (avoid 3.11+ due to VTracer compatibility)
- **Memory:** 16GB+ RAM for high-throughput processing
- **Storage:** SSD with 50GB+ free space
- **CPU:** 8+ cores for optimal parallel processing
- **Network:** Stable internet connection for dependencies

## Installation

### 1. Environment Setup

#### Using Virtual Environment (Recommended)

```bash
# Create Python 3.9 virtual environment
python3.9 -m venv venv39
source venv39/bin/activate  # Linux/macOS
# venv39\Scripts\activate   # Windows

# Verify Python version
python --version  # Should show Python 3.9.x
```

#### Using Conda

```bash
# Create conda environment with Python 3.9
conda create -n svg-ai python=3.9
conda activate svg-ai
```

### 2. Core Dependencies

```bash
# Install VTracer (requires temp directory workaround on macOS)
export TMPDIR=/tmp  # macOS only
pip install vtracer

# Install core requirements
pip install -r requirements.txt

# Verify VTracer installation
python -c "import vtracer; print('VTracer installed successfully')"
```

### 3. Optional AI Dependencies

For AI-enhanced conversion capabilities:

```bash
# Install AI dependencies
pip install -r requirements_ai_phase1.txt

# Or use the installation script
./scripts/install_ai_dependencies.sh

# Verify AI setup
python3 scripts/verify_ai_setup.py
```

### 4. Web Interface Dependencies

```bash
# Install Flask and web dependencies
pip install flask flask-cors werkzeug pillow

# For production deployment
pip install gunicorn  # Linux/macOS
pip install waitress  # Windows alternative
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=false
FLASK_PORT=8001

# File Upload Configuration
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
UPLOAD_FOLDER=uploads

# AI Configuration
AI_ENABLED=true
AI_TIMEOUT=10.0
AI_CACHE_SIZE=1000

# Cache Configuration
CACHE_ENABLED=true
CACHE_SIZE_LIMIT=1073741824  # 1GB in bytes
CACHE_TTL=3600  # 1 hour in seconds

# Security Configuration
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Performance Configuration
PARALLEL_WORKERS=4
MAX_BATCH_SIZE=10
PROCESSING_TIMEOUT=300  # 5 minutes

# Monitoring Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/svg-ai.log
METRICS_ENABLED=true
```

### Configuration Files

#### config.py

```python
import os
from pathlib import Path

class Config:
    # Basic Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

    # Upload configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

    # AI configuration
    AI_ENABLED = os.environ.get('AI_ENABLED', 'true').lower() == 'true'
    AI_TIMEOUT = float(os.environ.get('AI_TIMEOUT', 10.0))

    # Cache configuration
    CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_SIZE_LIMIT = int(os.environ.get('CACHE_SIZE_LIMIT', 1024 * 1024 * 1024))

    # Performance configuration
    PARALLEL_WORKERS = int(os.environ.get('PARALLEL_WORKERS', 4))
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 10))

class DevelopmentConfig(Config):
    DEBUG = True
    AI_ENABLED = True

class ProductionConfig(Config):
    DEBUG = False
    # Production-specific settings

class TestingConfig(Config):
    TESTING = True
    AI_ENABLED = False  # Disable AI for faster testing
```

### Logging Configuration

#### logging.conf

```ini
[loggers]
keys=root,svg_ai

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_svg_ai]
level=INFO
handlers=consoleHandler,fileHandler
qualname=svg_ai
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=INFO
formatter=detailedFormatter
args=('logs/svg-ai.log',)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s
```

## Deployment Options

### 1. Local Development

```bash
# Start development server
python backend/app.py

# Or with Flask CLI
export FLASK_APP=backend/app.py
flask run --port 8001

# With debug mode
FLASK_DEBUG=1 python backend/app.py
```

### 2. Production with Gunicorn (Linux/macOS)

```bash
# Install Gunicorn
pip install gunicorn

# Basic Gunicorn deployment
gunicorn --bind 0.0.0.0:8001 --workers 4 backend.app:app

# Production configuration
gunicorn \
  --bind 0.0.0.0:8001 \
  --workers 4 \
  --worker-class sync \
  --worker-connections 1000 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --timeout 300 \
  --keepalive 5 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --log-level info \
  backend.app:app
```

### 3. Production with Waitress (Windows)

```bash
# Install Waitress
pip install waitress

# Run with Waitress
waitress-serve --port=8001 --threads=8 backend.app:app

# Or programmatically
python -c "
from waitress import serve
from backend.app import app
serve(app, host='0.0.0.0', port=8001, threads=8)
"
```

### 4. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_ai_phase1.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_ai_phase1.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs cache

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "4", "--timeout", "300", "backend.app:app"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  svg-ai:
    build: .
    ports:
      - "8001:8001"
    environment:
      - FLASK_ENV=production
      - AI_ENABLED=true
      - CACHE_ENABLED=true
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

#### Build and Deploy

```bash
# Build Docker image
docker build -t svg-ai-converter .

# Run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs svg-ai
```

### 5. Kubernetes Deployment

#### deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: svg-ai-converter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: svg-ai-converter
  template:
    metadata:
      labels:
        app: svg-ai-converter
    spec:
      containers:
      - name: svg-ai
        image: svg-ai-converter:latest
        ports:
        - containerPort: 8001
        env:
        - name: FLASK_ENV
          value: "production"
        - name: AI_ENABLED
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: svg-ai-service
spec:
  selector:
    app: svg-ai-converter
  ports:
  - port: 80
    targetPort: 8001
  type: LoadBalancer
```

## Reverse Proxy Configuration

### Nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 20M;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/upload {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Increase timeout for file uploads
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

### Apache

```apache
<VirtualHost *:80>
    ServerName your-domain.com

    ProxyPreserveHost On
    ProxyRequests Off

    ProxyPass / http://127.0.0.1:8001/
    ProxyPassReverse / http://127.0.0.1:8001/

    # Increase limits for file uploads
    LimitRequestBody 20971520  # 20MB
    ProxyTimeout 300
</VirtualHost>
```

## Performance Optimization

### 1. Cache Configuration

#### Multi-Level Cache Setup

```python
# Cache configuration
CACHE_CONFIG = {
    'memory_cache': {
        'max_size': 1000,  # Number of items
        'ttl': 3600,       # 1 hour
    },
    'disk_cache': {
        'max_size': 5 * 1024 * 1024 * 1024,  # 5GB
        'ttl': 24 * 3600,  # 24 hours
        'directory': '/var/cache/svg-ai'
    },
    'distributed_cache': {
        'redis_url': 'redis://localhost:6379/0',
        'ttl': 7 * 24 * 3600,  # 7 days
    }
}
```

### 2. Resource Limits

```bash
# Set resource limits in systemd service
[Service]
LimitNOFILE=65536
LimitNPROC=4096
MemoryLimit=8G
CPUQuota=400%  # 4 CPU cores
```

### 3. Parallel Processing

```python
# Configure parallel processing
PARALLEL_CONFIG = {
    'max_workers': 8,           # CPU cores
    'batch_size': 10,           # Images per batch
    'timeout': 300,             # 5 minutes per conversion
    'memory_limit': '2GB',      # Per worker memory limit
}
```

## Monitoring and Maintenance

### 1. Health Monitoring

```bash
# Health check script
#!/bin/bash
# health_check.sh

ENDPOINT="http://localhost:8001/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)

if [ $RESPONSE -eq 200 ]; then
    echo "Service healthy"
    exit 0
else
    echo "Service unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

### 2. Log Rotation

```bash
# logrotate configuration
/var/log/svg-ai/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 app app
    postrotate
        systemctl reload svg-ai
    endscript
}
```

### 3. Backup Strategy

```bash
# Backup script
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/svg-ai"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup uploads
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz uploads/

# Backup cache (optional)
tar -czf $BACKUP_DIR/cache_$DATE.tar.gz cache/

# Backup configuration
cp -r config/ $BACKUP_DIR/config_$DATE/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -type f -mtime +30 -delete
```

## Security Configuration

### 1. File Upload Security

```python
# Enhanced file validation
SECURITY_CONFIG = {
    'max_file_size': 16 * 1024 * 1024,  # 16MB
    'allowed_mime_types': [
        'image/png',
        'image/jpeg',
        'image/gif'
    ],
    'scan_uploads': True,
    'quarantine_suspicious': True,
}
```

### 2. Rate Limiting

```bash
# Nginx rate limiting
http {
    limit_req_zone $binary_remote_addr zone=upload:10m rate=5r/m;
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;
}

server {
    location /api/upload {
        limit_req zone=upload burst=3 nodelay;
    }

    location /api/ {
        limit_req zone=api burst=10 nodelay;
    }
}
```

### 3. SSL/TLS Configuration

```bash
# Generate SSL certificate
certbot --nginx -d your-domain.com

# Or with Let's Encrypt
sudo certbot certonly --standalone -d your-domain.com
```

## Troubleshooting

### Common Issues

1. **VTracer Installation Fails**
   ```bash
   # Fix: Set temporary directory
   export TMPDIR=/tmp
   pip install vtracer
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   ps aux | grep python

   # Adjust worker processes
   gunicorn --workers 2 --max-requests 500 backend.app:app
   ```

3. **Timeout Errors**
   ```bash
   # Increase timeouts
   gunicorn --timeout 600 backend.app:app
   ```

4. **Permission Errors**
   ```bash
   # Fix upload directory permissions
   chmod 755 uploads/
   chown -R app:app uploads/
   ```

### Performance Monitoring

```bash
# Monitor system resources
htop
iotop
nethogs

# Application monitoring
python scripts/monitor_performance.py

# Check logs
tail -f logs/svg-ai.log
journalctl -u svg-ai -f
```

## Production Checklist

- [ ] Python 3.9 virtual environment configured
- [ ] All dependencies installed and verified
- [ ] Configuration files properly set
- [ ] SSL/TLS certificates configured
- [ ] Reverse proxy configured (Nginx/Apache)
- [ ] File upload limits set appropriately
- [ ] Cache directories created with proper permissions
- [ ] Log rotation configured
- [ ] Monitoring and health checks enabled
- [ ] Backup strategy implemented
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Firewall rules configured
- [ ] Service auto-start configured
- [ ] Documentation updated with environment-specific details