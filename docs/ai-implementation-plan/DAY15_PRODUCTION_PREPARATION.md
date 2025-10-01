# Day 15: Production Preparation

## 📋 Executive Summary
Package the optimized and tested system for production deployment with proper configuration, health checks, deployment documentation, and rollback procedures.

## 📅 Timeline
- **Date**: Day 15 of 21
- **Duration**: 8 hours
- **Developers**: 2 developers working in parallel
  - Developer A: Containerization & Configuration
  - Developer B: Deployment Scripts & Documentation

## 📚 Prerequisites
- [ ] Day 14 integration testing complete
- [ ] All tests passing
- [ ] Performance targets met
- [ ] Final code structure (~15 files) stable

## 🎯 Goals for Day 15
1. Create production configuration
2. Set up environment variables
3. Create deployment package (Docker optional)
4. Write deployment documentation
5. Implement health checks and monitoring

## 👥 Developer Assignments

### Developer A: Containerization & Configuration
**Time**: 8 hours total
**Focus**: Create production-ready packaging and configuration

### Developer B: Deployment Scripts & Documentation
**Time**: 8 hours total
**Focus**: Build deployment automation and comprehensive documentation

---

## 📋 Task Breakdown

### Task 1: Production Configuration Setup (2 hours) - Developer A
**File**: `config/production.py`

#### Subtask 1.1: Create Configuration Management (1 hour)
- [ ] Build comprehensive config system:
  ```python
  # config/settings.py
  """Production configuration management"""

  import os
  from pathlib import Path
  from typing import Dict, Any, Optional
  from pydantic import BaseSettings, Field, validator
  import json


  class Settings(BaseSettings):
      """Application settings with validation"""

      # Application settings
      app_name: str = "AI SVG Converter"
      app_version: str = "2.0.0"
      debug: bool = False
      environment: str = Field("production", env="ENVIRONMENT")

      # Server configuration
      host: str = Field("0.0.0.0", env="HOST")
      port: int = Field(8000, env="PORT")
      workers: int = Field(4, env="WORKERS")
      reload: bool = False

      # API settings
      api_prefix: str = "/api"
      api_version: str = "v1"
      cors_origins: list = Field(["*"], env="CORS_ORIGINS")
      max_request_size: int = Field(10485760, env="MAX_REQUEST_SIZE")  # 10MB

      # Model paths
      model_dir: Path = Field("models/", env="MODEL_DIR")
      classifier_model: str = Field("classifier.pth", env="CLASSIFIER_MODEL")
      optimizer_model: str = Field("optimizer.xgb", env="OPTIMIZER_MODEL")

      # Cache configuration
      cache_enabled: bool = Field(True, env="CACHE_ENABLED")
      cache_type: str = Field("memory", env="CACHE_TYPE")  # memory, disk, redis
      cache_ttl: int = Field(3600, env="CACHE_TTL")
      cache_max_size: int = Field(1000, env="CACHE_MAX_SIZE")

      # Redis settings (optional)
      redis_host: Optional[str] = Field(None, env="REDIS_HOST")
      redis_port: int = Field(6379, env="REDIS_PORT")
      redis_db: int = Field(0, env="REDIS_DB")
      redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")

      # Database settings (for quality tracking)
      database_url: Optional[str] = Field(
          "sqlite:///./quality_tracking.db",
          env="DATABASE_URL"
      )

      # Performance settings
      max_workers: int = Field(8, env="MAX_WORKERS")
      batch_size: int = Field(20, env="BATCH_SIZE")
      request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
      max_queue_size: int = Field(100, env="MAX_QUEUE_SIZE")

      # Rate limiting
      rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
      rate_limit_requests: int = Field(60, env="RATE_LIMIT_REQUESTS")
      rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")

      # Quality targets
      target_quality_simple: float = Field(0.95, env="TARGET_QUALITY_SIMPLE")
      target_quality_text: float = Field(0.90, env="TARGET_QUALITY_TEXT")
      target_quality_gradient: float = Field(0.85, env="TARGET_QUALITY_GRADIENT")
      target_quality_complex: float = Field(0.75, env="TARGET_QUALITY_COMPLEX")

      # Monitoring
      metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
      metrics_port: int = Field(9090, env="METRICS_PORT")
      log_level: str = Field("INFO", env="LOG_LEVEL")
      log_format: str = Field("json", env="LOG_FORMAT")

      # Security
      api_key_enabled: bool = Field(False, env="API_KEY_ENABLED")
      api_keys: list = Field([], env="API_KEYS")
      ssl_enabled: bool = Field(False, env="SSL_ENABLED")
      ssl_cert: Optional[str] = Field(None, env="SSL_CERT")
      ssl_key: Optional[str] = Field(None, env="SSL_KEY")

      @validator('model_dir')
      def validate_model_dir(cls, v):
          """Ensure model directory exists"""
          path = Path(v)
          if not path.exists():
              path.mkdir(parents=True, exist_ok=True)
          return path

      @validator('workers')
      def validate_workers(cls, v):
          """Ensure reasonable worker count"""
          import multiprocessing
          max_workers = multiprocessing.cpu_count() * 2
          return min(v, max_workers)

      class Config:
          env_file = ".env"
          env_file_encoding = 'utf-8'
          case_sensitive = False

      def to_dict(self) -> Dict[str, Any]:
          """Export settings as dictionary"""
          return self.dict(exclude_none=True)

      def save_to_file(self, path: str):
          """Save configuration to JSON file"""
          with open(path, 'w') as f:
              json.dump(self.to_dict(), f, indent=2, default=str)


  # Singleton instance
  settings = Settings()


  # config/environments.py
  """Environment-specific configurations"""

  def get_development_config() -> Dict[str, Any]:
      """Development environment settings"""
      return {
          'debug': True,
          'reload': True,
          'workers': 1,
          'log_level': 'DEBUG',
          'cache_type': 'memory',
          'rate_limit_enabled': False
      }

  def get_staging_config() -> Dict[str, Any]:
      """Staging environment settings"""
      return {
          'debug': False,
          'workers': 2,
          'log_level': 'INFO',
          'cache_type': 'redis',
          'rate_limit_enabled': True,
          'api_key_enabled': True
      }

  def get_production_config() -> Dict[str, Any]:
      """Production environment settings"""
      return {
          'debug': False,
          'workers': 4,
          'log_level': 'WARNING',
          'cache_type': 'redis',
          'rate_limit_enabled': True,
          'api_key_enabled': True,
          'ssl_enabled': True,
          'metrics_enabled': True
      }
  ```
- [ ] Create environment configs
- [ ] Add validation
- [ ] Support .env files

#### Subtask 1.2: Set Up Logging Configuration (1 hour)
- [ ] Configure production logging:
  ```python
  # config/logging_config.py
  """Production logging configuration"""

  import logging
  import logging.handlers
  import json
  import sys
  from datetime import datetime
  from pathlib import Path


  class JSONFormatter(logging.Formatter):
      """JSON log formatter for production"""

      def format(self, record):
          log_obj = {
              'timestamp': datetime.utcnow().isoformat(),
              'level': record.levelname,
              'logger': record.name,
              'message': record.getMessage(),
              'module': record.module,
              'function': record.funcName,
              'line': record.lineno
          }

          if record.exc_info:
              log_obj['exception'] = self.formatException(record.exc_info)

          # Add custom fields
          for key, value in record.__dict__.items():
              if key not in ['name', 'msg', 'args', 'created', 'filename',
                           'funcName', 'levelname', 'levelno', 'lineno',
                           'module', 'msecs', 'message', 'pathname', 'process',
                           'processName', 'relativeCreated', 'thread',
                           'threadName', 'exc_info', 'exc_text', 'stack_info']:
                  log_obj[key] = value

          return json.dumps(log_obj)


  def setup_logging(
      log_level: str = "INFO",
      log_format: str = "json",
      log_file: Optional[str] = None
  ):
      """Configure logging for production"""

      # Create logs directory
      log_dir = Path("logs")
      log_dir.mkdir(exist_ok=True)

      # Root logger configuration
      root_logger = logging.getLogger()
      root_logger.setLevel(getattr(logging, log_level.upper()))

      # Remove existing handlers
      for handler in root_logger.handlers[:]:
          root_logger.removeHandler(handler)

      # Console handler
      console_handler = logging.StreamHandler(sys.stdout)

      if log_format == "json":
          console_handler.setFormatter(JSONFormatter())
      else:
          console_handler.setFormatter(
              logging.Formatter(
                  '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
              )
          )

      root_logger.addHandler(console_handler)

      # File handler with rotation
      if log_file:
          file_handler = logging.handlers.RotatingFileHandler(
              filename=log_dir / log_file,
              maxBytes=10485760,  # 10MB
              backupCount=5
          )

          if log_format == "json":
              file_handler.setFormatter(JSONFormatter())
          else:
              file_handler.setFormatter(
                  logging.Formatter(
                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                  )
              )

          root_logger.addHandler(file_handler)

      # Configure specific loggers
      logging.getLogger("uvicorn").setLevel(logging.WARNING)
      logging.getLogger("fastapi").setLevel(logging.INFO)

      return root_logger


  # Structured logging helpers
  class StructuredLogger:
      """Helper for structured logging"""

      def __init__(self, logger_name: str):
          self.logger = logging.getLogger(logger_name)

      def log_request(self, method: str, path: str, status: int, duration: float):
          """Log API request"""
          self.logger.info(
              "API Request",
              extra={
                  'method': method,
                  'path': path,
                  'status': status,
                  'duration_ms': duration * 1000
              }
          )

      def log_conversion(self, image_type: str, quality: float, duration: float):
          """Log conversion result"""
          self.logger.info(
              "Conversion completed",
              extra={
                  'image_type': image_type,
                  'quality': quality,
                  'duration_ms': duration * 1000
              }
          )

      def log_error(self, error_type: str, message: str, **kwargs):
          """Log error with context"""
          self.logger.error(
              message,
              extra={
                  'error_type': error_type,
                  **kwargs
              }
          )
  ```
- [ ] Create JSON formatter
- [ ] Set up log rotation
- [ ] Add structured logging

**Acceptance Criteria**:
- Configuration validated
- Environment variables supported
- Logging configured properly
- Settings exportable

---

### Task 2: Docker Containerization (2.5 hours) - Developer A
**File**: `Dockerfile`, `docker-compose.yml`

#### Subtask 2.1: Create Multi-Stage Dockerfile (1.5 hours)
- [ ] Build optimized container:
  ```dockerfile
  # Dockerfile
  # Stage 1: Builder
  FROM python:3.9-slim as builder

  # Install build dependencies
  RUN apt-get update && apt-get install -y \
      gcc \
      g++ \
      git \
      && rm -rf /var/lib/apt/lists/*

  # Set working directory
  WORKDIR /build

  # Copy requirements
  COPY requirements.txt .

  # Create virtual environment and install dependencies
  RUN python -m venv /opt/venv
  ENV PATH="/opt/venv/bin:$PATH"
  RUN pip install --no-cache-dir --upgrade pip && \
      pip install --no-cache-dir -r requirements.txt

  # Stage 2: Runtime
  FROM python:3.9-slim

  # Install runtime dependencies
  RUN apt-get update && apt-get install -y \
      libgomp1 \
      curl \
      && rm -rf /var/lib/apt/lists/*

  # Create non-root user
  RUN useradd -m -u 1000 appuser

  # Set working directory
  WORKDIR /app

  # Copy virtual environment from builder
  COPY --from=builder /opt/venv /opt/venv

  # Copy application code
  COPY --chown=appuser:appuser backend/ ./backend/
  COPY --chown=appuser:appuser scripts/ ./scripts/
  COPY --chown=appuser:appuser config/ ./config/
  COPY --chown=appuser:appuser models/ ./models/
  COPY --chown=appuser:appuser tests/ ./tests/

  # Set environment variables
  ENV PATH="/opt/venv/bin:$PATH" \
      PYTHONPATH=/app \
      PYTHONUNBUFFERED=1 \
      ENVIRONMENT=production

  # Create necessary directories
  RUN mkdir -p /app/logs /app/cache /app/uploads && \
      chown -R appuser:appuser /app

  # Switch to non-root user
  USER appuser

  # Health check
  HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
      CMD curl -f http://localhost:8000/health || exit 1

  # Expose ports
  EXPOSE 8000 9090

  # Default command
  CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]


  # Dockerfile.dev (Development version)
  FROM python:3.9

  WORKDIR /app

  # Install all dependencies including dev tools
  COPY requirements.txt requirements-dev.txt ./
  RUN pip install -r requirements.txt -r requirements-dev.txt

  # Copy everything
  COPY . .

  # Development command with hot reload
  CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
  ```
- [ ] Use multi-stage build
- [ ] Optimize image size
- [ ] Add health check

#### Subtask 2.2: Create Docker Compose Configuration (1 hour)
- [ ] Set up complete stack:
  ```yaml
  # docker-compose.yml
  version: '3.8'

  services:
    app:
      build:
        context: .
        dockerfile: Dockerfile
      image: ai-svg-converter:latest
      container_name: ai-svg-converter
      ports:
        - "8000:8000"
        - "9090:9090"  # Metrics port
      environment:
        - ENVIRONMENT=production
        - LOG_LEVEL=INFO
        - CACHE_TYPE=redis
        - REDIS_HOST=redis
        - DATABASE_URL=postgresql://postgres:password@postgres:5432/svgai
      volumes:
        - ./models:/app/models:ro
        - ./logs:/app/logs
        - uploads:/app/uploads
        - cache:/app/cache
      depends_on:
        - redis
        - postgres
      restart: unless-stopped
      networks:
        - svgai-network

    redis:
      image: redis:7-alpine
      container_name: svgai-redis
      ports:
        - "6379:6379"
      volumes:
        - redis-data:/data
      command: redis-server --appendonly yes
      restart: unless-stopped
      networks:
        - svgai-network

    postgres:
      image: postgres:15-alpine
      container_name: svgai-postgres
      environment:
        - POSTGRES_DB=svgai
        - POSTGRES_USER=postgres
        - POSTGRES_PASSWORD=password
      ports:
        - "5432:5432"
      volumes:
        - postgres-data:/var/lib/postgresql/data
      restart: unless-stopped
      networks:
        - svgai-network

    nginx:
      image: nginx:alpine
      container_name: svgai-nginx
      ports:
        - "80:80"
        - "443:443"
      volumes:
        - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
        - ./nginx/ssl:/etc/nginx/ssl:ro
      depends_on:
        - app
      restart: unless-stopped
      networks:
        - svgai-network

  volumes:
    redis-data:
    postgres-data:
    uploads:
    cache:

  networks:
    svgai-network:
      driver: bridge


  # docker-compose.dev.yml (Development overrides)
  version: '3.8'

  services:
    app:
      build:
        context: .
        dockerfile: Dockerfile.dev
      volumes:
        - .:/app  # Mount entire directory for hot reload
      environment:
        - ENVIRONMENT=development
        - DEBUG=true
        - LOG_LEVEL=DEBUG
  ```
- [ ] Configure services
- [ ] Set up networking
- [ ] Add persistent volumes

**Acceptance Criteria**:
- Docker images build successfully
- Containers run properly
- Health checks pass
- Services communicate

---

### Task 3: Deployment Scripts (2 hours) - Developer B
**File**: `scripts/deploy/`

#### Subtask 3.1: Create Deployment Automation (1 hour)
- [ ] Build deployment scripts:
  ```bash
  #!/bin/bash
  # scripts/deploy/deploy.sh

  set -e

  # Configuration
  ENVIRONMENT=${1:-production}
  VERSION=${2:-latest}
  BACKUP_DIR="/backup/svgai"
  DEPLOY_DIR="/opt/svgai"

  echo "🚀 Deploying AI SVG Converter v${VERSION} to ${ENVIRONMENT}"

  # Function to check prerequisites
  check_prerequisites() {
      echo "📋 Checking prerequisites..."

      # Check Docker
      if ! command -v docker &> /dev/null; then
          echo "❌ Docker not installed"
          exit 1
      fi

      # Check Docker Compose
      if ! command -v docker-compose &> /dev/null; then
          echo "❌ Docker Compose not installed"
          exit 1
      fi

      # Check disk space
      available_space=$(df /var/lib/docker | awk 'NR==2 {print $4}')
      if [ "$available_space" -lt 5000000 ]; then
          echo "❌ Insufficient disk space"
          exit 1
      fi

      echo "✅ Prerequisites satisfied"
  }

  # Function to backup current deployment
  backup_current() {
      echo "💾 Backing up current deployment..."

      timestamp=$(date +%Y%m%d_%H%M%S)
      backup_path="${BACKUP_DIR}/${timestamp}"
      mkdir -p "$backup_path"

      # Backup database
      docker exec svgai-postgres pg_dump -U postgres svgai > \
          "${backup_path}/database.sql" 2>/dev/null || true

      # Backup configuration
      cp -r "${DEPLOY_DIR}/config" "${backup_path}/" 2>/dev/null || true

      # Backup models
      cp -r "${DEPLOY_DIR}/models" "${backup_path}/" 2>/dev/null || true

      echo "✅ Backup completed: ${backup_path}"
  }

  # Function to deploy new version
  deploy_new_version() {
      echo "📦 Deploying new version..."

      cd "${DEPLOY_DIR}"

      # Pull latest code
      git fetch --all
      git checkout "${VERSION}"

      # Build images
      docker-compose build --no-cache

      # Stop current services
      docker-compose down

      # Start new services
      docker-compose up -d

      # Wait for health checks
      echo "⏳ Waiting for services to be healthy..."
      sleep 10

      # Check health
      for i in {1..30}; do
          if curl -f http://localhost:8000/health &>/dev/null; then
              echo "✅ Services are healthy"
              break
          fi
          sleep 2
      done
  }

  # Function to run migrations
  run_migrations() {
      echo "🔄 Running migrations..."

      docker exec ai-svg-converter python -m backend.migrations.run

      echo "✅ Migrations completed"
  }

  # Function to run smoke tests
  run_smoke_tests() {
      echo "🧪 Running smoke tests..."

      docker exec ai-svg-converter pytest tests/test_smoke.py -v

      if [ $? -ne 0 ]; then
          echo "❌ Smoke tests failed"
          return 1
      fi

      echo "✅ Smoke tests passed"
  }

  # Main deployment flow
  main() {
      check_prerequisites
      backup_current
      deploy_new_version
      run_migrations
      run_smoke_tests

      if [ $? -ne 0 ]; then
          echo "❌ Deployment failed, rolling back..."
          ./rollback.sh
          exit 1
      fi

      echo "✅ Deployment successful!"
      echo "📊 View metrics at: http://localhost:9090/metrics"
      echo "📝 View logs: docker logs ai-svg-converter"
  }

  main
  ```
- [ ] Create deployment script
- [ ] Add rollback capability
- [ ] Include health checks

#### Subtask 3.2: Create Rollback Procedures (1 hour)
- [ ] Implement safe rollback:
  ```bash
  #!/bin/bash
  # scripts/deploy/rollback.sh

  set -e

  echo "🔄 Starting rollback procedure..."

  # Get latest backup
  BACKUP_DIR="/backup/svgai"
  LATEST_BACKUP=$(ls -t "${BACKUP_DIR}" | head -1)

  if [ -z "$LATEST_BACKUP" ]; then
      echo "❌ No backup found"
      exit 1
  fi

  echo "📦 Rolling back to: ${LATEST_BACKUP}"

  # Stop current services
  docker-compose down

  # Restore database
  echo "💾 Restoring database..."
  docker-compose up -d postgres
  sleep 5
  docker exec -i svgai-postgres psql -U postgres svgai < \
      "${BACKUP_DIR}/${LATEST_BACKUP}/database.sql"

  # Restore configuration
  echo "⚙️ Restoring configuration..."
  cp -r "${BACKUP_DIR}/${LATEST_BACKUP}/config"/* "${DEPLOY_DIR}/config/"

  # Restore models
  echo "🤖 Restoring models..."
  cp -r "${BACKUP_DIR}/${LATEST_BACKUP}/models"/* "${DEPLOY_DIR}/models/"

  # Rebuild and restart services with previous version
  docker-compose build
  docker-compose up -d

  # Wait for services
  sleep 10

  # Verify rollback
  if curl -f http://localhost:8000/health; then
      echo "✅ Rollback successful"
  else
      echo "❌ Rollback failed - manual intervention required"
      exit 1
  fi
  ```
- [ ] Create rollback script
- [ ] Test rollback process
- [ ] Document procedure

**Acceptance Criteria**:
- Deployment automated
- Rollback tested
- Backups created
- Health verification working

---

### Task 4: Health Checks & Monitoring (1.5 hours) - Developer A
**File**: `backend/monitoring/`

#### Subtask 4.1: Implement Health Check Endpoints (45 minutes)
- [ ] Create comprehensive health checks:
  ```python
  # backend/monitoring/health.py
  """Health check and readiness endpoints"""

  from fastapi import APIRouter, HTTPException
  from typing import Dict, Any
  import asyncio
  import time
  import psutil
  from datetime import datetime


  router = APIRouter()


  class HealthChecker:
      """System health checker"""

      def __init__(self):
          self.start_time = time.time()

      async def check_database(self) -> Dict[str, Any]:
          """Check database connectivity"""
          try:
              from backend.database import get_db
              db = get_db()
              # Simple query
              result = await db.execute("SELECT 1")
              return {"status": "healthy", "response_time_ms": 1}
          except Exception as e:
              return {"status": "unhealthy", "error": str(e)}

      async def check_redis(self) -> Dict[str, Any]:
          """Check Redis connectivity"""
          try:
              from backend.cache import redis_client
              if redis_client:
                  await redis_client.ping()
                  return {"status": "healthy"}
              return {"status": "not_configured"}
          except Exception as e:
              return {"status": "unhealthy", "error": str(e)}

      async def check_models(self) -> Dict[str, Any]:
          """Check model availability"""
          try:
              from backend.ai_modules.classification import ClassificationModule
              classifier = ClassificationModule()
              return {"status": "loaded", "models": ["classifier", "optimizer"]}
          except Exception as e:
              return {"status": "error", "error": str(e)}

      def check_system_resources(self) -> Dict[str, Any]:
          """Check system resource usage"""
          return {
              "cpu_percent": psutil.cpu_percent(interval=0.1),
              "memory_percent": psutil.virtual_memory().percent,
              "disk_percent": psutil.disk_usage('/').percent,
              "status": "healthy" if psutil.virtual_memory().percent < 90 else "warning"
          }

      async def get_health_status(self) -> Dict[str, Any]:
          """Get complete health status"""
          checks = await asyncio.gather(
              self.check_database(),
              self.check_redis(),
              self.check_models(),
              return_exceptions=True
          )

          database_health = checks[0] if not isinstance(checks[0], Exception) else {"status": "error"}
          redis_health = checks[1] if not isinstance(checks[1], Exception) else {"status": "error"}
          models_health = checks[2] if not isinstance(checks[2], Exception) else {"status": "error"}

          overall_healthy = all(
              h.get("status") in ["healthy", "loaded", "not_configured"]
              for h in [database_health, redis_health, models_health]
          )

          uptime_seconds = time.time() - self.start_time

          return {
              "status": "healthy" if overall_healthy else "unhealthy",
              "timestamp": datetime.utcnow().isoformat(),
              "uptime_seconds": uptime_seconds,
              "checks": {
                  "database": database_health,
                  "redis": redis_health,
                  "models": models_health,
                  "system": self.check_system_resources()
              }
          }


  health_checker = HealthChecker()


  @router.get("/health")
  async def health_check() -> Dict[str, Any]:
      """Basic health check endpoint"""
      status = await health_checker.get_health_status()

      if status["status"] == "unhealthy":
          raise HTTPException(status_code=503, detail=status)

      return status


  @router.get("/health/live")
  async def liveness_probe() -> Dict[str, str]:
      """Kubernetes liveness probe"""
      return {"status": "alive"}


  @router.get("/health/ready")
  async def readiness_probe() -> Dict[str, Any]:
      """Kubernetes readiness probe"""
      status = await health_checker.get_health_status()

      # Check if all critical services are ready
      critical_checks = [
          status["checks"]["models"]["status"] == "loaded"
      ]

      if not all(critical_checks):
          raise HTTPException(status_code=503, detail="Not ready")

      return {"status": "ready"}
  ```
- [ ] Create health endpoints
- [ ] Add readiness checks
- [ ] Include resource monitoring

#### Subtask 4.2: Set Up Metrics Collection (45 minutes)
- [ ] Implement Prometheus metrics:
  ```python
  # backend/monitoring/metrics.py
  """Prometheus metrics collection"""

  from prometheus_client import Counter, Histogram, Gauge, generate_latest
  from fastapi import APIRouter, Response
  import time
  from functools import wraps


  # Define metrics
  request_count = Counter(
      'http_requests_total',
      'Total HTTP requests',
      ['method', 'endpoint', 'status']
  )

  request_duration = Histogram(
      'http_request_duration_seconds',
      'HTTP request duration',
      ['method', 'endpoint']
  )

  conversion_count = Counter(
      'conversions_total',
      'Total conversions',
      ['image_type', 'tier']
  )

  conversion_quality = Histogram(
      'conversion_quality_score',
      'Conversion quality scores',
      ['image_type']
  )

  conversion_duration = Histogram(
      'conversion_duration_seconds',
      'Conversion processing time',
      ['image_type', 'tier']
  )

  active_connections = Gauge(
      'active_connections',
      'Number of active connections'
  )

  cache_hits = Counter(
      'cache_hits_total',
      'Cache hit count',
      ['cache_type']
  )

  cache_misses = Counter(
      'cache_misses_total',
      'Cache miss count',
      ['cache_type']
  )

  model_inference_duration = Histogram(
      'model_inference_seconds',
      'Model inference time',
      ['model_name']
  )

  error_count = Counter(
      'errors_total',
      'Total errors',
      ['error_type']
  )


  router = APIRouter()


  @router.get("/metrics")
  async def get_metrics() -> Response:
      """Prometheus metrics endpoint"""
      return Response(
          content=generate_latest(),
          media_type="text/plain"
      )


  # Decorators for metric collection
  def track_request(endpoint: str):
      """Decorator to track HTTP requests"""
      def decorator(func):
          @wraps(func)
          async def wrapper(*args, **kwargs):
              start = time.time()
              try:
                  result = await func(*args, **kwargs)
                  status = "success"
              except Exception as e:
                  status = "error"
                  raise
              finally:
                  duration = time.time() - start
                  request_count.labels(
                      method=kwargs.get('request', {}).get('method', 'GET'),
                      endpoint=endpoint,
                      status=status
                  ).inc()
                  request_duration.labels(
                      method=kwargs.get('request', {}).get('method', 'GET'),
                      endpoint=endpoint
                  ).observe(duration)
              return result
          return wrapper
      return decorator


  def track_conversion(image_type: str, tier: int):
      """Decorator to track conversions"""
      def decorator(func):
          @wraps(func)
          async def wrapper(*args, **kwargs):
              start = time.time()
              result = await func(*args, **kwargs)
              duration = time.time() - start

              conversion_count.labels(
                  image_type=image_type,
                  tier=str(tier)
              ).inc()

              conversion_duration.labels(
                  image_type=image_type,
                  tier=str(tier)
              ).observe(duration)

              if 'quality' in result:
                  conversion_quality.labels(
                      image_type=image_type
                  ).observe(result['quality'])

              return result
          return wrapper
      return decorator
  ```
- [ ] Set up Prometheus metrics
- [ ] Add metric collection
- [ ] Create metrics endpoint

**Acceptance Criteria**:
- Health checks comprehensive
- Metrics exposed properly
- Resource monitoring working
- Prometheus format correct

---

### Task 5: Deployment Documentation (2 hours) - Developer B
**File**: `docs/DEPLOYMENT.md`

#### Subtask 5.1: Write Deployment Guide (1 hour)
- [ ] Create comprehensive documentation:
  ```markdown
  # AI SVG Converter - Deployment Guide

  ## Table of Contents
  1. [Prerequisites](#prerequisites)
  2. [Configuration](#configuration)
  3. [Deployment Methods](#deployment-methods)
  4. [Post-Deployment](#post-deployment)
  5. [Troubleshooting](#troubleshooting)

  ## Prerequisites

  ### System Requirements
  - **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+
  - **CPU**: 4+ cores recommended
  - **RAM**: 8GB minimum, 16GB recommended
  - **Disk**: 20GB free space
  - **Network**: Ports 80, 443, 8000, 9090

  ### Software Requirements
  - Docker 20.10+
  - Docker Compose 2.0+
  - Git 2.30+
  - Python 3.9+ (for local deployment)

  ## Configuration

  ### Environment Variables
  Create `.env` file in project root:

  ```env
  # Application
  ENVIRONMENT=production
  APP_VERSION=2.0.0
  DEBUG=false

  # Server
  HOST=0.0.0.0
  PORT=8000
  WORKERS=4

  # Database
  DATABASE_URL=postgresql://user:pass@localhost:5432/svgai

  # Redis
  REDIS_HOST=localhost
  REDIS_PORT=6379

  # Cache
  CACHE_TYPE=redis
  CACHE_TTL=3600
  CACHE_MAX_SIZE=1000

  # Models
  MODEL_DIR=/app/models
  CLASSIFIER_MODEL=classifier.pth
  OPTIMIZER_MODEL=optimizer.xgb

  # Security
  API_KEY_ENABLED=true
  API_KEYS=["key1", "key2"]

  # Monitoring
  METRICS_ENABLED=true
  LOG_LEVEL=INFO
  ```

  ## Deployment Methods

  ### Method 1: Docker Compose (Recommended)

  1. **Clone repository**
     ```bash
     git clone https://github.com/yourorg/ai-svg-converter.git
     cd ai-svg-converter
     ```

  2. **Configure environment**
     ```bash
     cp .env.example .env
     # Edit .env with your settings
     ```

  3. **Build and start services**
     ```bash
     docker-compose build
     docker-compose up -d
     ```

  4. **Verify deployment**
     ```bash
     curl http://localhost:8000/health
     ```

  ### Method 2: Kubernetes

  1. **Apply configurations**
     ```bash
     kubectl apply -f k8s/namespace.yaml
     kubectl apply -f k8s/configmap.yaml
     kubectl apply -f k8s/secrets.yaml
     ```

  2. **Deploy application**
     ```bash
     kubectl apply -f k8s/deployment.yaml
     kubectl apply -f k8s/service.yaml
     kubectl apply -f k8s/ingress.yaml
     ```

  3. **Check status**
     ```bash
     kubectl get pods -n svgai
     kubectl logs -n svgai deployment/ai-svg-converter
     ```

  ### Method 3: Manual Deployment

  1. **Install dependencies**
     ```bash
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

  2. **Download models**
     ```bash
     ./scripts/download_models.sh
     ```

  3. **Start application**
     ```bash
     uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
     ```

  ## Post-Deployment

  ### Health Verification
  ```bash
  # Check health
  curl http://your-domain:8000/health

  # Check metrics
  curl http://your-domain:9090/metrics

  # Run smoke tests
  docker exec ai-svg-converter pytest tests/test_smoke.py
  ```

  ### Initial Configuration
  1. Set up API keys
  2. Configure rate limiting
  3. Enable SSL/TLS
  4. Set up monitoring alerts

  ### Performance Tuning
  - Adjust worker count based on CPU cores
  - Configure cache size based on memory
  - Tune database connection pool
  - Optimize model loading

  ## Monitoring

  ### Prometheus Setup
  ```yaml
  # prometheus.yml
  scrape_configs:
    - job_name: 'ai-svg-converter'
      static_configs:
        - targets: ['localhost:9090']
  ```

  ### Grafana Dashboard
  Import dashboard from `monitoring/grafana-dashboard.json`

  ## Backup & Recovery

  ### Backup Procedure
  ```bash
  ./scripts/deploy/backup.sh
  ```

  ### Restore Procedure
  ```bash
  ./scripts/deploy/restore.sh <backup-date>
  ```

  ## Troubleshooting

  ### Common Issues

  1. **Service won't start**
     - Check logs: `docker logs ai-svg-converter`
     - Verify port availability
     - Check environment variables

  2. **Models not loading**
     - Verify model files exist
     - Check file permissions
     - Review model paths in config

  3. **High memory usage**
     - Reduce worker count
     - Lower cache size
     - Enable model unloading

  4. **Slow performance**
     - Enable caching
     - Increase workers
     - Check database indexes

  ## Security Checklist

  - [ ] Change default passwords
  - [ ] Enable API authentication
  - [ ] Configure firewall rules
  - [ ] Enable SSL/TLS
  - [ ] Set up rate limiting
  - [ ] Enable audit logging
  - [ ] Regular security updates
  ```
- [ ] Document prerequisites
- [ ] Explain configuration
- [ ] Provide examples

#### Subtask 5.2: Create Operations Manual (1 hour)
- [ ] Write operations guide:
  ```markdown
  # Operations Manual

  ## Daily Operations

  ### Monitoring Checklist
  - [ ] Check service health
  - [ ] Review error logs
  - [ ] Monitor resource usage
  - [ ] Check conversion metrics
  - [ ] Verify cache hit rate

  ### Log Management
  ```bash
  # View application logs
  docker logs ai-svg-converter --tail 100 -f

  # Export logs
  docker logs ai-svg-converter > logs/app_$(date +%Y%m%d).log

  # Parse JSON logs
  docker logs ai-svg-converter | jq '.level == "ERROR"'
  ```

  ### Scaling Operations

  #### Horizontal Scaling
  ```bash
  # Increase replicas
  docker-compose up -d --scale app=4
  ```

  #### Vertical Scaling
  Adjust in docker-compose.yml:
  ```yaml
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
  ```

  ## Maintenance Tasks

  ### Update Models
  ```bash
  # Download new models
  ./scripts/update_models.sh

  # Restart service
  docker-compose restart app
  ```

  ### Database Maintenance
  ```bash
  # Vacuum database
  docker exec svgai-postgres psql -U postgres -c "VACUUM ANALYZE;"

  # Backup database
  ./scripts/backup_database.sh
  ```

  ### Cache Management
  ```bash
  # Clear Redis cache
  docker exec svgai-redis redis-cli FLUSHDB

  # Monitor cache usage
  docker exec svgai-redis redis-cli INFO memory
  ```

  ## Emergency Procedures

  ### Service Recovery
  1. Check health endpoint
  2. Review error logs
  3. Restart service
  4. If fails, rollback

  ### Rollback Procedure
  ```bash
  ./scripts/deploy/rollback.sh
  ```

  ### Disaster Recovery
  1. Restore from backup
  2. Verify data integrity
  3. Run validation tests
  4. Resume service

  ## Performance Optimization

  ### Identify Bottlenecks
  ```bash
  # CPU profiling
  docker stats ai-svg-converter

  # Memory profiling
  docker exec ai-svg-converter python -m memory_profiler scripts/profile.py
  ```

  ### Optimization Steps
  1. Enable caching
  2. Increase workers
  3. Optimize models
  4. Add resources
  ```
- [ ] Document operations
- [ ] Include troubleshooting
- [ ] Add emergency procedures

**Acceptance Criteria**:
- Documentation complete
- Examples provided
- Troubleshooting included
- Operations clear

---

## 📊 Deployment Validation

### Pre-Deployment Checklist
```bash
# Run pre-deployment checks
./scripts/pre_deploy_check.sh

# Verify all tests pass
pytest tests/ -v

# Check Docker build
docker build -t ai-svg-converter:test .

# Test configuration
python config/validate_config.py
```

### Post-Deployment Validation
```bash
# Health check
curl http://localhost:8000/health

# API test
curl -X POST http://localhost:8000/api/convert \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_data"}'

# Metrics check
curl http://localhost:9090/metrics | grep conversion_total

# Load test
python scripts/load_test.py --users 10 --duration 60
```

---

## ✅ Production Readiness Checklist

### Configuration
- [ ] Environment variables set
- [ ] Logging configured
- [ ] Secrets management
- [ ] SSL certificates

### Containerization
- [ ] Docker image optimized
- [ ] Health checks working
- [ ] Resource limits set
- [ ] Security scanning passed

### Deployment
- [ ] Automated deployment script
- [ ] Rollback procedure tested
- [ ] Backup strategy implemented
- [ ] Monitoring configured

### Documentation
- [ ] Deployment guide complete
- [ ] Operations manual ready
- [ ] API documentation updated
- [ ] Troubleshooting guide

### Testing
- [ ] Smoke tests passing
- [ ] Integration tests complete
- [ ] Performance validated
- [ ] Security tested

---

## 🎯 Success Metrics

### Deployment Goals
- [ ] Zero-downtime deployment
- [ ] Rollback < 5 minutes
- [ ] Container size < 500MB
- [ ] Startup time < 30 seconds

### Operational Goals
- [ ] Health checks responsive
- [ ] Metrics collecting
- [ ] Logs structured
- [ ] Alerts configured

---

## 📝 Handoff Package

Create handoff package with:
1. Deployment guide
2. Operations manual
3. Architecture diagram
4. API documentation
5. Troubleshooting guide
6. Contact information
7. Known issues list
8. Future roadmap

---

## 🔄 Next Steps

Week 4 (Days 16-21):
1. Day 16: Monitoring & Metrics
2. Day 17: Documentation
3. Day 18: Final Testing
4. Day 19: Production Deployment
5. Day 20: Knowledge Transfer
6. Day 21: Retrospective