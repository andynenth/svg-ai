# Development Plan - Day 4: Deployment Preparation & Security

**Date**: Production Readiness Sprint - Day 4
**Objective**: Prepare for production deployment with security and infrastructure setup
**Duration**: 8 hours
**Priority**: HIGH

## 🎯 Day 4 Success Criteria
- [ ] Production-ready containerization with Docker
- [ ] Security vulnerabilities addressed and hardened
- [ ] CI/CD pipeline operational
- [ ] Production configuration management implemented
- [ ] Deployment documentation complete

---

## 📊 Day 4 Starting Point

### Prerequisites (From Days 1-3)
- [x] Core functionality stable and tested
- [x] Performance targets met consistently
- [x] Error handling and recovery operational
- [x] Test coverage >80%

### Focus Areas
- **Containerization**: Docker setup for consistent deployment
- **Security**: Vulnerability scanning and hardening
- **Configuration**: Environment-specific settings
- **CI/CD**: Automated deployment pipeline
- **Documentation**: Production deployment guide

---

## 🚀 Task Breakdown

### Task 1: Production Containerization (2.5 hours) - CRITICAL
**Problem**: Need reliable, consistent deployment across environments

#### Subtask 1.1: Create Production Dockerfile (1 hour)
**Files**: `Dockerfile`, `.dockerignore`
**Dependencies**: None
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 1.1.1** (30 min): Create optimized multi-stage Dockerfile
  ```dockerfile
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
  ```

- [x] **Step 1.1.2** (15 min): Create comprehensive `.dockerignore`
  ```dockerignore
  .git
  .gitignore
  README.md
  Dockerfile
  .dockerignore
  .pytest_cache
  __pycache__
  *.pyc
  *.pyo
  *.pyd
  .Python
  .coverage
  coverage_html_report/
  venv*/
  node_modules/
  .DS_Store
  logs/
  *.log
  .env
  .env.local
  temp/
  *.tmp
  test_data/
  development_plan_*.md
  ```

- [x] **Step 1.1.3** (15 min): Test Docker build and basic functionality

#### Subtask 1.2: Create Docker Compose for Development/Production (1 hour)
**Files**: `docker-compose.yml`, `docker-compose.prod.yml`
**Dependencies**: Subtask 1.1
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 1.2.1** (30 min): Create development docker-compose.yml
  ```yaml
  version: '3.8'

  services:
    svg-ai:
      build:
        context: .
        dockerfile: Dockerfile
      ports:
        - "5000:5000"
      volumes:
        - .:/home/app
        - /home/app/__pycache__
      environment:
        - FLASK_ENV=development
        - FLASK_DEBUG=1
      depends_on:
        - redis
      networks:
        - svg-ai-network

    redis:
      image: redis:7-alpine
      ports:
        - "6379:6379"
      volumes:
        - redis_data:/data
      networks:
        - svg-ai-network

  volumes:
    redis_data:

  networks:
    svg-ai-network:
      driver: bridge
  ```

- [x] **Step 1.2.2** (30 min): Create production docker-compose.prod.yml
  ```yaml
  version: '3.8'

  services:
    svg-ai:
      build:
        context: .
        dockerfile: Dockerfile
        target: production
      ports:
        - "5000:5000"
      environment:
        - FLASK_ENV=production
        - WORKERS=4
      restart: unless-stopped
      depends_on:
        - redis
      networks:
        - svg-ai-network
      deploy:
        resources:
          limits:
            memory: 2G
          reservations:
            memory: 1G

    redis:
      image: redis:7-alpine
      restart: unless-stopped
      volumes:
        - redis_data:/data
      networks:
        - svg-ai-network
      deploy:
        resources:
          limits:
            memory: 256M

    nginx:
      image: nginx:alpine
      ports:
        - "80:80"
        - "443:443"
      volumes:
        - ./nginx.conf:/etc/nginx/nginx.conf:ro
        - ./ssl:/etc/nginx/ssl:ro
      depends_on:
        - svg-ai
      restart: unless-stopped
      networks:
        - svg-ai-network

  volumes:
    redis_data:

  networks:
    svg-ai-network:
      driver: bridge
  ```

#### Subtask 1.3: Container Optimization & Security (30 min)
**Files**: Dockerfile optimizations, security configs
**Dependencies**: Previous subtasks
**Estimated Time**: 30 minutes

**Implementation Steps**:
- [x] **Step 1.3.1** (15 min): Implement security best practices
  - Non-root user execution
  - Minimal base image
  - No unnecessary packages
  - Proper file permissions

- [x] **Step 1.3.2** (15 min): Optimize image size and build time
  - Multi-stage builds
  - Layer optimization
  - Cache-friendly ordering

---

### Task 2: Security Hardening (2.5 hours) - CRITICAL
**Problem**: Production deployment requires comprehensive security measures

#### Subtask 2.1: Vulnerability Assessment & Fixes (1 hour)
**Files**: Security configurations, dependency updates
**Dependencies**: None
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 2.1.1** (30 min): Run security vulnerability scan
  ```bash
  # Install security tools
  pip install safety bandit

  # Check for known vulnerabilities in dependencies
  safety check

  # Static security analysis
  bandit -r backend/ -f json -o security_report.json

  # Container security scanning
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd):/root/.cache/ aquasec/trivy image svg-ai:latest
  ```

- [x] **Step 2.1.2** (30 min): Address identified vulnerabilities
  - Update vulnerable dependencies
  - Fix security anti-patterns in code
  - Implement secure coding practices

#### Subtask 2.2: Input Validation & Sanitization (1 hour)
**Files**: `backend/utils/security.py`, API endpoint updates
**Dependencies**: Subtask 2.1
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 2.2.1** (45 min): Implement comprehensive input validation
  ```python
  import re
  from typing import Any, Dict, Optional
  import base64
  from pathlib import Path

  class SecurityValidator:
      def __init__(self):
          self.max_file_size = 10 * 1024 * 1024  # 10MB
          self.allowed_image_types = ['image/png', 'image/jpeg', 'image/gif']
          self.dangerous_patterns = [
              r'\.\./',          # Path traversal
              r'<script',        # Script injection
              r'javascript:',    # JavaScript protocol
              r'data:.*base64',  # Suspicious data URLs
              r'file://',        # File protocol
          ]

      def validate_image_upload(self, image_data: str, filename: str = None) -> Dict[str, Any]:
          """Validate uploaded image data"""
          result = {'valid': True, 'errors': []}

          # Validate base64 format
          try:
              decoded = base64.b64decode(image_data)
              if len(decoded) > self.max_file_size:
                  result['valid'] = False
                  result['errors'].append(f"File too large: {len(decoded)} bytes")
          except Exception:
              result['valid'] = False
              result['errors'].append("Invalid base64 encoding")

          # Validate filename if provided
          if filename:
              if not self.validate_filename(filename):
                  result['valid'] = False
                  result['errors'].append("Invalid filename")

          return result

      def validate_filename(self, filename: str) -> bool:
          """Validate filename for security"""
          # Check for dangerous patterns
          for pattern in self.dangerous_patterns:
              if re.search(pattern, filename, re.IGNORECASE):
                  return False

          # Check filename length
          if len(filename) > 255:
              return False

          # Check for only safe characters
          safe_pattern = r'^[a-zA-Z0-9._-]+$'
          return bool(re.match(safe_pattern, filename))

      def sanitize_output(self, data: Any) -> Any:
          """Sanitize output data"""
          if isinstance(data, str):
              # Remove potentially dangerous content
              data = re.sub(r'<script.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
              data = re.sub(r'javascript:', '', data, flags=re.IGNORECASE)
          elif isinstance(data, dict):
              return {k: self.sanitize_output(v) for k, v in data.items()}
          elif isinstance(data, list):
              return [self.sanitize_output(item) for item in data]

          return data
  ```

- [x] **Step 2.2.2** (15 min): Integrate validation into API endpoints

#### Subtask 2.3: Rate Limiting & DDoS Protection (30 min)
**Files**: Rate limiting middleware, protection configs
**Dependencies**: Previous subtasks
**Estimated Time**: 30 minutes

**Implementation Steps**:
- [x] **Step 2.3.1** (20 min): Implement rate limiting
  ```python
  from flask_limiter import Limiter
  from flask_limiter.util import get_remote_address
  import redis

  # Initialize rate limiter
  limiter = Limiter(
      app,
      key_func=get_remote_address,
      storage_uri="redis://localhost:6379",
      default_limits=["200 per day", "50 per hour"]
  )

  # Apply stricter limits to resource-intensive endpoints
  @app.route('/api/convert', methods=['POST'])
  @limiter.limit("10 per minute")
  def convert_endpoint():
      # Existing implementation
      pass

  @app.route('/api/batch-convert', methods=['POST'])
  @limiter.limit("2 per minute")
  def batch_convert():
      # Existing implementation
      pass
  ```

- [x] **Step 2.3.2** (10 min): Configure nginx rate limiting
  ```nginx
  http {
      limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
      limit_req_zone $binary_remote_addr zone=batch:10m rate=1r/s;

      server {
          location /api/convert {
              limit_req zone=api burst=20 nodelay;
              proxy_pass http://svg-ai:5000;
          }

          location /api/batch {
              limit_req zone=batch burst=5 nodelay;
              proxy_pass http://svg-ai:5000;
          }
      }
  }
  ```

---

### Task 3: CI/CD Pipeline Setup (2 hours) - HIGH PRIORITY
**Problem**: Need automated testing and deployment pipeline

#### Subtask 3.1: GitHub Actions Workflow (1 hour)
**Files**: `.github/workflows/ci-cd.yml`
**Dependencies**: Previous tasks
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 3.1.1** (45 min): Create comprehensive CI/CD workflow
  ```yaml
  name: CI/CD Pipeline

  on:
    push:
      branches: [ main, develop ]
    pull_request:
      branches: [ main ]

  jobs:
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          python-version: [3.9]

      steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements_ai_phase1.txt

      - name: Security scan
        run: |
          pip install safety bandit
          safety check
          bandit -r backend/ -f json -o security_report.json

      - name: Run tests with coverage
        run: |
          python -m pytest tests/ --cov=backend --cov-report=xml --cov-fail-under=80

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

    docker:
      needs: test
      runs-on: ubuntu-latest
      if: github.ref == 'refs/heads/main'

      steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

    deploy:
      needs: [test, docker]
      runs-on: ubuntu-latest
      if: github.ref == 'refs/heads/main'
      environment: production

      steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        run: |
          # Deployment script here
          echo "Deployment would happen here"
  ```

- [x] **Step 3.1.2** (15 min): Test workflow with dummy deployment

#### Subtask 3.2: Environment Configuration Management (1 hour)
**Files**: Environment configs, secrets management
**Dependencies**: Subtask 3.1
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 3.2.1** (30 min): Create environment-specific configurations
  ```python
  # config/environments.py
  import os
  from typing import Dict, Any

  class BaseConfig:
      SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
      REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
      MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
      UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')

  class DevelopmentConfig(BaseConfig):
      DEBUG = True
      FLASK_ENV = 'development'
      TESTING = False

  class ProductionConfig(BaseConfig):
      DEBUG = False
      FLASK_ENV = 'production'
      TESTING = False
      SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production

  class TestingConfig(BaseConfig):
      TESTING = True
      DEBUG = True

  config = {
      'development': DevelopmentConfig,
      'production': ProductionConfig,
      'testing': TestingConfig,
      'default': DevelopmentConfig
  }
  ```

- [ ] **Step 3.2.2** (30 min): Create deployment scripts and environment management

---

### Task 4: Production Monitoring & Logging (1 hour) - MEDIUM
**Problem**: Need visibility into production system behavior

#### Subtask 4.1: Implement Structured Logging (1 hour)
**Files**: `backend/utils/logging_config.py`
**Dependencies**: None
**Estimated Time**: 1 hour

**Implementation Steps**:
- [x] **Step 4.1.1** (45 min): Setup structured logging
  ```python
  import logging
  import json
  import time
  from typing import Dict, Any

  class StructuredLogger:
      def __init__(self, service_name: str = "svg-ai"):
          self.service_name = service_name
          self.logger = logging.getLogger(service_name)
          self._setup_handlers()

      def _setup_handlers(self):
          # Console handler with JSON formatting
          console_handler = logging.StreamHandler()
          console_handler.setFormatter(JSONFormatter())

          # File handler for persistent logs
          file_handler = logging.FileHandler('logs/application.log')
          file_handler.setFormatter(JSONFormatter())

          self.logger.addHandler(console_handler)
          self.logger.addHandler(file_handler)
          self.logger.setLevel(logging.INFO)

      def info(self, message: str, extra: Dict[str, Any] = None):
          self._log('info', message, extra)

      def warning(self, message: str, extra: Dict[str, Any] = None):
          self._log('warning', message, extra)

      def error(self, message: str, extra: Dict[str, Any] = None):
          self._log('error', message, extra)

      def _log(self, level: str, message: str, extra: Dict[str, Any] = None):
          log_data = {
              'timestamp': time.time(),
              'service': self.service_name,
              'level': level,
              'message': message,
              **(extra or {})
          }

          getattr(self.logger, level)(json.dumps(log_data))

  class JSONFormatter(logging.Formatter):
      def format(self, record):
          log_data = {
              'timestamp': record.created,
              'level': record.levelname,
              'message': record.getMessage(),
              'module': record.module,
              'function': record.funcName,
              'line': record.lineno
          }

          if hasattr(record, 'extra'):
              log_data.update(record.extra)

          return json.dumps(log_data)
  ```

- [x] **Step 4.1.2** (15 min): Integrate structured logging into application

---

## 📈 Progress Tracking

### Hourly Checkpoints
- **Hour 1**: ⏳ Docker configuration complete
- **Hour 2**: ⏳ Container orchestration ready
- **Hour 3**: ⏳ Security vulnerabilities addressed
- **Hour 4**: ⏳ Input validation implemented
- **Hour 5**: ⏳ CI/CD pipeline operational
- **Hour 6**: ⏳ Environment configuration ready
- **Hour 7**: ⏳ Production monitoring setup
- **Hour 8**: ⏳ Deployment validation complete

### Success Metrics Tracking
- [ ] Docker Build: SUCCESS/FAILURE
- [ ] Security Scan: ___/0 critical vulnerabilities
- [ ] CI/CD Pipeline: PASSING/FAILING
- [ ] Production Deploy: READY/NOT READY

---

## 📋 End of Day 4 Deliverables

### Required Outputs
- [ ] **Production Docker Configuration**: Multi-stage, optimized, secure
- [ ] **Security Assessment Report**: Vulnerabilities addressed
- [ ] **CI/CD Pipeline**: Fully operational automated testing and deployment
- [ ] **Deployment Documentation**: Complete production setup guide

### Production Readiness Checklist
- [ ] Container security hardened
- [ ] Input validation comprehensive
- [ ] Rate limiting operational
- [ ] Monitoring and logging configured
- [ ] CI/CD pipeline tested

---

## 🎯 Day 4 Completion Criteria

**MANDATORY (All must pass)**:
✅ Docker production build successful
✅ Security scan: 0 critical vulnerabilities
✅ CI/CD pipeline: All tests passing
✅ Deployment ready: All configs validated

**READY FOR DAY 5 IF**:
- Containerization production-ready
- Security hardening complete
- CI/CD pipeline operational
- Deployment infrastructure ready

---

*Day 4 establishes the foundation for secure, reliable production deployment with proper DevOps practices.*