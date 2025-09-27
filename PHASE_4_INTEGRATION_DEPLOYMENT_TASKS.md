# Phase 4: Integration, Testing & Deployment - Task Breakdown

## Overview
Integrate Phase 2 (Backend) and Phase 3 (Frontend), perform comprehensive testing, and prepare for deployment.

---

## Task Groups

### 1. Backend-Frontend Integration (45 min)

#### 1.1 Verify Backend is Running
- [x] Open terminal
- [x] Navigate to backend: `cd backend`
- [x] Check Python version: `python --version`
- [x] Activate venv: `source ../venv39/bin/activate`
- [x] Install dependencies: `pip install -r requirements.txt`
- [x] Start server: `python app.py`
- [x] Test health endpoint: `curl http://localhost:8000/health`
- [x] Verify response: `{"status": "ok"}`

#### 1.2 Configure CORS Properly
- [x] Open `backend/app.py`
- [x] Find CORS initialization
- [x] Add origins: `CORS(app, origins=['http://localhost:3000', 'http://localhost:8080'])`
- [x] Add methods: `methods=['GET', 'POST', 'OPTIONS']`
- [x] Add headers: `allow_headers=['Content-Type']`
- [x] Save file
- [x] Restart backend

#### 1.3 Serve Frontend from Backend
- [x] In app.py, add import: `from flask import send_from_directory`
- [x] Add route: `@app.route('/')`
- [x] Add function: `def serve_frontend():`
- [x] Return: `return send_from_directory('../frontend', 'index.html')`
- [x] Add static route: `@app.route('/<path:path>')`
- [x] Add function: `def serve_static(path):`
- [x] Return: `return send_from_directory('../frontend', path)`
- [x] Test: Visit `http://localhost:8000`
- [x] Verify: Frontend loads

#### 1.4 Update Frontend API URL
- [x] Open `frontend/script.js`
- [x] Find: `const API_BASE = 'http://localhost:8000';`
- [x] Change to: `const API_BASE = '';`
- [x] This makes it relative to current host
- [x] Save file
- [x] Refresh browser
- [x] Test upload still works

---

### 2. End-to-End Testing Setup (40 min)

#### 2.1 Create Test Data Directory
- [x] Create folder: `mkdir test-data`
- [x] Copy test images: `cp data/logos/simple_geometric/circle_00.png test-data/`
- [x] Copy: `cp data/logos/text_based/text_00.png test-data/`
- [x] Copy: `cp data/logos/gradients/gradient_00.png test-data/`
- [x] List files: `ls test-data/`
- [x] Verify 3 test files

#### 2.2 Create E2E Test Script
- [x] Create file: `test_e2e.py`
- [x] Add shebang: `#!/usr/bin/env python3`
- [x] Add docstring: `"""End-to-end tests for PNG to SVG converter"""`
- [x] Import requests: `import requests`
- [x] Import os: `import os`
- [x] Import json: `import json`
- [x] Import sys: `import sys`

#### 2.3 Define Test Configuration
- [x] Add BASE_URL: `BASE_URL = 'http://localhost:8000'`
- [x] Add test files: `TEST_FILES = ['circle_00.png', 'text_00.png', 'gradient_00.png']`
- [x] Add test dir: `TEST_DIR = 'test-data'`
- [x] Add results: `results = []`

#### 2.4 Create Upload Test Function
- [x] Define function: `def test_upload(filename):`
- [x] Build path: `filepath = os.path.join(TEST_DIR, filename)`
- [x] Check exists: `if not os.path.exists(filepath):`
- [x] Return error: `return None, f"File not found: {filepath}"`
- [x] Open file: `with open(filepath, 'rb') as f:`
- [x] Create files: `files = {'file': f}`
- [x] Post request: `r = requests.post(f'{BASE_URL}/api/upload', files=files)`
- [x] Check status: `if r.status_code != 200:`
- [x] Return error: `return None, f"Upload failed: {r.status_code}"`
- [x] Return success: `return r.json()['file_id'], None`

#### 2.5 Create Convert Test Function
- [x] Define function: `def test_convert(file_id, converter='alpha'):`
- [x] Create data: `data = {'file_id': file_id, 'converter': converter}`
- [x] Post request: `r = requests.post(f'{BASE_URL}/api/convert', json=data)`
- [x] Check status: `if r.status_code != 200:`
- [x] Return error: `return None, f"Convert failed: {r.status_code}"`
- [x] Get result: `result = r.json()`
- [x] Check success: `if not result.get('success'):`
- [x] Return error: `return None, result.get('error')`
- [x] Return result: `return result, None`

---

### 3. Comprehensive Test Suite (50 min)

#### 3.1 Test All Converters
- [x] Create function: `def test_all_converters():`
- [x] Define converters: `converters = ['alpha', 'vtracer', 'potrace']`
- [x] For each file: `for test_file in TEST_FILES:`
- [x] Upload file: `file_id, error = test_upload(test_file)`
- [x] Check error: `if error: print(f"Upload error: {error}")`
- [x] For each converter: `for converter in converters:`
- [x] Test convert: `result, error = test_convert(file_id, converter)`
- [x] Record result: `results.append({...})`

#### 3.2 Test Invalid Inputs
- [x] Create function: `def test_invalid_inputs():`
- [x] Test no file: `r = requests.post(f'{BASE_URL}/api/upload')`
- [x] Assert 400: `assert r.status_code == 400`
- [x] Test bad file_id: `r = requests.post(f'{BASE_URL}/api/convert', json={'file_id': 'invalid'})`
- [x] Assert 404: `assert r.status_code == 404`
- [x] Test bad converter: `json={'file_id': 'test', 'converter': 'invalid'}`
- [x] Check error response

#### 3.3 Test File Size Limits
- [x] Create function: `def test_file_limits():`
- [x] Create large file: `large_data = b'x' * (20 * 1024 * 1024)`
- [x] Try upload: `files = {'file': ('large.png', large_data)}`
- [x] Post request: `r = requests.post(f'{BASE_URL}/api/upload', files=files)`
- [x] Check rejected: `assert r.status_code == 413 or 'too large' in r.text.lower()`

#### 3.4 Test Concurrent Requests
- [x] Import threading: `import threading`
- [x] Create function: `def test_concurrent():`
- [x] Create threads: `threads = []`
- [x] Define worker: `def worker(file_id):`
- [x] Call convert: `test_convert(file_id)`
- [x] Create 5 threads: `for i in range(5):`
- [x] Start thread: `t = threading.Thread(target=worker, args=(file_id,))`
- [x] Append: `threads.append(t)`
- [x] Start all: `t.start()`
- [x] Wait for all: `for t in threads: t.join()`

#### 3.5 Create Test Runner
- [x] Define main: `if __name__ == '__main__':`
- [x] Print header: `print("Running E2E Tests...")`
- [x] Run upload test: `test_all_converters()`
- [x] Run invalid test: `test_invalid_inputs()`
- [x] Run limit test: `test_file_limits()`
- [x] Run concurrent: `test_concurrent()`
- [x] Print summary: `print(f"Tests completed: {len(results)} conversions tested")`

---

### 4. Performance Optimization (40 min)

#### 4.1 Add File Cleanup
- [x] In backend/app.py, import: `import tempfile`
- [x] Import: `import atexit`
- [x] Import: `from datetime import datetime, timedelta`
- [x] Create cleanup function: `def cleanup_old_files():`
- [x] Get upload dir: `upload_dir = UPLOAD_FOLDER`
- [x] Current time: `now = datetime.now()`
- [x] For each file: `for filename in os.listdir(upload_dir):`
- [x] Get file time: `file_path = os.path.join(upload_dir, filename)`
- [x] Get modified: `modified = datetime.fromtimestamp(os.path.getmtime(file_path))`
- [x] Check age: `if now - modified > timedelta(hours=1):`
- [x] Remove file: `os.remove(file_path)`

#### 4.2 Add Scheduled Cleanup
- [x] Import: `from threading import Timer`
- [x] Create function: `def schedule_cleanup():`
- [x] Run cleanup: `cleanup_old_files()`
- [x] Schedule next: `Timer(3600, schedule_cleanup).start()`
- [x] Start scheduler: `schedule_cleanup()`

#### 4.3 Add Response Caching
- [x] Create cache dict: `conversion_cache = {}`
- [x] In convert function: `cache_key = f"{file_id}_{converter}_{threshold}"`
- [x] Check cache: `if cache_key in conversion_cache:`
- [x] Return cached: `return jsonify(conversion_cache[cache_key])`
- [x] After conversion: `conversion_cache[cache_key] = result`
- [x] Limit cache size: `if len(conversion_cache) > 100:`
- [x] Clear oldest: `conversion_cache.pop(next(iter(conversion_cache)))`

#### 4.4 Optimize SVG Output
- [x] Import: `import gzip`
- [x] In convert response: `if len(svg_content) > 10000:`
- [x] Compress: `response.headers['Content-Encoding'] = 'gzip'`
- [x] Return compressed: `return gzip.compress(svg_content.encode())`

#### 4.5 Add Request Rate Limiting
- [x] Install: `pip install flask-limiter`
- [x] Import: `from flask_limiter import Limiter`
- [x] Initialize: `limiter = Limiter(app, key_func=lambda: request.remote_addr)`
- [x] Add to upload: `@limiter.limit("10 per minute")`
- [x] Add to convert: `@limiter.limit("20 per minute")`

---

### 5. Security Hardening (35 min)

#### 5.1 Validate File Types
- [x] Install: `pip install python-magic`
- [x] Import: `import magic`
- [x] In upload: `file_mime = magic.from_buffer(content, mime=True)`
- [x] Check mime: `if not file_mime.startswith('image/'):`
- [x] Reject: `return jsonify({'error': 'Invalid file type'}), 400`

#### 5.2 Sanitize Filenames
- [x] Import: `from werkzeug.utils import secure_filename`
- [x] In upload: `original_name = secure_filename(file.filename)`
- [x] Remove dots: `original_name = original_name.replace('..', '')`
- [x] Limit length: `original_name = original_name[:100]`

#### 5.3 Add Request Validation
- [x] Create decorator: `def validate_json(*expected_args):`
- [x] Check content-type: `if request.content_type != 'application/json':`
- [x] Return error: `return jsonify({'error': 'Content-Type must be application/json'}), 400`
- [x] Check args: `for arg in expected_args:`
- [x] If missing: `if arg not in request.json:`
- [x] Return error: `return jsonify({'error': f'Missing required field: {arg}'}), 400`

#### 5.4 Add CSRF Protection
- [x] Generate token: `import secrets`
- [x] Create token: `csrf_token = secrets.token_hex(16)`
- [x] Add to session: `session['csrf_token'] = csrf_token`
- [x] Check token: `if request.form.get('csrf_token') != session.get('csrf_token'):`
- [x] Reject: `return jsonify({'error': 'Invalid CSRF token'}), 403`

#### 5.5 Secure Headers
- [x] Install: `pip install flask-talisman`
- [x] Import: `from flask_talisman import Talisman`
- [x] Initialize: `Talisman(app, force_https=False)`
- [x] Add CSP: `csp = {'default-src': "'self'"}`
- [x] Apply CSP: `Talisman(app, content_security_policy=csp)`

---

### 6. Production Configuration (30 min)

#### 6.1 Create Config File
- [x] Create `backend/config.py`
- [x] Add class: `class Config:`
- [x] Add debug: `DEBUG = False`
- [x] Add testing: `TESTING = False`
- [x] Add secret: `SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key')`
- [x] Add upload: `UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')`
- [x] Add max size: `MAX_CONTENT_LENGTH = 16 * 1024 * 1024`

#### 6.2 Create Development Config
- [x] Add class: `class DevelopmentConfig(Config):`
- [x] Set debug: `DEBUG = True`
- [x] Set host: `HOST = '127.0.0.1'`
- [x] Set port: `PORT = 8000`

#### 6.3 Create Production Config
- [x] Add class: `class ProductionConfig(Config):`
- [x] Set debug: `DEBUG = False`
- [x] Set host: `HOST = '0.0.0.0'`
- [x] Set port: `PORT = int(os.environ.get('PORT', 8000))`

#### 6.4 Update App to Use Config
- [x] In app.py: `from config import Config, DevelopmentConfig, ProductionConfig`
- [x] Get env: `env = os.environ.get('FLASK_ENV', 'development')`
- [x] Select config: `config = ProductionConfig if env == 'production' else DevelopmentConfig`
- [x] Apply: `app.config.from_object(config)`

#### 6.5 Create Environment File
- [x] Create `.env.example`
- [x] Add: `FLASK_ENV=development`
- [x] Add: `SECRET_KEY=your-secret-key-here`
- [x] Add: `UPLOAD_FOLDER=uploads`
- [x] Add: `MAX_FILE_SIZE=16777216`
- [x] Copy to .env: `cp .env.example .env`

---

### 7. Docker Setup (35 min)

#### 7.1 Create Backend Dockerfile
- [x] Create `backend/Dockerfile`
- [x] Add base: `FROM python:3.9-slim`
- [x] Set workdir: `WORKDIR /app`
- [x] Copy requirements: `COPY requirements.txt .`
- [x] Install deps: `RUN pip install --no-cache-dir -r requirements.txt`
- [x] Copy backend: `COPY . .`
- [x] Expose port: `EXPOSE 8000`
- [x] Set command: `CMD ["python", "app.py"]`

#### 7.2 Create Frontend Dockerfile
- [x] Create `frontend/Dockerfile`
- [x] Add base: `FROM nginx:alpine`
- [x] Copy files: `COPY . /usr/share/nginx/html`
- [x] Copy config: `COPY nginx.conf /etc/nginx/nginx.conf`
- [x] Expose port: `EXPOSE 80`

#### 7.3 Create Nginx Config
- [x] Create `frontend/nginx.conf`
- [x] Add events: `events { worker_connections 1024; }`
- [x] Add http: `http { }`
- [x] Add server: `server { listen 80; }`
- [x] Add location: `location / { root /usr/share/nginx/html; }`
- [x] Add API proxy: `location /api { proxy_pass http://backend:8000; }`

#### 7.4 Create Docker Compose
- [x] Create `docker-compose.yml` in root
- [x] Add version: `version: '3.8'`
- [x] Add services: `services:`
- [x] Add backend: `backend:`
- [x] Build: `build: ./backend`
- [x] Ports: `ports: - "8000:8000"`
- [x] Volumes: `volumes: - ./uploads:/app/uploads`
- [x] Add frontend: `frontend:`
- [x] Build: `build: ./frontend`
- [x] Ports: `ports: - "80:80"`
- [x] Depends: `depends_on: - backend`

#### 7.5 Test Docker Setup
- [x] Build images: `docker-compose build`
- [x] Start services: `docker-compose up`
- [x] Test backend: `curl http://localhost:8000/health`
- [x] Test frontend: Open `http://localhost`
- [x] Stop services: `docker-compose down`

---

### 8. API Documentation (25 min)

#### 8.1 Create API Docs
- [x] Create `API_DOCUMENTATION.md`
- [x] Add title: `# PNG to SVG Converter API Documentation`
- [x] Add version: `Version: 1.0.0`
- [x] Add base URL: `Base URL: http://localhost:8000`

#### 8.2 Document Health Endpoint
- [x] Add heading: `## Health Check`
- [x] Add method: `GET /health`
- [x] Add description: `Check if server is running`
- [x] Add response: `200 OK: {"status": "ok"}`
- [x] Add example: `curl http://localhost:8000/health`

#### 8.3 Document Upload Endpoint
- [x] Add heading: `## Upload Image`
- [x] Add method: `POST /api/upload`
- [x] Add content-type: `Content-Type: multipart/form-data`
- [x] Add parameters: `file: Image file (PNG/JPEG)`
- [x] Add success: `200: {"file_id": "...", "filename": "..."}`
- [x] Add errors: `400: {"error": "No file provided"}`
- [x] Add example curl command

#### 8.4 Document Convert Endpoint
- [x] Add heading: `## Convert Image`
- [x] Add method: `POST /api/convert`
- [x] Add content-type: `Content-Type: application/json`
- [x] Add body: `{"file_id": "...", "converter": "alpha", "threshold": 128}`
- [x] Add converters: List alpha, vtracer, potrace
- [x] Add success: `200: {"success": true, "svg": "...", "ssim": 0.95}`
- [x] Add errors: `404: {"error": "File not found"}`
- [x] Add example curl command

#### 8.5 Add Error Codes
- [x] Add heading: `## Error Responses`
- [x] Add 400: `Bad Request - Invalid input`
- [x] Add 404: `Not Found - File not found`
- [x] Add 413: `Payload Too Large - File too big`
- [x] Add 429: `Too Many Requests - Rate limited`
- [x] Add 500: `Internal Server Error`

---

### 9. Deployment Scripts (30 min)

#### 9.1 Create Start Script
- [x] Create `start.sh`
- [x] Add shebang: `#!/bin/bash`
- [x] Add comment: `# Start the application`
- [x] Check Python: `python3 --version`
- [x] Activate venv: `source venv39/bin/activate`
- [x] Install deps: `pip install -r backend/requirements.txt`
- [x] Start backend: `cd backend && python app.py`

#### 9.2 Create Stop Script
- [x] Create `stop.sh`
- [x] Add shebang: `#!/bin/bash`
- [x] Find process: `PID=$(ps aux | grep 'python app.py' | grep -v grep | awk '{print $2}')`
- [x] Check if running: `if [ -n "$PID" ]; then`
- [x] Kill process: `kill $PID`
- [x] Print message: `echo "Server stopped"`
- [x] Else: `echo "Server not running"`

#### 9.3 Create Deploy Script
- [x] Create `deploy.sh`
- [x] Add shebang: `#!/bin/bash`
- [x] Pull latest: `git pull origin main`
- [x] Install backend: `pip install -r backend/requirements.txt`
- [x] Create uploads: `mkdir -p backend/uploads`
- [x] Set permissions: `chmod 755 backend/uploads`
- [x] Restart server: `./stop.sh && ./start.sh`

#### 9.4 Create Backup Script
- [x] Create `backup.sh`
- [x] Add shebang: `#!/bin/bash`
- [x] Create dir: `BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"`
- [x] Make dir: `mkdir -p $BACKUP_DIR`
- [x] Backup uploads: `cp -r backend/uploads $BACKUP_DIR/`
- [x] Backup config: `cp backend/.env $BACKUP_DIR/`
- [x] Create archive: `tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR`
- [x] Remove dir: `rm -rf $BACKUP_DIR`
- [x] Print: `echo "Backup created: $BACKUP_DIR.tar.gz"`

#### 9.5 Make Scripts Executable
- [x] Chmod start: `chmod +x start.sh`
- [x] Chmod stop: `chmod +x stop.sh`
- [x] Chmod deploy: `chmod +x deploy.sh`
- [x] Chmod backup: `chmod +x backup.sh`
- [x] Test start: `./start.sh`
- [x] Test stop: `./stop.sh`

---

### 10. Monitoring Setup (25 min)

#### 10.1 Add Logging
- [x] In app.py, import: `import logging`
- [x] Import: `from logging.handlers import RotatingFileHandler`
- [x] Create logger: `if not app.debug:`
- [x] Create handler: `handler = RotatingFileHandler('app.log', maxBytes=10240, backupCount=10)`
- [x] Set format: `handler.setFormatter(logging.Formatter(...))`
- [x] Set level: `handler.setLevel(logging.INFO)`
- [x] Add handler: `app.logger.addHandler(handler)`

#### 10.2 Log Important Events
- [x] Log upload: `app.logger.info(f'File uploaded: {file_id} ({original_name})')`
- [x] Log convert: `app.logger.info(f'Conversion: {file_id} with {converter}')`
- [x] Log errors: `app.logger.error(f'Conversion failed: {str(e)}')`
- [x] Log cleanup: `app.logger.info(f'Cleaned up {count} old files')`

#### 10.3 Create Health Metrics
- [x] Add metrics dict: `metrics = {'uploads': 0, 'conversions': 0, 'errors': 0}`
- [x] In upload: `metrics['uploads'] += 1`
- [x] In convert: `metrics['conversions'] += 1`
- [x] On error: `metrics['errors'] += 1`
- [x] Add endpoint: `@app.route('/api/metrics')`
- [x] Return metrics: `return jsonify(metrics)`

#### 10.4 Add Uptime Tracking
- [x] Import: `from datetime import datetime`
- [x] Set start: `start_time = datetime.now()`
- [x] In metrics: `uptime = (datetime.now() - start_time).total_seconds()`
- [x] Add to response: `metrics['uptime_seconds'] = uptime`

#### 10.5 Create Monitor Script
- [x] Create `monitor.sh`
- [x] Add shebang: `#!/bin/bash`
- [x] Check health: `curl -s http://localhost:8000/health`
- [x] Check metrics: `curl -s http://localhost:8000/api/metrics`
- [x] Check disk: `df -h | grep uploads`
- [x] Check memory: `free -m`
- [x] Check processes: `ps aux | grep python`

---

### 11. Final Integration Tests (30 min)

#### 11.1 Test Complete Flow
- [x] Start backend: `cd backend && python app.py`
- [x] Open browser: `http://localhost:8000`
- [x] Upload test image
- [x] Adjust threshold slider
- [x] Select each converter
- [x] Convert image
- [x] Check SVG displays
- [x] Check metrics display
- [x] Download SVG
- [x] Open downloaded file

#### 11.2 Test Error Handling
- [x] Upload non-image file
- [x] Convert without uploading
- [x] Stop backend, try convert
- [x] Upload very large file
- [x] Upload corrupted image

#### 11.3 Performance Test
- [x] Upload 10 images rapidly
- [x] Convert all with different settings
- [x] Check response times
- [x] Monitor memory usage
- [x] Check for memory leaks

#### 11.4 Cross-Browser Test
- [x] Test in Chrome
- [x] Test in Firefox
- [x] Test in Safari
- [x] Test in Edge
- [x] Test on mobile browser

#### 11.5 Security Test
- [x] Try SQL injection in parameters
- [x] Try XSS in filename
- [x] Try path traversal in file_id
- [x] Check CORS headers
- [x] Verify rate limiting works

---

### 12. Documentation Finalization (20 min)

#### 12.1 Update README
- [x] Add project description
- [x] Add features list
- [x] Add requirements
- [x] Add installation steps
- [x] Add usage instructions
- [x] Add API documentation link
- [x] Add troubleshooting section

#### 12.2 Create DEPLOYMENT.md
- [x] Add deployment options
- [x] Add Docker instructions
- [x] Add manual deployment
- [x] Add environment variables
- [x] Add nginx configuration
- [x] Add SSL setup notes

#### 12.3 Create TESTING.md
- [x] Add test setup
- [x] Add unit test instructions
- [x] Add E2E test instructions
- [x] Add performance test guide
- [x] Add security test checklist

#### 12.4 Add License
- [x] Create LICENSE file
- [x] Choose license (MIT, Apache, etc.)
- [x] Add copyright notice
- [x] Add license text

#### 12.5 Final Cleanup
- [x] Remove debug prints
- [x] Remove test files
- [x] Update .gitignore
- [x] Commit all changes
- [x] Tag release: `git tag v1.0.0`

---

## Deployment Checklist

### Pre-Deployment
- [x] All tests pass
- [x] No console errors
- [x] Documentation complete
- [x] Security measures in place
- [x] Performance acceptable
- [x] Backup system ready

### Deployment Steps
- [x] Set production environment
- [x] Configure domain/DNS
- [x] Setup SSL certificate
- [x] Configure firewall
- [x] Start services
- [x] Verify endpoints work
- [x] Monitor logs

### Post-Deployment
- [x] Monitor metrics
- [x] Check error logs
- [x] Verify backups work
- [x] Test from external network
- [x] Update documentation with URL

---

## Success Criteria

✅ Frontend and Backend integrated
✅ All endpoints tested
✅ Security measures implemented
✅ Performance optimized
✅ Docker setup working
✅ Documentation complete
✅ Deployment scripts ready
✅ Monitoring in place
✅ Backup system working
✅ Production ready

---

## Time Estimate

- Backend-Frontend Integration: 45 minutes
- E2E Testing Setup: 40 minutes
- Comprehensive Tests: 50 minutes
- Performance Optimization: 40 minutes
- Security Hardening: 35 minutes
- Production Config: 30 minutes
- Docker Setup: 35 minutes
- API Documentation: 25 minutes
- Deployment Scripts: 30 minutes
- Monitoring Setup: 25 minutes
- Final Integration Tests: 30 minutes
- Documentation: 20 minutes

**Total: ~7 hours**

---

## Notes

- Test each integration point thoroughly
- Don't skip security measures
- Document everything
- Keep deployment scripts simple
- Test in production-like environment
- Have rollback plan ready
- Monitor after deployment