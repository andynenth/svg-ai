# SVG-AI Troubleshooting Guide

## Quick Diagnostic Commands

### System Health Check
```bash
# Overall system status
curl -s http://localhost/health | jq '.'

# Component status
curl -s http://localhost/api/classification-status | jq '.'

# Container status
docker-compose ps

# Resource usage
docker stats --no-stream
```

### Log Analysis
```bash
# Application logs
docker-compose logs -f svg-ai

# Error logs only
docker-compose logs svg-ai | grep -i error

# Recent logs with timestamps
docker-compose logs --tail=50 --timestamps svg-ai

# Redis logs
docker-compose logs redis

# Nginx logs
docker-compose logs nginx
```

## Common Issues and Solutions

### 1. Application Won't Start

#### Symptoms
- Container exits immediately
- Health check fails
- Port binding errors

#### Diagnostic Steps
```bash
# Check container status
docker-compose ps

# View startup logs
docker-compose logs svg-ai

# Check port conflicts
netstat -tulpn | grep :5000
```

#### Solutions
**Port Already in Use:**
```bash
# Kill process using port
sudo lsof -t -i:5000 | xargs kill -9

# Or change port in docker-compose.yml
ports:
  - "5001:5000"  # Use different external port
```

**Environment Variable Issues:**
```bash
# Check environment variables
docker-compose exec svg-ai env | grep FLASK

# Recreate with fresh environment
docker-compose down
docker-compose up -d
```

**Permission Issues:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh
```

### 2. High Response Times

#### Symptoms
- API responses >5 seconds
- Timeouts on complex images
- User complaints about slowness

#### Diagnostic Steps
```bash
# Check CPU usage
docker stats --no-stream

# Monitor memory usage
docker-compose exec svg-ai free -h

# Check queue depth
docker-compose logs svg-ai | grep "Processing.*images"

# Test response time
time curl -X POST http://localhost/api/convert -d @test_payload.json
```

#### Solutions
**High CPU Usage:**
```bash
# Scale horizontally
docker-compose up -d --scale svg-ai=3

# Increase resource limits
# In docker-compose.yml:
resources:
  limits:
    cpus: '2.0'
    memory: 4G
```

**Memory Issues:**
```bash
# Restart containers to clear memory
docker-compose restart svg-ai

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHALL

# Check for memory leaks
docker-compose exec svg-ai python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
"
```

**Large Image Processing:**
```bash
# Implement image preprocessing
# Add to application configuration:
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB limit
RESIZE_LARGE_IMAGES = True
MAX_DIMENSION = 2000
```

### 3. Redis Connection Issues

#### Symptoms
- Rate limiting not working
- Cache misses
- Connection refused errors

#### Diagnostic Steps
```bash
# Test Redis connectivity
docker-compose exec svg-ai redis-cli ping

# Check Redis status
docker-compose exec redis redis-cli info server

# Monitor Redis connections
docker-compose exec redis redis-cli client list
```

#### Solutions
**Redis Container Down:**
```bash
# Restart Redis
docker-compose restart redis

# Check Redis logs
docker-compose logs redis

# Verify data persistence
docker-compose exec redis redis-cli lastsave
```

**Connection Pool Issues:**
```bash
# Clear connection pools
docker-compose restart svg-ai

# Monitor connection count
watch 'docker-compose exec redis redis-cli client list | wc -l'
```

### 4. Rate Limiting Problems

#### Symptoms
- Rate limits not enforced
- Users getting 429 errors frequently
- Rate limit bypass

#### Diagnostic Steps
```bash
# Check rate limiting configuration
docker-compose exec svg-ai python -c "
from flask_limiter import Limiter
print('Rate limiter configured')
"

# Test rate limiting
for i in {1..15}; do
  curl -X POST http://localhost/api/convert -d @test.json
  echo "Request $i"
done
```

#### Solutions
**Rate Limiter Not Working:**
```bash
# Verify Redis connection for rate limiting
docker-compose exec svg-ai python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(r.ping())
"

# Check rate limit storage
docker-compose exec redis redis-cli keys "LIMITER*"
```

**Adjust Rate Limits:**
```python
# In backend/app.py, modify:
@limiter.limit("20 per minute")  # Increase limit
def convert():
    pass
```

### 5. Image Conversion Failures

#### Symptoms
- Conversion returns errors
- Poor quality output
- Specific image types failing

#### Diagnostic Steps
```bash
# Test with known good image
curl -X POST http://localhost/api/convert \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w 0 test_simple.png)'","format":"png"}'

# Check converter status
docker-compose exec svg-ai python -c "
try:
    import vtracer
    print('VTracer available')
except ImportError:
    print('VTracer not available')
"

# Test different converters
curl -X POST http://localhost/api/convert \
  -d '{"image":"...","converter":"potrace"}'
```

#### Solutions
**VTracer Installation Issues:**
```bash
# Reinstall VTracer
docker-compose exec svg-ai pip install --force-reinstall vtracer

# Check VTracer dependencies
docker-compose exec svg-ai python -c "
import subprocess
result = subprocess.run(['which', 'vtracer'], capture_output=True, text=True)
print(result.stdout)
"
```

**Image Format Issues:**
```bash
# Validate image format
docker-compose exec svg-ai python -c "
from PIL import Image
img = Image.open('problem_image.png')
print(f'Format: {img.format}, Mode: {img.mode}, Size: {img.size}')
"

# Convert problematic images
docker-compose exec svg-ai python -c "
from PIL import Image
img = Image.open('problem_image.png')
img = img.convert('RGB')
img.save('converted_image.png')
"
```

### 6. Memory Leaks

#### Symptoms
- Memory usage increases over time
- Container becomes unresponsive
- OOM (Out of Memory) kills

#### Diagnostic Steps
```bash
# Monitor memory over time
while true; do
  docker stats --no-stream | grep svg-ai
  sleep 60
done

# Check for memory leaks
docker-compose exec svg-ai python -c "
import gc, sys
print(f'Objects: {len(gc.get_objects())}')
print(f'Memory: {sys.getsizeof(gc.get_objects())} bytes')
"

# Profile memory usage
docker-compose exec svg-ai python -m memory_profiler app.py
```

#### Solutions
**Clear Caches Regularly:**
```bash
# Add memory monitoring script
cat > scripts/memory_monitor.sh << EOF
#!/bin/bash
while true; do
  MEMORY_USAGE=\$(docker stats --no-stream --format "{{.MemPerc}}" svg-ai_svg-ai_1)
  if [[ \${MEMORY_USAGE%\%} -gt 80 ]]; then
    echo "High memory usage: \$MEMORY_USAGE"
    docker-compose exec svg-ai python -c "
    import gc
    gc.collect()
    print('Garbage collection triggered')
    "
  fi
  sleep 300  # Check every 5 minutes
done
EOF
chmod +x scripts/memory_monitor.sh
```

**Implement Memory Limits:**
```yaml
# In docker-compose.yml
services:
  svg-ai:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### 7. SSL/TLS Issues

#### Symptoms
- HTTPS not working
- Certificate errors
- Mixed content warnings

#### Diagnostic Steps
```bash
# Check certificate status
openssl x509 -in ssl/certificate.crt -text -noout

# Test SSL connection
openssl s_client -connect localhost:443 -servername your-domain.com

# Check Nginx SSL configuration
docker-compose exec nginx nginx -t
```

#### Solutions
**Generate Self-Signed Certificates:**
```bash
# Create SSL directory and certificates
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/private.key \
  -out ssl/certificate.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

**Update Nginx Configuration:**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;

    location / {
        proxy_pass http://svg-ai:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 8. Database/Cache Corruption

#### Symptoms
- Inconsistent responses
- Cache hit/miss anomalies
- Data persistence issues

#### Diagnostic Steps
```bash
# Check Redis data integrity
docker-compose exec redis redis-cli info keyspace

# Test cache operations
docker-compose exec redis redis-cli set test_key test_value
docker-compose exec redis redis-cli get test_key

# Check for corrupted data
docker-compose exec redis redis-cli --scan | head -10
```

#### Solutions
**Clear Corrupted Cache:**
```bash
# Backup current data
docker-compose exec redis redis-cli save
docker cp $(docker-compose ps -q redis):/data/dump.rdb backup_dump.rdb

# Clear cache
docker-compose exec redis redis-cli flushall

# Restart Redis
docker-compose restart redis
```

**Restore from Backup:**
```bash
# Stop Redis
docker-compose stop redis

# Restore backup
docker cp backup_dump.rdb $(docker-compose ps -q redis):/data/dump.rdb

# Start Redis
docker-compose start redis
```

## Performance Troubleshooting

### Slow API Responses

#### Investigation Steps
```bash
# Enable request timing
export FLASK_DEBUG=1

# Monitor response times
docker-compose logs svg-ai | grep "Processing time"

# Profile specific endpoints
curl -w "@curl-format.txt" -o /dev/null -s \
  -X POST http://localhost/api/convert -d @test.json
```

#### Optimization Strategies
1. **Implement Caching:**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=128)
   def convert_cached(image_hash, parameters):
       return convert_image(image_hash, parameters)
   ```

2. **Optimize Image Processing:**
   ```python
   # Resize large images before processing
   if image.size[0] > 2000 or image.size[1] > 2000:
       image.thumbnail((2000, 2000), Image.LANCZOS)
   ```

3. **Use Async Processing:**
   ```python
   from celery import Celery

   app = Celery('svg-ai')

   @app.task
   def convert_async(image_data):
       return convert_image(image_data)
   ```

## Emergency Procedures

### Complete System Failure

1. **Immediate Response:**
   ```bash
   # Stop all services
   docker-compose down

   # Check disk space
   df -h

   # Check system load
   uptime
   ```

2. **Recovery Steps:**
   ```bash
   # Clear temporary files
   rm -rf /tmp/*

   # Restart with fresh containers
   docker-compose up -d --force-recreate

   # Monitor startup
   docker-compose logs -f
   ```

3. **Validation:**
   ```bash
   # Test health endpoint
   curl http://localhost/health

   # Test conversion
   curl -X POST http://localhost/api/convert -d @test.json

   # Check monitoring
   curl http://localhost:3000/api/health
   ```

### Data Recovery

1. **Backup Current State:**
   ```bash
   # Export current configuration
   docker-compose config > current_config.yml

   # Backup volumes
   docker run --rm -v svg-ai_redis_data:/data -v $(pwd):/backup \
     alpine tar czf /backup/redis_backup.tar.gz /data
   ```

2. **Restore from Backup:**
   ```bash
   # Restore configuration
   cp backup/docker-compose.yml docker-compose.yml

   # Restore data
   docker run --rm -v svg-ai_redis_data:/data -v $(pwd):/backup \
     alpine tar xzf /backup/redis_backup.tar.gz -C /
   ```

## Monitoring and Alerting

### Custom Health Checks
```bash
#!/bin/bash
# health_check.sh

# Check API responsiveness
if ! curl -f http://localhost/health >/dev/null 2>&1; then
    echo "CRITICAL: API health check failed"
    exit 2
fi

# Check response time
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost/health)
if (( $(echo "$RESPONSE_TIME > 5" | bc -l) )); then
    echo "WARNING: Slow response time: ${RESPONSE_TIME}s"
    exit 1
fi

echo "OK: System healthy, response time: ${RESPONSE_TIME}s"
exit 0
```

### Log Monitoring
```bash
# Monitor for critical errors
tail -f logs/application.log | grep -i "critical\|error\|exception"

# Watch for memory issues
docker-compose logs -f svg-ai | grep -i "memory\|oom"

# Monitor conversion failures
docker-compose logs -f svg-ai | grep "Conversion failed"
```

## Performance Optimization

### Query Optimization
```python
# Cache frequently used data
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_optimized_parameters(image_type):
    return parameter_lookup[image_type]
```

### Resource Management
```yaml
# Optimize resource allocation
services:
  svg-ai:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Connection Pooling
```python
# Optimize Redis connections
import redis.connection
redis.connection.ConnectionPool(
    host='redis',
    port=6379,
    max_connections=20,
    retry_on_timeout=True
)
```

## Getting Additional Help

### Log Collection for Support
```bash
# Collect comprehensive logs
mkdir -p support_logs
docker-compose logs --no-color > support_logs/container_logs.txt
docker stats --no-stream > support_logs/resource_usage.txt
docker-compose config > support_logs/configuration.yml
curl -s http://localhost/health > support_logs/health_status.json

# Create support package
tar czf support_package_$(date +%Y%m%d_%H%M%S).tar.gz support_logs/
```

### Diagnostic Information
When contacting support, include:
- System configuration (OS, Docker version)
- Application version and commit hash
- Error messages and stack traces
- Steps to reproduce the issue
- Recent changes or deployments
- Resource usage patterns

### Contact Information
- **Technical Support**: support@company.com
- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Documentation**: https://docs.svg-ai.com
- **GitHub Issues**: https://github.com/company/svg-ai/issues