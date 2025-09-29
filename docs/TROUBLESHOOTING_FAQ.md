# Troubleshooting and FAQ

## Overview

This document provides solutions to common issues, frequently asked questions, and debugging guidance for the SVG-AI Converter system.

## Quick Diagnostics

### System Health Check

Run this quick diagnostic to identify common issues:

```bash
# Check Python version
python --version  # Should be 3.9.x

# Check VTracer installation
python -c "import vtracer; print('VTracer OK')"

# Check AI modules (optional)
python3 scripts/verify_ai_setup.py

# Test basic conversion
python -c "
from backend.converters.vtracer_converter import VTracerConverter
print('Basic converter test:', 'PASS' if VTracerConverter() else 'FAIL')
"

# Check web server
curl http://localhost:8001/health
```

### Log Analysis

Check logs for common error patterns:

```bash
# View recent errors
tail -100 logs/svg-ai.log | grep ERROR

# Monitor real-time logs
tail -f logs/svg-ai.log

# Search for specific errors
grep -i "vtracer\|import\|timeout" logs/svg-ai.log
```

## Installation Issues

### VTracer Installation Problems

**Problem:** VTracer installation fails with permission errors

```
ERROR: Could not install packages due to an EnvironmentError
```

**Solution:**
```bash
# macOS: Set temporary directory
export TMPDIR=/tmp
pip install vtracer

# Linux: Install build dependencies
sudo apt-get install build-essential
pip install vtracer

# Windows: Use conda
conda install -c conda-forge vtracer
```

**Problem:** VTracer installation fails with Rust compilation errors

**Solution:**
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Retry VTracer installation
pip install vtracer
```

### Python Version Issues

**Problem:** "Python 3.11 not supported" or compatibility warnings

**Solution:**
```bash
# Use Python 3.9 specifically
python3.9 -m venv venv39
source venv39/bin/activate
pip install vtracer

# Verify version
python --version  # Must be 3.9.x
```

### Dependency Conflicts

**Problem:** Package version conflicts during installation

**Solution:**
```bash
# Create clean environment
python3.9 -m venv fresh_env
source fresh_env/bin/activate

# Install in specific order
pip install vtracer
pip install -r requirements.txt
pip install -r requirements_ai_phase1.txt
```

## Runtime Errors

### VTracer Conversion Failures

**Problem:** `pyo3_runtime.PanicException: assertion failed`

**Symptoms:**
```
thread '<unnamed>' panicked at assertion failed: is_same_color_a < 8
```

**Cause:** Invalid VTracer parameters or corrupted image data

**Solution:**
```python
# Use conservative parameters
safe_params = {
    'color_precision': 6,
    'layer_difference': 16,
    'corner_threshold': 60,
    'path_precision': 5
}

# Validate image before conversion
def validate_image(image_path):
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False
```

**Problem:** `No image file found at specified input path`

**Cause:** File path issues or file permissions

**Solution:**
```python
import os
from pathlib import Path

def debug_file_path(image_path):
    print(f"Path exists: {os.path.exists(image_path)}")
    print(f"Is file: {os.path.isfile(image_path)}")
    print(f"Readable: {os.access(image_path, os.R_OK)}")
    print(f"Absolute path: {Path(image_path).resolve()}")
```

### AI Module Issues

**Problem:** AI modules not found or import errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
ImportError: cannot import name 'FeatureExtractor'
```

**Solution:**
```bash
# Install AI dependencies
pip install -r requirements_ai_phase1.txt

# Or install manually
pip install torch torchvision scikit-learn

# Verify installation
python3 scripts/verify_ai_setup.py
```

**Problem:** AI analysis returns empty features

**Cause:** Image processing failures or invalid input

**Solution:**
```python
# Debug AI processing
from backend.ai_modules.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()

# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with known good image
features = extractor.extract_features('test_image.png')
print("Features:", features)
```

### Memory Issues

**Problem:** Out of memory errors during conversion

**Symptoms:**
```
MemoryError: Unable to allocate array
Process killed (OOM)
```

**Solution:**
```python
# Reduce image size before processing
from PIL import Image

def resize_if_large(image_path, max_size=1000):
    img = Image.open(image_path)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        img.save(image_path)

# Configure memory limits
import resource
resource.setrlimit(resource.RLIMIT_AS, (2048*1024*1024, -1))  # 2GB limit
```

### Web Server Issues

**Problem:** Flask server not starting

**Symptoms:**
```
ModuleNotFoundError: No module named 'converter'
Address already in use
```

**Solution:**
```bash
# Fix import path
export PYTHONPATH=/path/to/svg-ai:$PYTHONPATH

# Kill existing process
lsof -i :8001
kill -9 <PID>

# Start with explicit path
cd /path/to/svg-ai
python backend/app.py
```

**Problem:** File upload failures

**Symptoms:**
```
413 Request Entity Too Large
400 Bad Request: Invalid file format
```

**Solution:**
```python
# Check Flask configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Verify file validation
def debug_file_upload(file_content, filename):
    print(f"File size: {len(file_content)} bytes")
    print(f"Magic bytes: {file_content[:8]}")
    print(f"Extension: {filename.split('.')[-1]}")
```

## Performance Issues

### Slow Conversion Times

**Problem:** Conversions taking too long (>10 seconds)

**Diagnosis:**
```python
import time
from backend.converters.vtracer_converter import VTracerConverter

def profile_conversion(image_path):
    converter = VTracerConverter()

    start = time.time()
    result = converter.convert_with_metrics(image_path)
    total_time = time.time() - start

    print(f"Conversion time: {total_time:.2f}s")
    print(f"Success: {result['success']}")
    if result['svg']:
        print(f"SVG size: {len(result['svg'])} bytes")
```

**Solutions:**
- Reduce image size: `color_precision=4` instead of 8
- Lower detail: `corner_threshold=80` for smoother output
- Reduce iterations: `max_iterations=10` instead of 20
- Use faster converter: `converter='potrace'` for simple images

### High Memory Usage

**Problem:** Memory usage growing over time

**Diagnosis:**
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Monitor during conversions
monitor_memory()
```

**Solutions:**
- Enable garbage collection: `import gc; gc.collect()`
- Clear caches periodically
- Restart workers in production: `--max-requests 1000`
- Use process pooling instead of threading

### Cache Performance

**Problem:** Cache not improving performance

**Diagnosis:**
```python
from backend.ai_modules.advanced_cache import MultiLevelCache

cache = MultiLevelCache()
stats = cache.get_comprehensive_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Total entries: {stats['total_entries']}")
```

**Solutions:**
- Increase cache size: `max_size=2000`
- Adjust TTL: `ttl=7200` (2 hours)
- Enable distributed cache with Redis
- Clear corrupted cache: `cache.clear()`

## Configuration Issues

### Environment Variables

**Problem:** Configuration not being applied

**Diagnosis:**
```bash
# Check environment variables
env | grep -E "FLASK|AI|CACHE"

# Verify in Python
python -c "
import os
print('FLASK_ENV:', os.environ.get('FLASK_ENV'))
print('AI_ENABLED:', os.environ.get('AI_ENABLED'))
"
```

**Solution:**
```bash
# Create .env file
cat > .env << EOF
FLASK_ENV=production
AI_ENABLED=true
CACHE_ENABLED=true
MAX_CONTENT_LENGTH=16777216
EOF

# Load in application
from dotenv import load_dotenv
load_dotenv()
```

### File Permissions

**Problem:** Permission denied errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'uploads/file.png'
OSError: cannot write to directory
```

**Solution:**
```bash
# Fix upload directory permissions
sudo chown -R $USER:$USER uploads/
chmod 755 uploads/

# Fix cache directory
mkdir -p cache/
chmod 755 cache/

# Fix log directory
mkdir -p logs/
chmod 755 logs/
```

## API Issues

### Upload Failures

**Problem:** File uploads failing via API

**Diagnosis:**
```bash
# Test upload directly
curl -X POST \
  -F "file=@test.png" \
  http://localhost:8001/api/upload \
  -v

# Check content type
curl -H "Content-Type: multipart/form-data" \
  -F "file=@test.png" \
  http://localhost:8001/api/upload
```

**Common Issues:**
- Missing Content-Type header
- File size exceeds limit
- Invalid file format
- CORS issues in browser

### Conversion API Errors

**Problem:** Conversion API returning errors

**Diagnosis:**
```bash
# Test conversion with minimal parameters
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_id":"test123","converter":"vtracer"}' \
  http://localhost:8001/api/convert \
  -v
```

**Common Issues:**
- Invalid file_id
- Unsupported parameters
- Timeout on large files
- Missing Content-Type header

## Frequently Asked Questions

### General Questions

**Q: What image formats are supported?**

A: PNG and JPEG are fully supported. GIF, BMP, and TIFF are supported but may require conversion to PNG first.

**Q: What's the maximum file size?**

A: Default limit is 16MB. This can be configured via `MAX_CONTENT_LENGTH` environment variable.

**Q: Can I process multiple images simultaneously?**

A: Yes, the system supports parallel processing. Use the batch processing APIs or multiple concurrent requests.

### AI Features

**Q: How do I know if AI enhancement is working?**

A: Check the response for `ai_enhanced: true` and look for classification and feature data.

**Q: Can I disable AI features?**

A: Yes, set `AI_ENABLED=false` in environment or `ai_disable=true` in API requests.

**Q: What if AI classification is wrong?**

A: You can override with manual parameters. AI suggestions are optimizations, not requirements.

### Performance

**Q: How can I improve conversion speed?**

A:
- Use simpler parameters (`color_precision=4`)
- Resize large images before conversion
- Enable caching for repeated conversions
- Use appropriate converter for image type

**Q: Why is the first conversion slow?**

A: Initial conversions include model loading and cache warming. Subsequent conversions are faster.

**Q: How much memory does the system use?**

A: Base usage is ~200MB. Each conversion uses 50-200MB depending on image size and parameters.

### Deployment

**Q: Can I run this in Docker?**

A: Yes, see the deployment guide for Docker and docker-compose configurations.

**Q: What about Kubernetes?**

A: Kubernetes deployment is supported with provided YAML configurations.

**Q: How do I scale horizontally?**

A: Use load balancers with multiple application instances. Ensure shared cache (Redis) for optimal performance.

### Security

**Q: Is file upload secure?**

A: Yes, files are validated by content (magic bytes), size limits are enforced, and paths are sanitized.

**Q: What about malicious images?**

A: The system validates file headers and uses safe processing libraries. Very malformed files are rejected.

**Q: Can I run this on a public server?**

A: Yes, but enable proper authentication, rate limiting, and monitoring for production use.

## Advanced Debugging

### Enable Debug Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable specific component debugging
logging.getLogger('backend.converters').setLevel(logging.DEBUG)
logging.getLogger('backend.ai_modules').setLevel(logging.DEBUG)
```

### Performance Profiling

```python
import cProfile
import pstats
from backend.converters.vtracer_converter import VTracerConverter

def profile_conversion():
    converter = VTracerConverter()

    profiler = cProfile.Profile()
    profiler.enable()

    result = converter.convert_with_metrics('test.png')

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

profile_conversion()
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_conversion():
    from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter

    converter = AIEnhancedSVGConverter()
    result = converter.convert_with_ai_analysis('large_image.png')
    return result

memory_intensive_conversion()
```

## Getting Help

### Community Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the complete documentation set
- **Examples**: Review example scripts in the `examples/` directory

### Professional Support

For production deployments requiring professional support:

1. **Performance Optimization**: Custom tuning for your use case
2. **Security Audits**: Comprehensive security review
3. **Custom Features**: Development of specialized converters
4. **Training**: Team training on system administration

### Reporting Issues

When reporting issues, include:

1. **Environment Details**: OS, Python version, VTracer version
2. **Error Messages**: Complete error output with stack traces
3. **Reproduction Steps**: Minimal example to reproduce the issue
4. **Sample Files**: Non-sensitive test images that trigger the issue
5. **Configuration**: Relevant configuration settings (remove secrets)

### Issue Template

```
**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.9.16]
- VTracer: [e.g., 0.6.11]
- SVG-AI Version: [e.g., commit hash]

**Issue Description:**
[Clear description of the problem]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Error Output:**
```
[Include full error messages and stack traces]
```

**Additional Context:**
[Any other relevant information]
```