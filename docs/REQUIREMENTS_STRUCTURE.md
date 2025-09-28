# Requirements Structure

This document describes the new separated requirements structure for the SVG-AI project.

## Overview

The project now uses a modular requirements structure to support different deployment environments:

```
requirements/
├── base.txt     # Core dependencies for basic functionality
├── dev.txt      # Development tools and testing frameworks
├── prod.txt     # Production optimizations and monitoring
```

## File Descriptions

### `requirements/base.txt`
**Purpose**: Core dependencies required for basic SVG-AI functionality

**Includes**:
- Core Python packages (pillow, numpy, click, requests)
- VTracer conversion engine
- Image processing libraries (opencv-python, scikit-image)
- SVG manipulation tools (cairosvg, svgwrite, svgpathtools)
- Web API framework (fastapi, uvicorn)
- Data processing (pandas, pydantic)
- Performance utilities (joblib, tqdm)

**Use Case**: Minimal installation for production or containerized deployments

### `requirements/dev.txt`
**Purpose**: Development environment with all tools needed for coding, testing, and debugging

**Includes**:
- All base requirements (`-r base.txt`)
- Testing framework (pytest, pytest-cov, pytest-asyncio)
- Code quality tools (black, flake8, mypy)
- Development utilities (ipython)
- Analysis and visualization (matplotlib, seaborn, rich)

**Use Case**: Local development, CI/CD testing, code review

### `requirements/prod.txt`
**Purpose**: Production-optimized deployment with monitoring and performance enhancements

**Includes**:
- All base requirements (`-r base.txt`)
- Production WSGI server (gunicorn with gevent)
- System monitoring (psutil)
- Structured logging (structlog)
- Security enhancements (cryptography)
- Performance optimizations (httpx, pympler)
- Environment management (python-dotenv)

**Use Case**: Production deployment, staging environments, performance-critical deployments

## Installation Instructions

### Development Environment
```bash
# Install development environment (recommended for local work)
pip install -r requirements/dev.txt
```

### Production Environment
```bash
# Install production environment (for deployment)
pip install -r requirements/prod.txt
```

### Minimal Environment
```bash
# Install only core functionality
pip install -r requirements/base.txt
```

## Docker Integration

### Development Dockerfile
```dockerfile
FROM python:3.9-slim
COPY requirements/dev.txt /tmp/
RUN pip install -r /tmp/dev.txt
```

### Production Dockerfile
```dockerfile
FROM python:3.9-slim
COPY requirements/prod.txt /tmp/
RUN pip install -r /tmp/prod.txt
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Install dependencies (development)
  run: pip install -r requirements/dev.txt

- name: Run tests
  run: pytest

- name: Install dependencies (production)
  run: pip install -r requirements/prod.txt

- name: Test production build
  run: python -c "import backend.app; print('Production build OK')"
```

## Version Management

Each requirements file maintains the same version pinning strategy:
- **All versions are pinned** to specific releases
- **Compatibility is tested** across all environment files
- **Updates are coordinated** to maintain consistency

When updating versions:
1. Update `base.txt` first
2. Test with minimal environment
3. Update `dev.txt` and test development workflow
4. Update `prod.txt` and test production deployment
5. Update documentation if new packages are added

## Migration from Single requirements.txt

### For Existing Deployments
The original `requirements.txt` is preserved for backward compatibility but is no longer maintained.

**Migration path**:
```bash
# Replace old installation
pip uninstall -r requirements.txt -y
pip install -r requirements/prod.txt  # or dev.txt for development
```

### For Docker
Update Dockerfile to use appropriate requirements file:
```dockerfile
# Old
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# New (production)
COPY requirements/prod.txt /tmp/
RUN pip install -r /tmp/prod.txt
```

## Package Categories

### Core Functionality
- **vtracer**: PNG to SVG conversion engine
- **fastapi**: Web API framework
- **pillow**: Image processing
- **opencv-python**: Computer vision

### Development Only
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Type checking
- **matplotlib**: Data visualization

### Production Only
- **gunicorn**: Production WSGI server
- **psutil**: System monitoring
- **structlog**: Structured logging
- **pympler**: Memory profiling

## Troubleshooting

### Common Issues

1. **ImportError in production**: Check if package is in `base.txt`
2. **Missing development tools**: Install `dev.txt` instead of `base.txt`
3. **Docker build fails**: Ensure correct requirements file is copied
4. **Performance issues**: Consider `prod.txt` for optimized packages

### Dependency Conflicts
If conflicts arise between environments:
1. Update base requirements first
2. Test each environment file individually
3. Use `pip check` to validate compatibility
4. Update conflicting packages across all files