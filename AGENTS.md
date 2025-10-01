# AGENTS.md - AI-Enhanced SVG Converter Project Guide

## 1. Project Overview

**SVG-AI** is an advanced PNG to SVG conversion system that leverages AI-enhanced parameter optimization for high-quality vectorization. The project combines VTracer (Rust-based vectorization engine) with machine learning techniques to automatically optimize conversion parameters based on image characteristics.

### Core Capabilities
- **AI-Enhanced Vectorization**: Automatic parameter optimization using feature extraction and ML models
- **Multi-Tier Processing**: Fast rule-based, ML-enhanced, and adaptive optimization methods
- **Quality Metrics**: Comprehensive SSIM, MSE, PSNR quality measurement
- **Batch Processing**: Parallel conversion of multiple images
- **Web Interface**: FastAPI-based drag-and-drop interface
- **Caching System**: Multi-layer caching for performance optimization
- **Docker Deployment**: Production-ready containerization

### Tech Stack
- **Backend**: Python 3.9+, FastAPI, VTracer (Rust)
- **AI/ML**: PyTorch, scikit-learn, XGBoost, OpenCV
- **Quality Metrics**: scikit-image, cairosvg
- **Frontend**: HTML5, JavaScript (drag-and-drop interface)
- **Database/Cache**: In-memory LRU + disk caching
- **Deployment**: Docker, Docker Compose
- **Testing**: pytest, pytest-cov

## 2. Repository Structure

```
svg-ai/
├── backend/                     # Core application logic
│   ├── ai_modules/             # AI components (classification, optimization, quality)
│   │   ├── classification.py   # Logo type classification (simple, text, gradient, complex)
│   │   ├── optimization.py     # Parameter optimization engines
│   │   ├── quality.py         # Quality measurement system
│   │   ├── feature_extraction.py # Image feature extraction
│   │   └── pipeline/          # Unified AI processing pipeline
│   ├── converters/            # Conversion implementations
│   │   ├── base.py           # Abstract converter interface
│   │   ├── vtracer_converter.py # VTracer integration
│   │   └── ai_enhanced_converter.py # AI-optimized converter
│   ├── utils/                 # Utility modules
│   │   ├── cache.py          # Hybrid caching system
│   │   ├── quality_metrics.py # SSIM/PSNR calculations
│   │   ├── parallel_processor.py # Parallel processing utilities
│   │   └── performance_monitor.py # Performance tracking
│   └── api/                   # API endpoints and routing
├── scripts/                   # Automation and testing scripts
│   ├── verify_ai_setup.py    # AI environment validation
│   ├── test_ai_*.py          # AI performance test suite
│   ├── create_full_dataset.py # Generate test datasets
│   └── install_ai_dependencies.sh # AI dependency installation
├── data/                      # Test datasets and logos
│   └── logos/                # 5-category logo test dataset
├── docs/                      # Comprehensive documentation
│   ├── phase1-foundation/    # Foundation setup guides
│   ├── phase2-core-ai/       # AI implementation guides
│   ├── phase3-6-advanced/    # Advanced AI architecture
│   └── system/               # System documentation
├── config/                    # Configuration files
│   ├── ai_production.py      # AI-enhanced production config
│   └── base_production.py    # Base production configuration
├── tests/                     # Test suite
├── templates/                # Web interface templates
├── static/                   # Static web assets
├── performance_reports/      # Performance analysis results
├── convert.py               # CLI conversion tool
├── batch_convert.py         # Batch processing tool
├── web_server.py            # FastAPI web server
├── benchmark.py             # Performance benchmarking
├── Dockerfile               # Base container configuration
├── Dockerfile.ai            # AI-enhanced container
├── docker-compose.yml       # Multi-service orchestration
├── Makefile                 # Build automation
├── requirements.txt         # Core Python dependencies
├── requirements_ai_phase1.txt # AI-specific dependencies
└── CLAUDE.md               # Project-specific guidance for Claude Code
```

## 3. Setup Instructions

### Prerequisites
- **Python 3.9+** (VTracer compatibility requirement)
- **Docker** (for containerized deployment)
- **Git** (for version control)

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd svg-ai

# Create Python 3.9 virtual environment (VTracer requirement)
python3.9 -m venv venv39
source venv39/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install AI dependencies (optional, for enhanced features)
pip install -r requirements_ai_phase1.txt

# Create test dataset
make dataset

# Verify setup
python scripts/verify_ai_setup.py
```

### VTracer Installation (macOS fix)
```bash
# Workaround for macOS permission issues
export TMPDIR=/tmp
pip install vtracer
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Setup pre-commit hooks
make format lint

# Run tests
make test

# Start development server
make server-dev
```

### Docker Setup
```bash
# Build base container
docker build -t svg-ai:latest .

# Build AI-enhanced container
docker build -f Dockerfile.ai -t svg-ai:ai .

# Run with Docker Compose
docker-compose up
```

## 4. Development Guidelines

### Code Style
- **Formatter**: Black (line length: 100)
- **Linter**: Flake8 (max-line-length: 100, ignore E501,W503)
- **Type Hints**: Required for all public functions
- **Docstrings**: Required for all modules, classes, and public functions

### Naming Conventions
- **Files**: snake_case (e.g., `ai_enhanced_converter.py`)
- **Classes**: PascalCase (e.g., `AIEnhancedConverter`)
- **Functions/Variables**: snake_case (e.g., `extract_features`)
- **Constants**: UPPER_CASE (e.g., `DEFAULT_QUALITY_TARGET`)

### Architecture Patterns
- **Lazy Loading**: Use factory functions from `backend/__init__.py` for AI components
- **Error Handling**: Comprehensive try-catch with fallback mechanisms
- **Caching**: Multi-layer caching (memory LRU + disk persistence)
- **Monitoring**: Performance tracking for all major operations
- **Converter Pattern**: All converters inherit from `BaseConverter`

### Import Performance
⚠️ **CRITICAL**: Use lazy loading for AI modules to avoid 13.93s import overhead:
```python
# ✅ CORRECT: Use factory functions
from backend import get_classification_module, get_optimization_engine

# ❌ AVOID: Direct imports cause slow startup
from backend.ai_modules.classification import ClassificationModule
```

### Quality Targets
- **Simple Logos**: SSIM > 0.95
- **Text Logos**: SSIM > 0.99
- **Gradient Logos**: SSIM > 0.97
- **Complex Logos**: SSIM > 0.85

## 5. Common Tasks

### Add a New Converter
```bash
# 1. Create new converter class
cat > backend/converters/my_converter.py << 'EOF'
from .base import BaseConverter

class MyConverter(BaseConverter):
    def __init__(self):
        super().__init__("My Converter")

    def convert(self, image_path: str, **kwargs) -> str:
        # Implementation here
        pass
EOF

# 2. Add tests
cat > tests/test_my_converter.py << 'EOF'
import pytest
from backend.converters.my_converter import MyConverter

def test_my_converter():
    converter = MyConverter()
    # Test implementation
EOF

# 3. Run tests
make test
```

### Add API Endpoint
```python
# In backend/api/routes.py
@router.post("/api/my-endpoint")
async def my_endpoint(request: MyRequest):
    # Implementation
    return {"result": "success"}
```

### Add AI Feature
```python
# 1. Implement in appropriate ai_modules/ file
# 2. Update lazy loading in backend/__init__.py
def get_my_ai_component():
    from .ai_modules.my_component import MyComponent
    return MyComponent()

# 3. Add factory function to __all__
__all__ = [..., "get_my_ai_component"]
```

### Run Performance Tests
```bash
# AI performance suite
python scripts/test_ai_model_loading.py
python scripts/test_ai_inference_speed.py
python scripts/test_ai_quality_improvement.py
python scripts/test_ai_memory_usage.py

# Container size test
scripts/test_ai_container_size.sh

# System benchmarks
make benchmark
```

### Optimize Parameters for Logo Type
```python
# Use the optimization workflow
python optimize_iterative.py logo.png --target-ssim 0.9 --verbose

# With visual comparison
python optimize_iterative.py logo.png --save-comparison --save-history

# Batch optimization
python batch_optimize.py data/logos --target-ssim 0.85 --parallel 4
```

## 6. Testing & Validation

### Test Structure
```bash
tests/
├── test_converters.py      # Converter functionality
├── test_quality_metrics.py # Quality measurement
├── test_ai_modules.py      # AI component tests
├── test_api.py            # API endpoint tests
├── test_cache.py          # Caching system tests
└── test_performance.py    # Performance benchmarks
```

### Running Tests
```bash
# Full test suite
make test

# Fast tests only (skip slow integration tests)
make test-fast

# With coverage report
make test-coverage

# Specific test file
pytest tests/test_converters.py -v

# AI-specific tests
python scripts/verify_ai_setup.py
```

### Performance Validation
```bash
# Benchmark against test dataset
make benchmark

# Quick performance check
make benchmark-quick

# Memory and speed profiling
python scripts/test_ai_memory_usage.py
python scripts/test_ai_inference_speed.py
```

### Quality Validation
```bash
# AI quality improvement test
python scripts/test_ai_quality_improvement.py

# Success metrics validation
python scripts/verify_success_metrics.py
```

## 7. Deployment Notes

### Production Deployment
```bash
# Build production containers
docker build -t svg-ai:production .
docker build -f Dockerfile.ai -t svg-ai:ai-production .

# Deploy with optimized configuration
docker-compose -f docker-compose.prod.yml up -d

# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/api/ai-status
```

### Environment Variables
```bash
# AI Configuration
AI_ENHANCED=true
MODEL_DIR=/app/models/production
CLASSIFIER_MODEL=classifier.pth
OPTIMIZER_MODEL=optimizer.xgb

# Performance Tuning
CACHE_SIZE=1000
PARALLEL_WORKERS=4
TARGET_QUALITY_SIMPLE=0.95
TARGET_QUALITY_COMPLEX=0.85

# Production Settings
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE=10MB
```

### Monitoring & Health Checks
- **Health Endpoint**: `/health` - Basic system health
- **AI Status**: `/api/ai-status` - AI component health
- **Metrics**: Performance metrics via `/api/metrics`
- **Logs**: Structured logging with performance timings

## 8. Cautions / Do Not Touch

### 🚨 Critical Files - Do Not Modify
```bash
# VTracer binary and installation
venv39/lib/python3.9/site-packages/vtracer/

# Generated model files
models/*.pth
models/*.xgb
models/*.pkl

# Cache directories (auto-managed)
cache/
temp/
__pycache__/

# Performance reports (auto-generated)
performance_reports/
results/*.json

# Virtual environment
venv39/
```

### ⚠️ Sensitive Configurations
```bash
# Production secrets
.env
config/secrets.py

# Docker registry credentials
.docker/

# CI/CD configuration
.github/workflows/
```

### 🔧 Auto-Generated Files
```bash
# Test coverage reports
htmlcov/
.coverage

# Documentation builds
docs/_build/
*.html (from pydoc)

# Benchmark results
results/benchmark_*.json
results/comparison_*.png
```

### 📋 Import Performance Critical
⚠️ **Never import AI modules directly** - always use lazy loading factory functions from `backend/__init__.py` to maintain sub-second startup times.

### 🧪 Test Dataset Integrity
- **Do not modify** files in `data/logos/` - these are reference datasets
- **Do not commit** temporary test files or generated SVGs
- **Use temp directories** for test artifacts

### 🐳 Container Build Context
- **Dockerfile** and **Dockerfile.ai** are optimized for layer caching
- **Do not add** large files to Docker context
- **Use** `.dockerignore` to exclude unnecessary files

---

## Quick Reference Commands

```bash
# Development
make help              # Show all available commands
make setup            # Initial project setup
make test             # Run test suite
make server-dev       # Start development server
make format lint      # Code formatting and linting

# Conversion
make convert FILE=image.png    # Convert single image
make batch DIR=directory       # Batch convert directory
make benchmark                 # Run performance benchmarks

# AI Testing
python scripts/verify_ai_setup.py           # Verify AI environment
python scripts/test_ai_inference_speed.py   # Test AI performance
python scripts/test_ai_quality_improvement.py # Test quality gains

# Production
docker-compose up     # Start full system
make docker-build     # Build production container
make clean           # Clean generated files
```

This guide provides autonomous coding agents with comprehensive context for contributing effectively to the SVG-AI project while maintaining code quality, performance, and architectural integrity.