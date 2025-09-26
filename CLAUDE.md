# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNG to SVG converter using VTracer (Rust-based vectorization). The project converts raster logos to scalable vectors with quality metrics, caching, and web interface capabilities.

## Essential Commands

### Optimization Workflow
```bash
# Single file optimization with quality target
python optimize_iterative.py logo.png --target-ssim 0.9 --verbose

# With visual comparison and history
python optimize_iterative.py logo.png --save-comparison --save-history

# Batch optimization
python batch_optimize.py data/logos --target-ssim 0.85 --parallel 4

# With report generation
python batch_optimize.py data/logos --report results.json --save-comparisons
```

### Setup and Dependencies
```bash
# Use Python 3.9 virtual environment (VTracer compatibility)
source venv39/bin/activate

# Install VTracer (requires temp directory workaround on macOS)
export TMPDIR=/tmp
pip install vtracer

# Install all dependencies
pip install -r requirements.txt
```

### Core Conversion Commands
```bash
# Single file conversion
python convert.py data/logos/simple_geometric/circle_00.png

# With logo optimization
python convert.py logo.png --optimize-logo

# Batch conversion with parallel processing
python batch_convert.py data/logos --parallel 4

# Potrace alternative (black & white only)
python convert_potrace.py image.png --threshold 128
```

### Development Workflow
```bash
# Generate 50-logo test dataset
python scripts/create_full_dataset.py

# Run benchmarks
python benchmark.py --test-dir data/logos --report

# Quick test
make test-fast

# Start web interface
python web_server.py  # Visit http://localhost:8000

# Format and lint
make format lint
```

## Architecture

### Converter System
- **Base Pattern**: All converters inherit from `BaseConverter` in `converters/base.py`
- **VTracer Integration**: Primary converter using `vtracer.convert_image_to_svg_py()` which requires output path parameter (v0.6.11+)
- **Parameter Tuning**: VTracer accepts 8 key parameters (color_precision, corner_threshold, etc.) that dramatically affect output quality

### Quality Metrics System
- **SSIM Comparison**: Renders SVG back to PNG and compares with original
- **Metric Classes**: `ConversionMetrics` (basic) and `ComprehensiveMetrics` (full analysis)
- **Caching**: Hybrid memory LRU + disk cache in `utils/cache.py`

### Web Interface
- **FastAPI Server**: `web_server.py` with drag-and-drop upload
- **Background Tasks**: Async conversion processing
- **WebSocket Support**: Real-time progress updates

## VTracer Parameter Guidelines

For different logo types:
- **Simple Geometric**: `color_precision=3-4, corner_threshold=30`
- **Text-Based**: `color_precision=2, corner_threshold=20, path_precision=10`
- **Gradients**: `color_precision=8-10, layer_difference=8`
- **Complex**: `max_iterations=20, splice_threshold=60`

## Known Issues & Solutions

### VTracer Installation Fails
```bash
# macOS permission error fix
export TMPDIR=/tmp
pip install vtracer
```

### VTracer API Change
VTracer 0.6.11+ requires output path. The converter handles this with tempfile:
```python
with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
    vtracer.convert_image_to_svg_py(input_path, tmp.name, **params)
```

## Testing Approach

1. Always test with real logos from `data/logos/` categories
2. Measure quality with SSIM (target: >0.85 for simple, >0.70 for complex)
3. Check file size reduction (target: 50-80% smaller than PNG)
4. Verify SVG renders correctly by opening in browser

## Optimization System

The project includes an **automated optimization workflow** that iteratively tunes VTracer parameters to achieve target quality levels:

### Key Features:
- **Automatic Logo Type Detection**: Analyzes images to detect simple, text, gradient, or complex types
- **Iterative Parameter Tuning**: Adjusts 7 VTracer parameters based on quality feedback
- **Quality Metrics**: Real-time SSIM, MSE, and PSNR calculation
- **Visual Comparison**: Side-by-side comparison grids showing original, converted, and difference
- **Batch Processing**: Optimize entire directories with parallel processing
- **Detailed Reporting**: JSON reports with statistics and parameter recommendations

### Typical Results:
- Simple geometric logos: 98-99% SSIM
- Text-based logos: 99%+ SSIM
- Gradient logos: 97-98% SSIM
- Complex logos: 85-95% SSIM

## Project Status

**Completed (Week 1-2 Foundation + Optimization):**
- VTracer integration with parameter control
- 50-logo test dataset across 5 categories
- Quality metrics (SSIM, MSE, PSNR)
- Benchmark system with reporting
- FastAPI web interface
- Parallel batch processing
- Hybrid caching system
- **Iterative optimization workflow with auto-tuning**
- **Visual comparison and difference visualization**
- **Batch optimization with parallel processing**
- **SVG post-processing utilities**

**Next Phase Options:**
- Optimize VTracer parameters per logo type
- Add ML models (OmniSVG integration)
- Implement smart converter routing
- SVG post-processing pipeline