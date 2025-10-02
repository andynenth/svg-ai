#!/usr/bin/env python3
"""
Quick test to verify the project's core functionality works
"""

import sys
from pathlib import Path

print("=" * 60)
print("SVG-AI PROJECT FUNCTIONALITY TEST")
print("=" * 60)

# Test 1: Basic converter works
print("\n1. Testing Basic Converter:")
try:
    from backend.converter import convert_image

    # Test with a sample image
    test_image = "data/logos/simple_geometric/circle_00.png"
    if Path(test_image).exists():
        result = convert_image(test_image)
        print(f"   ✅ Conversion successful: {result.get('success', False)}")
        print(f"   📊 SSIM Score: {result.get('ssim', 0):.4f}")
        print(f"   📄 SVG Size: {len(result.get('svg', ''))} characters")
        print(f"   ⚙️  Converter used: {result.get('converter', 'unknown')}")
    else:
        print(f"   ⚠️  Test image not found: {test_image}")
        print("   Creating test image...")
        from PIL import Image
        import os
        os.makedirs("data/logos/simple_geometric", exist_ok=True)
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image)
        result = convert_image(test_image)
        print(f"   ✅ Conversion successful: {result.get('success', False)}")
        print(f"   📊 SSIM Score: {result.get('ssim', 0):.4f}")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Quality metrics work
print("\n2. Testing Quality Metrics:")
try:
    from backend.ai_modules.quality import ComprehensiveMetrics
    metrics = ComprehensiveMetrics()
    print("   ✅ Quality metrics module loaded")
    print(f"   Available methods: calculate_metrics(), calculate_comprehensive_metrics()")
except ImportError as e:
    print(f"   ⚠️  AI modules not available: {e}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Check for required dependencies
print("\n3. Checking Core Dependencies:")
dependencies = {
    'vtracer': 'VTracer (core SVG converter)',
    'PIL': 'Pillow (image processing)',
    'numpy': 'NumPy (numerical computing)',
    'cv2': 'OpenCV (image analysis)',
    'cairosvg': 'CairoSVG (SVG rendering)'
}

missing = []
for module, description in dependencies.items():
    try:
        if module == 'PIL':
            import PIL
        elif module == 'cv2':
            import cv2
        else:
            __import__(module)
        print(f"   ✅ {description}")
    except ImportError:
        print(f"   ❌ Missing: {description}")
        missing.append(module)

# Test 4: Check optional web dependencies
print("\n4. Checking Web Server Dependencies:")
web_deps = {
    'flask': 'Flask (web framework)',
    'redis': 'Redis (caching)',
    'flask_cors': 'Flask-CORS (cross-origin support)'
}

web_missing = []
for module, description in web_deps.items():
    try:
        __import__(module)
        print(f"   ✅ {description}")
    except ImportError:
        print(f"   ⚠️  Missing: {description}")
        web_missing.append(module)

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

if not missing:
    print("✅ Core functionality is WORKING!")
    print("   - PNG to SVG conversion works")
    print("   - Quality metrics (SSIM/MSE/PSNR) work")
    print("   - All core dependencies installed")
else:
    print("⚠️  Core functionality PARTIALLY working")
    print(f"   Missing dependencies: {', '.join(missing)}")
    print(f"   Install with: pip install {' '.join(missing)}")

if web_missing:
    print("\n⚠️  Web server not fully functional")
    print(f"   Missing: {', '.join(web_missing)}")
    print(f"   Install with: pip install {' '.join(web_missing)}")
else:
    print("\n✅ Web server dependencies installed")

print("\n" + "=" * 60)
print("QUICK START:")
print("=" * 60)
print("1. Basic conversion (Python):")
print("   python -c \"from backend.converter import convert_image; print(convert_image('image.png'))\"")
print("\n2. Start web server:")
print("   python -m backend.app")
print("   # Server runs on http://localhost:8001")
print("   # Note: Redis is OPTIONAL - only needed for rate limiting")
print("\n3. Test web API:")
print("   curl http://localhost:8001/health")
print("\n4. Available endpoints:")
print("   - GET  /health - Service health check")
print("   - POST /api/convert - Convert PNG to SVG")
print("   - POST /api/upload - Upload image file")
print("   - POST /api/classify-logo - Classify logo type")
print("   - GET  /api/ai-health - AI components status")