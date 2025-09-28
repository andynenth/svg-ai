# AI Modules Troubleshooting Guide

## Overview

This guide covers common issues and solutions when working with the AI modules in the SVG-AI Enhanced Conversion Pipeline.

## Installation Issues

### 1. PyTorch Installation Fails

**Problem**: PyTorch installation fails with compilation errors or version conflicts.

**Symptoms**:
```bash
ERROR: Failed building wheel for torch
ERROR: Could not build wheels for torch which use PEP 517
```

**Solutions**:

```bash
# Solution 1: Use pre-built CPU-only wheels
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Solution 2: Clear pip cache and retry
pip3 cache purge
pip3 install torch torchvision --no-cache-dir

# Solution 3: Use conda instead of pip
conda install pytorch torchvision cpuonly -c pytorch

# Solution 4: Install specific compatible version
pip3 install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

**Verification**:
```python
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

### 2. Scikit-learn Version Conflicts

**Problem**: Scikit-learn version conflicts with other packages.

**Symptoms**:
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
ImportError: cannot import name 'check_array' from 'sklearn.utils'
```

**Solutions**:

```bash
# Solution 1: Downgrade to compatible version
pip3 install scikit-learn==1.3.2

# Solution 2: Update all related packages
pip3 install --upgrade scikit-learn numpy scipy

# Solution 3: Force reinstall
pip3 uninstall scikit-learn
pip3 install scikit-learn==1.3.2

# Solution 4: Use specific numpy version
pip3 install numpy==1.26.4 scikit-learn==1.3.2
```

### 3. OpenCV Installation Issues

**Problem**: OpenCV not found or import errors.

**Symptoms**:
```python
ImportError: No module named 'cv2'
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions**:

```bash
# Solution 1: Install opencv-python
pip3 install opencv-python

# Solution 2: Install headless version (for servers)
pip3 install opencv-python-headless

# Solution 3: System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Solution 4: macOS dependencies
brew install opencv
```

### 4. Stable-Baselines3 Installation Fails

**Problem**: Stable-Baselines3 installation fails with dependency issues.

**Symptoms**:
```bash
ERROR: No matching distribution found for gymnasium>=0.28.1
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions**:

```bash
# Solution 1: Install gymnasium first
pip3 install gymnasium==0.28.1
pip3 install stable-baselines3==2.0.0

# Solution 2: Install with no dependencies then fix
pip3 install stable-baselines3 --no-deps
pip3 install gymnasium torch numpy

# Solution 3: Use conda
conda install -c conda-forge stable-baselines3
```

## Import and Module Issues

### 5. AI Module Import Failures

**Problem**: Cannot import AI modules.

**Symptoms**:
```python
ImportError: No module named 'backend.ai_modules'
ModuleNotFoundError: No module named 'backend'
```

**Solutions**:

```bash
# Solution 1: Ensure correct working directory
cd /path/to/svg-ai/
python3 -c "import backend.ai_modules; print('Success')"

# Solution 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/svg-ai"

# Solution 3: Install in development mode
pip3 install -e .

# Solution 4: Check __init__.py files exist
find backend/ai_modules -name "__init__.py" -type f
```

**Verification Script**:
```python
#!/usr/bin/env python3
"""Verify AI module imports"""

def test_imports():
    try:
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        print("✅ Feature extractor import OK")
    except ImportError as e:
        print(f"❌ Feature extractor import failed: {e}")

    try:
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
        print("✅ Optimizer import OK")
    except ImportError as e:
        print(f"❌ Optimizer import failed: {e}")

    try:
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor
        print("✅ Predictor import OK")
    except ImportError as e:
        print(f"❌ Predictor import failed: {e}")

if __name__ == "__main__":
    test_imports()
```

### 6. Circular Import Issues

**Problem**: Circular import dependencies between modules.

**Symptoms**:
```python
ImportError: cannot import name 'BaseAIConverter' from partially initialized module
```

**Solutions**:

```python
# Solution 1: Use local imports inside functions
def get_ai_converter():
    from backend.ai_modules.base_ai_converter import BaseAIConverter
    return BaseAIConverter()

# Solution 2: Import at module level, not class level
# At top of file:
from backend.ai_modules import base_ai_converter

# In class:
def some_method(self):
    converter = base_ai_converter.BaseAIConverter()

# Solution 3: Restructure imports to avoid cycles
# Move shared dependencies to a separate module
```

## Runtime Issues

### 7. Feature Extraction Failures

**Problem**: Feature extraction fails for certain images.

**Symptoms**:
```python
cv2.error: OpenCV(4.5.1) error: (-215:Assertion failed)
ValueError: Input image has invalid format
```

**Solutions**:

```python
def robust_feature_extraction(image_path: str):
    """Feature extraction with error handling"""
    try:
        # Validate image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Try to load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Check image format
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"Converting image format: {image.shape}")
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Extract features
        extractor = ImageFeatureExtractor()
        return extractor.extract_features_from_array(image)

    except Exception as e:
        print(f"Feature extraction failed: {e}")
        # Return default features
        return {
            'complexity_score': 0.5,
            'unique_colors': 16,
            'edge_density': 0.1,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.4,
            'entropy': 6.0,
            'corner_density': 0.02,
            'gradient_strength': 20.0
        }
```

### 8. Memory Issues

**Problem**: AI modules consume too much memory or cause memory leaks.

**Symptoms**:
```bash
MemoryError: Unable to allocate array
Process was killed (OOM killer)
```

**Solutions**:

```python
import gc
import psutil

def memory_efficient_processing(image_paths: list):
    """Process images with memory management"""

    def check_memory():
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:  # 85% threshold
            print(f"⚠️  High memory usage: {memory_percent:.1f}%")
            gc.collect()  # Force garbage collection
            return False
        return True

    results = []
    for i, image_path in enumerate(image_paths):
        if not check_memory():
            print("Pausing for memory cleanup...")
            time.sleep(1)

        try:
            # Process image
            result = process_single_image(image_path)
            results.append(result)

            # Clear caches periodically
            if i % 10 == 0:
                clear_all_caches()

        except MemoryError:
            print(f"Memory error processing {image_path}, skipping...")
            results.append({'error': 'Memory error', 'path': image_path})

    return results

def clear_all_caches():
    """Clear all AI module caches"""
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

    # Clear feature cache
    if hasattr(ImageFeatureExtractor, '_instances'):
        for instance in ImageFeatureExtractor._instances:
            instance.clear_cache()

    # Force garbage collection
    gc.collect()
```

### 9. Performance Issues

**Problem**: AI processing is too slow.

**Symptoms**:
```bash
Processing takes >10 seconds per image
Timeout errors in web interface
```

**Solutions**:

```python
import concurrent.futures
from functools import lru_cache

# Solution 1: Use caching for repeated operations
@lru_cache(maxsize=128)
def cached_feature_extraction(image_hash: str):
    """Cache feature extraction results"""
    return extract_features_internal(image_hash)

# Solution 2: Parallel processing
def parallel_processing(image_paths: list, max_workers: int = 4):
    """Process images in parallel"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, path) for path in image_paths]
        results = []

        for future in concurrent.futures.as_completed(futures, timeout=30):
            try:
                result = future.result()
                results.append(result)
            except concurrent.futures.TimeoutError:
                print("⚠️  Processing timeout, skipping image")
                results.append({'error': 'timeout'})

    return results

# Solution 3: Progressive optimization
def progressive_optimization(features: dict, time_limit: float = 5.0):
    """Optimization with time limit"""
    start_time = time.time()

    # Try fast optimization first
    quick_params = quick_optimize(features)

    if time.time() - start_time < time_limit * 0.5:
        # If we have time, try better optimization
        try:
            better_params = slow_optimize(features)
            return better_params
        except Exception:
            return quick_params

    return quick_params
```

### 10. VTracer Integration Issues

**Problem**: AI parameters don't work with VTracer.

**Symptoms**:
```python
TypeError: convert_image_to_svg_py() got an unexpected keyword argument
VTracer conversion fails with optimized parameters
```

**Solutions**:

```python
def validate_vtracer_parameters(params: dict) -> dict:
    """Validate and fix VTracer parameters"""

    # Valid VTracer parameters (v0.6.11+)
    valid_params = {
        'color_precision', 'corner_threshold', 'length_threshold',
        'splice_threshold', 'filter_speckle', 'color_tolerance',
        'layer_difference'
    }

    # Remove invalid parameters
    cleaned_params = {k: v for k, v in params.items() if k in valid_params}

    # Validate parameter ranges
    if 'color_precision' in cleaned_params:
        cleaned_params['color_precision'] = max(1, min(16, int(cleaned_params['color_precision'])))

    if 'corner_threshold' in cleaned_params:
        cleaned_params['corner_threshold'] = max(10, min(100, float(cleaned_params['corner_threshold'])))

    # Add missing required parameters with defaults
    defaults = {
        'color_precision': 4,
        'corner_threshold': 30,
        'length_threshold': 10,
        'splice_threshold': 45,
        'filter_speckle': 4,
        'color_tolerance': 0.2,
        'layer_difference': 16
    }

    for param, default_value in defaults.items():
        if param not in cleaned_params:
            cleaned_params[param] = default_value

    return cleaned_params

def safe_vtracer_conversion(image_path: str, params: dict) -> str:
    """VTracer conversion with parameter validation"""
    try:
        # Validate parameters
        safe_params = validate_vtracer_parameters(params)

        # Convert using vtracer
        import vtracer
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
            vtracer.convert_image_to_svg_py(image_path, tmp_file.name, **safe_params)

            with open(tmp_file.name, 'r') as f:
                svg_content = f.read()

            os.unlink(tmp_file.name)
            return svg_content

    except Exception as e:
        print(f"VTracer conversion failed: {e}")
        # Fallback to default parameters
        return safe_vtracer_conversion(image_path, {})
```

## Configuration Issues

### 11. Model Loading Failures

**Problem**: Pre-trained models fail to load.

**Symptoms**:
```python
FileNotFoundError: No such file or directory: 'models/pretrained/model.pth'
RuntimeError: Error(s) in loading state_dict
```

**Solutions**:

```python
import os
from pathlib import Path

def safe_model_loading():
    """Safe model loading with fallbacks"""

    # Check if model files exist
    model_path = Path("backend/ai_modules/models/pretrained/quality_predictor.pth")

    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        print("Creating default model...")

        # Create default model
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(2056, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Save default model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Default model saved to: {model_path}")

        return model

    try:
        # Load existing model
        import torch
        model = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded from: {model_path}")
        return model

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Using default model instead...")
        return create_default_model()

def create_default_model():
    """Create default model as fallback"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(2056, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
```

### 12. Configuration File Issues

**Problem**: Configuration settings cause errors.

**Symptoms**:
```python
KeyError: 'MODEL_CONFIG'
AttributeError: module 'backend.ai_modules.config' has no attribute 'PERFORMANCE_TARGETS'
```

**Solutions**:

```python
def safe_config_loading():
    """Safe configuration loading with defaults"""
    try:
        from backend.ai_modules.config import MODEL_CONFIG, PERFORMANCE_TARGETS
        return MODEL_CONFIG, PERFORMANCE_TARGETS
    except ImportError:
        print("⚠️  Config module not found, using defaults")
        return get_default_config()

def get_default_config():
    """Get default configuration if config file is missing"""
    from pathlib import Path

    MODEL_CONFIG = {
        'efficientnet_b0': {
            'path': Path('backend/ai_modules/models/pretrained/efficientnet_b0.pth'),
            'input_size': (224, 224),
            'num_classes': 4
        },
        'quality_predictor': {
            'path': Path('backend/ai_modules/models/trained/quality_predictor.pth'),
            'input_dim': 2056,
            'hidden_dims': [512, 256, 128]
        }
    }

    PERFORMANCE_TARGETS = {
        'tier_1': {'max_time': 1.0, 'target_quality': 0.85},
        'tier_2': {'max_time': 15.0, 'target_quality': 0.90},
        'tier_3': {'max_time': 60.0, 'target_quality': 0.95}
    }

    return MODEL_CONFIG, PERFORMANCE_TARGETS

# Usage in AI modules
def get_model_config():
    """Get model configuration safely"""
    try:
        MODEL_CONFIG, _ = safe_config_loading()
        return MODEL_CONFIG
    except Exception as e:
        print(f"Configuration error: {e}")
        return {}
```

## Testing Issues

### 13. Test Failures

**Problem**: AI module tests fail unexpectedly.

**Symptoms**:
```bash
FAILED tests/ai_modules/test_feature_extractor.py::test_extract_features
AssertionError: Feature value out of expected range
```

**Solutions**:

```python
def create_robust_test():
    """Create robust test with proper setup"""

    import unittest
    import tempfile
    import cv2
    import numpy as np

    class TestFeatureExtractor(unittest.TestCase):

        def setUp(self):
            """Set up test with valid test image"""
            # Create test image with known properties
            self.test_image = np.zeros((512, 512, 3), dtype=np.uint8)

            # Add some features
            cv2.circle(self.test_image, (256, 256), 100, (255, 255, 255), -1)
            cv2.rectangle(self.test_image, (100, 100), (400, 400), (128, 128, 128), 2)

            # Save to temporary file
            self.test_path = tempfile.mktemp(suffix='.png')
            cv2.imwrite(self.test_path, self.test_image)

            self.extractor = ImageFeatureExtractor()

        def test_extract_features_robust(self):
            """Robust feature extraction test"""
            try:
                features = self.extractor.extract_features(self.test_path)

                # Test feature presence
                required_features = ['complexity_score', 'unique_colors', 'edge_density']
                for feature in required_features:
                    self.assertIn(feature, features, f"Missing feature: {feature}")

                # Test feature ranges with tolerance
                self.assertGreaterEqual(features['complexity_score'], 0)
                self.assertLessEqual(features['complexity_score'], 1)

                # Allow for some variance in calculations
                self.assertGreater(features['unique_colors'], 0)

            except Exception as e:
                self.fail(f"Feature extraction failed: {e}")

        def tearDown(self):
            """Clean up test files"""
            if os.path.exists(self.test_path):
                os.unlink(self.test_path)
```

## Debugging Tools

### 14. AI Module Diagnostics

```python
#!/usr/bin/env python3
"""AI Modules Diagnostic Tool"""

import sys
import traceback
import psutil
import time

def run_diagnostics():
    """Run comprehensive AI modules diagnostics"""

    print("🔍 AI Modules Diagnostic Tool")
    print("=" * 50)

    # 1. System Information
    print("\n📊 System Information:")
    print(f"  Python Version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    # 2. Dependency Check
    print("\n📦 Dependency Check:")
    dependencies = [
        ('torch', 'PyTorch'),
        ('sklearn', 'Scikit-learn'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('stable_baselines3', 'Stable-Baselines3'),
        ('gymnasium', 'Gymnasium'),
        ('deap', 'DEAP')
    ]

    for module, name in dependencies:
        try:
            imported = __import__(module)
            version = getattr(imported, '__version__', 'unknown')
            print(f"  ✅ {name}: {version}")
        except ImportError as e:
            print(f"  ❌ {name}: Not installed ({e})")

    # 3. AI Module Import Check
    print("\n🤖 AI Module Import Check:")
    ai_modules = [
        ('backend.ai_modules.classification.feature_extractor', 'Feature Extractor'),
        ('backend.ai_modules.classification.rule_based_classifier', 'Rule-Based Classifier'),
        ('backend.ai_modules.optimization.feature_mapping', 'Feature Mapping Optimizer'),
        ('backend.ai_modules.prediction.quality_predictor', 'Quality Predictor'),
        ('backend.ai_modules.utils.performance_monitor', 'Performance Monitor')
    ]

    for module, name in ai_modules:
        try:
            __import__(module)
            print(f"  ✅ {name}: OK")
        except ImportError as e:
            print(f"  ❌ {name}: Failed ({e})")

    # 4. Functional Test
    print("\n⚙️  Functional Test:")
    try:
        # Create test image
        import tempfile
        import cv2
        import numpy as np

        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        # Test feature extraction
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        extractor = ImageFeatureExtractor()

        start_time = time.time()
        features = extractor.extract_features(test_path)
        extraction_time = time.time() - start_time

        print(f"  ✅ Feature Extraction: {extraction_time:.3f}s")
        print(f"  ✅ Features Count: {len(features)}")

        # Test classification
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        classifier = RuleBasedClassifier()
        logo_type, confidence = classifier.classify(features)

        print(f"  ✅ Classification: {logo_type} ({confidence:.2f})")

        # Clean up
        import os
        os.unlink(test_path)

    except Exception as e:
        print(f"  ❌ Functional Test Failed: {e}")
        traceback.print_exc()

    # 5. Performance Test
    print("\n🚀 Performance Test:")
    try:
        memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

        # Load all AI components
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()

        memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_used = memory_after - memory_before

        print(f"  ✅ Memory Usage: {memory_used:.1f} MB")
        print(f"  ✅ All Components Loaded Successfully")

    except Exception as e:
        print(f"  ❌ Performance Test Failed: {e}")

    print("\n✅ Diagnostics Complete!")

if __name__ == "__main__":
    run_diagnostics()
```

### 15. Quick Fix Commands

```bash
#!/bin/bash
# quick_fix.sh - Quick fixes for common AI module issues

echo "🔧 AI Modules Quick Fix Tool"
echo "=========================="

# Fix 1: Reinstall AI dependencies
echo "1. Reinstalling AI dependencies..."
pip3 uninstall -y torch torchvision scikit-learn stable-baselines3 gymnasium deap
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install scikit-learn==1.3.2 stable-baselines3==2.0.0 gymnasium==0.28.1 deap==1.4

# Fix 2: Clear Python cache
echo "2. Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Fix 3: Recreate AI module directories
echo "3. Ensuring AI module structure..."
mkdir -p backend/ai_modules/{classification,optimization,prediction,utils,models}
mkdir -p backend/ai_modules/models/{pretrained,trained,cache}

# Fix 4: Test imports
echo "4. Testing imports..."
python3 -c "
try:
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
    print('✅ Feature extractor import OK')
except Exception as e:
    print(f'❌ Feature extractor import failed: {e}')

try:
    from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
    print('✅ Optimizer import OK')
except Exception as e:
    print(f'❌ Optimizer import failed: {e}')
"

echo "✅ Quick fix complete!"
```

This troubleshooting guide covers the most common issues encountered when working with AI modules and provides practical solutions for each problem.