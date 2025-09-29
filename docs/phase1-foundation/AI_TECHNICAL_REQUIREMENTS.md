# AI Pipeline Technical Requirements & Dependencies

## System Specifications

### **Current Environment** ✅
- **OS**: macOS Darwin 24.6.0
- **Architecture**: Intel x86_64 (no GPU acceleration)
- **Python**: 3.9.22
- **Working Directory**: `/Users/nrw/python/svg-ai`
- **Memory**: Recommended 8GB+ for concurrent AI processing

### **Deployment Target**
- **Processing Mode**: CPU-only optimization
- **Concurrency**: 4 simultaneous conversions max
- **Performance**: 0.5-60s per conversion (tier-dependent)
- **Memory Peak**: <500MB under normal load

---

## Dependency Requirements

### **Core AI Dependencies** (New Installations Required)

```bash
# PyTorch CPU (25MB download, 200MB installed)
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Machine Learning (50MB download)
pip install scikit-learn==1.3.2

# Reinforcement Learning (10MB download)
pip install stable-baselines3==2.0.0
pip install gymnasium==0.29.1

# Genetic Algorithms (2MB download)
pip install deap==1.4.1

# Neural Network Utilities (5MB download)
pip install transformers==4.36.0 --no-deps  # For model utilities only
```

### **Existing Dependencies** ✅ (Already Installed)
```bash
# Core image processing
opencv-python==4.12.0.88      # Computer vision
numpy==2.0.2                   # Numerical computing
pillow==11.3.0                 # Image loading
scikit-image==0.24.0           # Image metrics (SSIM)
matplotlib==3.9.4              # Visualization
seaborn==0.13.2                # Statistical plots

# Web framework
fastapi==0.117.1               # API endpoints
uvicorn[standard]==0.37.0      # ASGI server
python-multipart==0.0.20       # File uploads

# Utilities
pandas==2.3.2                  # Data handling
joblib==1.5.2                  # Parallel processing
tqdm==4.67.1                   # Progress bars
rich==13.7.0                   # Console output
```

### **Model Storage Requirements**
```bash
# Model files to be downloaded/trained
backend/ai_modules/models/
├── pretrained/
│   ├── efficientnet_b0.pth         # 20MB (classification)
│   ├── resnet50_features.pth       # 100MB (quality prediction)
│   └── vtracer_ppo_agent.zip       # 5MB (RL optimization)
├── trained/
│   ├── logo_classifier.pth         # 25MB (fine-tuned)
│   ├── quality_predictor.pth       # 110MB (trained)
│   └── feature_scaler.pkl          # 1MB (normalization)
└── cache/
    ├── feature_cache.db            # 100MB (feature vectors)
    └── prediction_cache.db         # 50MB (quality predictions)

# Total storage: ~410MB
```

---

## Code Architecture Requirements

### **New Module Structure**
```bash
backend/ai_modules/
├── __init__.py
├── classification/
│   ├── __init__.py
│   ├── logo_classifier.py          # EfficientNet-B0 + rules
│   ├── feature_extractor.py        # OpenCV feature extraction
│   └── rule_based_classifier.py    # Fast mathematical rules
├── optimization/
│   ├── __init__.py
│   ├── feature_mapping.py          # Method 1: Correlation mapping
│   ├── rl_optimizer.py             # Method 2: PPO reinforcement learning
│   ├── adaptive_optimizer.py       # Method 3: Spatial complexity
│   └── vtracer_environment.py      # RL environment for VTracer
├── prediction/
│   ├── __init__.py
│   ├── quality_predictor.py        # ResNet-50 + MLP
│   └── model_utils.py              # Model loading/caching
├── training/
│   ├── __init__.py
│   ├── data_collector.py           # Training data pipeline
│   ├── train_classifier.py         # Classification training
│   ├── train_predictor.py          # Quality prediction training
│   └── train_rl_agent.py           # RL agent training
├── utils/
│   ├── __init__.py
│   ├── model_cache.py              # Model loading and caching
│   ├── feature_cache.py            # Feature vector caching
│   └── performance_monitor.py      # Speed/memory monitoring
└── models/                         # Model storage (see above)
```

### **Integration Points**

#### **1. BaseConverter Extension**
```python
# backend/converters/ai_enhanced_converter.py
class AIEnhancedSVGConverter(BaseConverter):
    """Drop-in replacement for existing converters"""

    def __init__(self):
        super().__init__("AI-Enhanced")
        # Initialize AI components

    def convert(self, image_path: str, **kwargs) -> str:
        """7-phase AI processing pipeline"""
        # Integrates with existing convert() interface
```

#### **2. API Enhancement**
```python
# backend/app.py - New routes
@app.route('/api/convert-ai', methods=['POST'])
def convert_ai():
    """AI-enhanced conversion with metadata"""

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Image analysis without conversion"""

@app.route('/api/ai-status', methods=['GET'])
def ai_status():
    """AI component health check"""
```

#### **3. Frontend Integration**
```javascript
// frontend/js/modules/aiConverter.js
class AIConverter {
    async convertWithAI(file, tier = 'auto') {
        // Call AI-enhanced API
        // Display AI insights
        // Handle tier selection
    }
}
```

---

## Performance Requirements

### **Processing Time Targets**
```python
# Tier-based performance requirements
PERFORMANCE_TARGETS = {
    'tier_1': {
        'max_time': 1.0,      # seconds
        'components': ['feature_extraction', 'rule_classification', 'feature_mapping'],
        'target_quality': 0.85  # SSIM
    },
    'tier_2': {
        'max_time': 15.0,     # seconds
        'components': ['tier_1', 'rl_optimization', 'quality_prediction'],
        'target_quality': 0.90  # SSIM
    },
    'tier_3': {
        'max_time': 60.0,     # seconds
        'components': ['tier_2', 'adaptive_regions', 'spatial_analysis'],
        'target_quality': 0.95  # SSIM
    }
}
```

### **Memory Usage Targets**
```python
# Memory constraints for CPU deployment
MEMORY_TARGETS = {
    'model_loading': 200,     # MB (all models loaded)
    'feature_extraction': 50, # MB per image
    'peak_processing': 100,   # MB per conversion
    'concurrent_limit': 4,    # simultaneous conversions
    'total_peak': 500        # MB maximum
}
```

### **Quality Improvement Targets**
```python
# Expected improvements over manual parameter selection
QUALITY_TARGETS = {
    'simple_logos': {
        'manual_ssim': 0.75,
        'ai_tier1_ssim': 0.87,    # +16% improvement
        'ai_tier2_ssim': 0.92,    # +23% improvement
        'ai_tier3_ssim': 0.96     # +28% improvement
    },
    'text_logos': {
        'manual_ssim': 0.80,
        'ai_tier1_ssim': 0.90,    # +13% improvement
        'ai_tier2_ssim': 0.95,    # +19% improvement
        'ai_tier3_ssim': 0.98     # +23% improvement
    },
    'gradient_logos': {
        'manual_ssim': 0.65,
        'ai_tier1_ssim': 0.78,    # +20% improvement
        'ai_tier2_ssim': 0.85,    # +31% improvement
        'ai_tier3_ssim': 0.92     # +42% improvement
    },
    'complex_logos': {
        'manual_ssim': 0.60,
        'ai_tier1_ssim': 0.72,    # +20% improvement
        'ai_tier2_ssim': 0.82,    # +37% improvement
        'ai_tier3_ssim': 0.88     # +47% improvement
    }
}
```

---

## Development Environment Setup

### **1. Dependency Installation Script**
```bash
#!/bin/bash
# setup_ai_dependencies.sh

echo "Installing AI dependencies for CPU deployment..."

# Install PyTorch CPU version
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install ML libraries
pip install scikit-learn==1.3.2
pip install stable-baselines3==2.0.0
pip install gymnasium==0.29.1
pip install deap==1.4.1

# Install utilities
pip install transformers==4.36.0 --no-deps

# Verify installations
python3 -c "
import torch
import torchvision
import sklearn
import stable_baselines3
import gymnasium
import deap

print('✅ All AI dependencies installed successfully!')
print(f'PyTorch: {torch.__version__} (CPU)')
print(f'Torchvision: {torchvision.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print(f'Stable-Baselines3: {stable_baselines3.__version__}')
print(f'Gymnasium: {gymnasium.__version__}')
"
```

### **2. Model Download Script**
```bash
#!/bin/bash
# download_models.sh

echo "Downloading pre-trained models..."

mkdir -p backend/ai_modules/models/pretrained

# Download EfficientNet-B0 (will be cached by torchvision)
python3 -c "
import torchvision.models as models
model = models.efficientnet_b0(pretrained=True)
print('✅ EfficientNet-B0 downloaded')
"

# Download ResNet-50 (will be cached by torchvision)
python3 -c "
import torchvision.models as models
model = models.resnet50(pretrained=True)
print('✅ ResNet-50 downloaded')
"

echo "✅ All models ready for training!"
```

### **3. Environment Verification Script**
```python
#!/usr/bin/env python3
# verify_ai_setup.py

import sys
import importlib
import torch
import cv2
import numpy as np
from pathlib import Path

def verify_dependencies():
    """Verify all AI dependencies are working"""

    required_modules = [
        'torch', 'torchvision', 'sklearn', 'stable_baselines3',
        'gymnasium', 'deap', 'cv2', 'numpy', 'PIL'
    ]

    print("🔍 Verifying AI dependencies...")

    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False

    return True

def verify_performance():
    """Test basic AI operations performance"""

    print("\n🚀 Testing performance...")

    # Test PyTorch CPU performance
    start = time.time()
    x = torch.randn(100, 100)
    y = torch.mm(x, x.t())
    torch_time = time.time() - start
    print(f"✅ PyTorch matrix multiplication: {torch_time:.3f}s")

    # Test OpenCV performance
    start = time.time()
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    edges = cv2.Canny(img, 50, 150)
    cv_time = time.time() - start
    print(f"✅ OpenCV edge detection: {cv_time:.3f}s")

    return torch_time < 0.1 and cv_time < 0.05

def verify_directories():
    """Ensure required directories exist"""

    print("\n📁 Verifying directory structure...")

    required_dirs = [
        'backend/ai_modules',
        'backend/ai_modules/models',
        'data/training',
        'data/validation'
    ]

    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")

    return True

if __name__ == "__main__":
    import time

    print("🤖 AI Pipeline Setup Verification")
    print("=" * 40)

    success = True
    success &= verify_dependencies()
    success &= verify_performance()
    success &= verify_directories()

    if success:
        print("\n🎉 AI environment setup complete!")
        print("Ready to begin AI pipeline development.")
    else:
        print("\n❌ Setup verification failed!")
        print("Please resolve issues before proceeding.")
        sys.exit(1)
```

---

## Security & Compliance

### **Dependency Security**
- All dependencies pinned to specific versions
- No network requests during inference (models cached locally)
- Input validation for all AI components
- Fallback mechanisms for AI failures

### **Data Privacy**
- No image data sent to external services
- All processing performed locally
- Optional feature: user data collection (opt-in only)
- GDPR compliance for training data

### **Resource Management**
- Memory limits enforced per conversion
- CPU usage monitoring and throttling
- Graceful degradation under load
- Health check endpoints for monitoring

---

## Testing Requirements

### **Unit Test Coverage**
- Feature extraction: >95% coverage
- Classification models: >90% coverage
- Parameter optimization: >90% coverage
- Quality prediction: >90% coverage
- API endpoints: >95% coverage

### **Integration Test Coverage**
- End-to-end pipeline: All code paths
- Error handling: All failure modes
- Performance: All tier targets met
- Memory usage: Under limits in all scenarios

### **Benchmark Requirements**
- Processing time measurement for each component
- Memory usage profiling under load
- Quality improvement validation
- Model accuracy validation

This technical specification ensures the AI pipeline can be successfully implemented within the existing infrastructure while meeting performance and quality targets.