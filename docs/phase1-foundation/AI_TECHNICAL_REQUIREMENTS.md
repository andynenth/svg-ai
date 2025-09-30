# AI Pipeline Technical Requirements & Dependencies - Colab-Hybrid Architecture

## System Specifications

### **Training Environment** (Google Colab)
- **Platform**: Google Colab Pro/Pro+ (Recommended)
- **GPU Types**: T4, V100, A100 (allocation-dependent)
- **GPU Memory**: 12-40GB VRAM (depending on allocated GPU)
- **System Memory**: 12-25GB RAM
- **Storage**: 100GB+ for datasets, models, and training artifacts
- **Python**: 3.10+ (Colab default)
- **Training Duration**: Sessions up to 24 hours (Pro+)

### **Local Inference Environment** ✅
- **OS**: macOS Darwin 24.6.0 (Primary), Linux/Windows (Compatible)
- **Architecture**: Intel x86_64, Apple Silicon M1/M2 (Universal support)
- **Python**: 3.9.22+ (Local installation)
- **Working Directory**: `/Users/nrw/python/svg-ai`
- **Memory**: 4GB+ for exported model inference
- **Acceleration**: CPU, Apple MPS, Intel MKL-DNN

### **Deployment Target** (Hybrid Architecture)
- **Training**: GPU-accelerated in Google Colab (10-100x faster)
- **Inference**: Lightweight exported models on local CPU/MPS
- **Concurrency**: 8+ simultaneous conversions (improved efficiency)
- **Performance**: 0.1-30s per conversion (GPU-trained optimization)
- **Memory Peak**: <300MB under normal load (exported models)

---

## Dependency Requirements

### **Google Colab Training Dependencies** (GPU Environment)

```python
# Colab notebook installation (first cell)
!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Enhanced ML libraries for GPU training
!pip install scikit-learn==1.3.2
!pip install stable-baselines3[extra]==2.0.0  # With extra dependencies
!pip install gymnasium[all]==0.29.1  # Full environment support
!pip install deap==1.4.1

# Model export and optimization
!pip install onnx==1.15.0
!pip install onnxruntime-gpu==1.16.0  # GPU inference during training
!pip install tensorboard==2.15.0  # Training visualization

# Advanced training utilities
!pip install transformers==4.36.0  # Full installation for training
!pip install accelerate==0.25.0  # Multi-GPU training support
!pip install datasets==2.16.0  # Dataset handling
!pip install wandb==0.16.0  # Experiment tracking (optional)
```

### **Local Inference Dependencies** (Lightweight)

```bash
# Minimal PyTorch for inference (cross-platform)
pip install torch==2.1.0 torchvision==0.16.0  # Auto-detects CPU/MPS

# Model runtime and export format support
pip install onnxruntime==1.16.0  # CPU inference optimization
pip install numpy==2.0.2  # Numerical computing
pip install pillow==11.3.0  # Image processing

# Optional: Apple Silicon optimization
pip install onnxruntime-coreml==1.16.0  # Apple Neural Engine support (macOS only)

# Minimal ML utilities
pip install scikit-learn==1.3.2  # Feature preprocessing only
pip install joblib==1.5.2  # Model serialization
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

#### **Google Colab Training Storage**
```bash
# Colab training workspace (/content/drive/MyDrive/svg-ai-training/)
colab_workspace/
├── datasets/
│   ├── training_logos/             # 2GB (50,000+ logo samples)
│   ├── validation_logos/           # 500MB (validation set)
│   └── test_benchmarks/            # 200MB (benchmark logos)
├── models/
│   ├── checkpoints/                # 1GB (training checkpoints)
│   ├── final_models/               # 500MB (best trained models)
│   └── pretrained_base/            # 300MB (base models)
├── training_logs/
│   ├── tensorboard/                # 200MB (training metrics)
│   ├── experiment_logs/            # 100MB (detailed logs)
│   └── export_artifacts/           # 50MB (export metadata)
└── exports/
    ├── production_ready/           # 150MB (optimized exports)
    └── deployment_packages/        # 100MB (deployment bundles)

# Colab Total: ~5GB
```

#### **Local Deployment Storage** (Exported Models Only)
```bash
# Local lightweight storage
backend/ai_modules/models/
├── exported/
│   ├── logo_classifier.onnx        # 15MB (optimized classification)
│   ├── logo_classifier.torchscript # 18MB (PyTorch JIT)
│   ├── quality_predictor.onnx      # 45MB (optimized prediction)
│   ├── quality_predictor.torchscript # 52MB (PyTorch JIT)
│   ├── parameter_optimizer.onnx    # 8MB (lightweight RL agent)
│   └── feature_preprocessor.pkl    # 2MB (preprocessing pipeline)
├── metadata/
│   ├── model_configs.json          # 10KB (model configurations)
│   ├── export_manifests.json       # 5KB (version tracking)
│   └── performance_benchmarks.json # 15KB (benchmark results)
└── cache/
    ├── inference_cache.db          # 50MB (prediction cache)
    └── feature_cache.db            # 25MB (feature vectors)

# Local Total: ~165MB (80% reduction from training models)
```

#### **Model Export Pipeline**
```python
# Export format specifications
EXPORT_FORMATS = {
    'torchscript': {
        'extension': '.torchscript',
        'optimization': 'JIT compilation',
        'compatibility': 'PyTorch native',
        'inference_speed': 'Fast',
        'file_size': 'Medium'
    },
    'onnx': {
        'extension': '.onnx',
        'optimization': 'Runtime optimization',
        'compatibility': 'Cross-platform',
        'inference_speed': 'Fastest',
        'file_size': 'Smallest'
    },
    'coreml': {
        'extension': '.mlmodel',
        'optimization': 'Apple Neural Engine',
        'compatibility': 'Apple devices only',
        'inference_speed': 'Fastest (Apple)',
        'file_size': 'Small'
    }
}
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

### **Integration Points** (Colab-Hybrid Architecture)

#### **1. Model Loading System**
```python
# backend/ai_modules/utils/model_loader.py
class HybridModelLoader:
    """Loads exported models for local inference"""

    def __init__(self, model_dir: str = "backend/ai_modules/models/exported"):
        self.model_dir = Path(model_dir)
        self.loaded_models = {}  # Model cache
        self.runtime_type = self._detect_runtime()  # CPU/MPS/CoreML

    def load_classifier(self, format_type: str = 'auto') -> Any:
        """Load exported logo classifier"""
        # Auto-select best format based on platform
        # Supports: ONNX, TorchScript, CoreML

    def load_quality_predictor(self, format_type: str = 'auto') -> Any:
        """Load exported quality prediction model"""
        # Optimized for fast inference

    def _detect_runtime(self) -> str:
        """Detect optimal inference runtime"""
        # Returns: 'apple_mps', 'intel_mkl', 'cpu_generic'
```

#### **2. AI-Enhanced Converter with Exported Models**
```python
# backend/converters/ai_enhanced_converter.py
class AIEnhancedSVGConverter(BaseConverter):
    """GPU-trained models with local inference"""

    def __init__(self):
        super().__init__("AI-Enhanced-Hybrid")
        self.model_loader = HybridModelLoader()
        self.classifier = self.model_loader.load_classifier()
        self.predictor = self.model_loader.load_quality_predictor()
        self.inference_cache = InferenceCache()

    def convert(self, image_path: str, **kwargs) -> str:
        """7-phase pipeline with exported models"""
        # Phase 1: Fast feature extraction (local)
        # Phase 2: Logo classification (exported ONNX model)
        # Phase 3: Quality prediction (exported TorchScript)
        # Phase 4-7: Optimization with cached predictions

    async def convert_async(self, image_path: str, **kwargs) -> str:
        """Async inference for batch processing"""
        # Concurrent model inference for multiple images
```

#### **3. API Enhancement for Hybrid Architecture**
```python
# backend/app.py - Updated routes
@app.route('/api/convert-ai-hybrid', methods=['POST'])
def convert_ai_hybrid():
    """AI conversion with exported models"""
    # Uses local inference with GPU-trained models

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Information about loaded exported models"""
    # Model versions, formats, performance metrics

@app.route('/api/colab-training-status', methods=['GET'])
def colab_training_status():
    """Check for updated models from Colab training"""
    # Integration with Google Drive model sync

@app.route('/api/update-models', methods=['POST'])
def update_models():
    """Download and deploy new models from Colab"""
    # Hot-swap models without server restart
```

#### **4. Colab Training Integration**
```python
# colab_notebooks/model_exporter.py
class ColabModelExporter:
    """Export trained models for local deployment"""

    def export_all_models(self, output_dir: str):
        """Export all trained models in multiple formats"""
        # TorchScript export for PyTorch compatibility
        # ONNX export for cross-platform optimization
        # CoreML export for Apple Silicon acceleration

    def create_deployment_package(self, version: str):
        """Create complete deployment package"""
        # Models + metadata + benchmarks + configs
        # Compressed package for easy download

    def sync_to_local(self, local_path: str):
        """Sync exported models to local development"""
        # Google Drive integration for model transfer
```

---

## Performance Requirements

### **Training Performance Targets** (Google Colab GPU)
```python
# GPU training performance specifications
COLAB_TRAINING_TARGETS = {
    'gpu_acceleration': {
        'speedup_vs_cpu': '10-100x',  # Depending on model complexity
        'training_time_classifier': '30-60 minutes',  # Logo classification
        'training_time_predictor': '60-120 minutes',  # Quality prediction
        'training_time_rl_agent': '120-300 minutes',  # RL optimization
    },
    'model_accuracy': {
        'logo_classification': 0.97,  # 97% accuracy on validation
        'quality_prediction_mae': 0.03,  # Mean Absolute Error < 3%
        'parameter_optimization': 0.92,  # 92% parameter sets achieve target quality
    },
    'gpu_memory_usage': {
        't4_max_batch_size': 32,      # T4 GPU (12GB VRAM)
        'v100_max_batch_size': 64,    # V100 GPU (16GB VRAM)
        'a100_max_batch_size': 128,   # A100 GPU (40GB VRAM)
    }
}
```

### **Inference Performance Targets** (Local Exported Models)
```python
# Local inference with exported models
INFERENCE_TARGETS = {
    'tier_1': {
        'max_time': 0.5,      # seconds (2x faster than CPU training)
        'components': ['onnx_classification', 'cached_feature_mapping'],
        'target_quality': 0.87,  # Improved from GPU training
        'model_format': 'ONNX (optimized)'
    },
    'tier_2': {
        'max_time': 8.0,      # seconds (50% faster than original)
        'components': ['tier_1', 'torchscript_prediction', 'rl_inference'],
        'target_quality': 0.93,  # Improved accuracy
        'model_format': 'TorchScript + ONNX'
    },
    'tier_3': {
        'max_time': 30.0,     # seconds (50% faster)
        'components': ['tier_2', 'ensemble_prediction', 'adaptive_optimization'],
        'target_quality': 0.97,  # Near-perfect results
        'model_format': 'Multi-model ensemble'
    }
}
```

### **Memory Usage Targets** (Lightweight Inference)
```python
# Optimized memory usage with exported models
MEMORY_TARGETS = {
    'exported_model_loading': 50,   # MB (compressed models)
    'feature_extraction': 30,       # MB per image (optimized)
    'inference_peak': 60,           # MB per conversion (reduced)
    'concurrent_limit': 8,          # simultaneous conversions (improved)
    'total_peak': 300,             # MB maximum (40% reduction)

    # Platform-specific optimizations
    'apple_mps_acceleration': True,  # Metal Performance Shaders
    'intel_mkl_optimization': True,  # Math Kernel Library
    'onnx_runtime_optimization': True,  # ONNX Runtime optimizations
}
```

### **Export Quality Retention**
```python
# Model export optimization targets
EXPORT_QUALITY_TARGETS = {
    'accuracy_retention': {
        'torchscript_export': 0.999,  # 99.9% accuracy retention
        'onnx_export': 0.995,         # 99.5% accuracy retention
        'coreml_export': 0.998,       # 99.8% accuracy retention (Apple)
    },
    'model_compression': {
        'onnx_size_reduction': 0.30,  # 30% smaller than PyTorch
        'torchscript_optimization': 0.15,  # 15% smaller with JIT
        'quantization_available': True,  # INT8 quantization support
    },
    'inference_speed': {
        'onnx_speedup': '1.5-3x',     # vs PyTorch
        'apple_neural_engine': '2-5x',  # CoreML on Apple Silicon
        'batch_inference': '10x+',     # Batch processing optimization
    }
}
```

### **Quality Improvement Targets** (GPU-Trained vs Manual)
```python
# Enhanced improvements with GPU-trained models
QUALITY_TARGETS = {
    'simple_logos': {
        'manual_ssim': 0.75,
        'hybrid_tier1_ssim': 0.89,    # +19% improvement (GPU-trained)
        'hybrid_tier2_ssim': 0.94,    # +25% improvement
        'hybrid_tier3_ssim': 0.98     # +31% improvement
    },
    'text_logos': {
        'manual_ssim': 0.80,
        'hybrid_tier1_ssim': 0.92,    # +15% improvement
        'hybrid_tier2_ssim': 0.97,    # +21% improvement
        'hybrid_tier3_ssim': 0.995    # +24% improvement (near-perfect)
    },
    'gradient_logos': {
        'manual_ssim': 0.65,
        'hybrid_tier1_ssim': 0.82,    # +26% improvement (major boost)
        'hybrid_tier2_ssim': 0.89,    # +37% improvement
        'hybrid_tier3_ssim': 0.95     # +46% improvement
    },
    'complex_logos': {
        'manual_ssim': 0.60,
        'hybrid_tier1_ssim': 0.76,    # +27% improvement
        'hybrid_tier2_ssim': 0.86,    # +43% improvement
        'hybrid_tier3_ssim': 0.92     # +53% improvement
    },
    'artistic_logos': {  # New category enabled by GPU training
        'manual_ssim': 0.50,  # Previously poor results
        'hybrid_tier1_ssim': 0.68,    # +36% improvement
        'hybrid_tier2_ssim': 0.78,    # +56% improvement
        'hybrid_tier3_ssim': 0.85     # +70% improvement
    }
}

# Training data scaling benefits
DATA_SCALING_BENEFITS = {
    'training_set_size': '50,000+ logos',  # 100x larger than original
    'validation_coverage': {
        'logo_types': 15,         # vs 5 original
        'style_variations': 200,  # vs 20 original
        'complexity_levels': 10,  # vs 3 original
    },
    'generalization': {
        'unseen_logo_accuracy': 0.94,  # 94% on completely new logos
        'cross_domain_transfer': 0.88,  # Works on non-logo images
        'parameter_robustness': 0.92,   # Consistent across parameter ranges
    }
}
```

---

## Development Environment Setup

### **1. Google Colab Setup Script**
```python
# colab_setup.ipynb - First cell
"""Google Colab Environment Setup for SVG-AI Training"""

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create project workspace
import os
project_dir = '/content/drive/MyDrive/svg-ai-training'
os.makedirs(project_dir, exist_ok=True)
os.chdir(project_dir)

# Clone repository or sync code
!git clone https://github.com/your-repo/svg-ai.git .
# OR upload code manually to Drive

# Install GPU-optimized dependencies
!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

!pip install onnx onnxruntime-gpu tensorboard accelerate datasets wandb
!pip install stable-baselines3[extra] gymnasium[all] scikit-learn

# Verify GPU access
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Download training datasets
!wget https://your-storage.com/svg-ai-training-data.zip
!unzip svg-ai-training-data.zip -d datasets/

print("🚀 Colab environment ready for training!")
```

### **2. Local Inference Setup Script**
```bash
#!/bin/bash
# setup_local_inference.sh

echo "Setting up local inference environment..."

# Detect platform and install appropriate PyTorch
if [[ "$(uname -m)" == "arm64" ]]; then
    echo "Detected Apple Silicon - installing MPS-optimized PyTorch"
    pip install torch==2.1.0 torchvision==0.16.0
else
    echo "Detected Intel/AMD - installing CPU-optimized PyTorch"
    pip install torch==2.1.0+cpu torchvision==0.16.0+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html
fi

# Install inference runtime dependencies
pip install onnxruntime==1.16.0
pip install numpy==2.0.2 pillow==11.3.0 scikit-learn==1.3.2
pip install joblib==1.5.2  # For model serialization

# Optional: Apple-specific optimizations
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing Apple-specific optimizations..."
    pip install onnxruntime-coreml==1.16.0
fi

# Create model directory structure
mkdir -p backend/ai_modules/models/{exported,metadata,cache}

# Verify installation
python3 -c "
import torch
import onnxruntime as ort
import numpy as np

print('✅ PyTorch:', torch.__version__)
print('✅ ONNX Runtime:', ort.__version__)

# Test MPS availability (Apple Silicon)
if torch.backends.mps.is_available():
    print('✅ Apple Metal Performance Shaders: Available')
    device = torch.device('mps')
    x = torch.randn(100, 100).to(device)
    print('✅ MPS tensor operations: Working')
else:
    print('ℹ️  Using CPU inference (Intel/AMD or older macOS)')

print('🎉 Local inference environment ready!')
"
```

### **3. Model Download and Sync Script**
```bash
#!/bin/bash
# sync_trained_models.sh

echo "Downloading trained models from Google Drive..."

# Set up Google Drive API or manual download
MODEL_DRIVE_ID="your-google-drive-folder-id"
LOCAL_MODEL_DIR="backend/ai_modules/models/exported"

# Method 1: Using rclone (recommended for automation)
if command -v rclone &> /dev/null; then
    echo "Using rclone for Google Drive sync..."
    rclone copy "gdrive:svg-ai-models/latest/" "$LOCAL_MODEL_DIR/"
else
    echo "rclone not found. Please download models manually from:"
    echo "https://drive.google.com/drive/folders/$MODEL_DRIVE_ID"
    echo "And place them in: $LOCAL_MODEL_DIR"
fi

# Method 2: Direct download (if public links available)
# wget https://drive.google.com/uc?id=model1_id -O logo_classifier.onnx
# wget https://drive.google.com/uc?id=model2_id -O quality_predictor.onnx

# Verify model integrity
python3 -c "
import onnxruntime as ort
import torch
from pathlib import Path

model_dir = Path('backend/ai_modules/models/exported')
success_count = 0

for model_file in model_dir.glob('*.onnx'):
    try:
        session = ort.InferenceSession(str(model_file))
        print(f'✅ {model_file.name}: Valid ONNX model')
        success_count += 1
    except Exception as e:
        print(f'❌ {model_file.name}: {e}')

for model_file in model_dir.glob('*.torchscript'):
    try:
        model = torch.jit.load(str(model_file))
        print(f'✅ {model_file.name}: Valid TorchScript model')
        success_count += 1
    except Exception as e:
        print(f'❌ {model_file.name}: {e}')

print(f'\n📊 Successfully loaded {success_count} models')
"

echo "✅ Model sync complete!"
```

### **3. Hybrid Environment Verification Script**
```python
#!/usr/bin/env python3
# verify_hybrid_setup.py

import sys
import importlib
import time
import torch
import numpy as np
from pathlib import Path

def verify_inference_dependencies():
    """Verify inference dependencies for local deployment"""

    required_modules = [
        'torch', 'torchvision', 'onnxruntime', 'numpy', 'PIL', 'sklearn', 'joblib'
    ]

    print("🔍 Verifying local inference dependencies...")

    for module in required_modules:
        try:
            mod = importlib.import_module(module)
            if module == 'torch':
                print(f"✅ {module} {mod.__version__} - Device: {get_torch_device()}")
            elif hasattr(mod, '__version__'):
                print(f"✅ {module} {mod.__version__}")
            else:
                print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False

    return True

def get_torch_device():
    """Detect optimal PyTorch device"""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple MPS (Metal Performance Shaders)"
    else:
        return "CPU"

def verify_model_loading():
    """Test exported model loading performance"""

    print("\n🚀 Testing model loading performance...")

    model_dir = Path('backend/ai_modules/models/exported')

    # Test ONNX Runtime
    try:
        import onnxruntime as ort

        # Test with a dummy ONNX model if available
        onnx_models = list(model_dir.glob('*.onnx'))
        if onnx_models:
            start = time.time()
            session = ort.InferenceSession(str(onnx_models[0]))
            load_time = time.time() - start
            print(f"✅ ONNX model loading: {load_time:.3f}s")

            # Test inference speed
            inputs = session.get_inputs()
            if inputs:
                dummy_input = np.random.randn(*inputs[0].shape).astype(np.float32)
                start = time.time()
                output = session.run(None, {inputs[0].name: dummy_input})
                inference_time = time.time() - start
                print(f"✅ ONNX inference: {inference_time:.3f}s")
        else:
            print("ℹ️  No ONNX models found - run model sync first")

    except ImportError:
        print("❌ ONNX Runtime not available")
        return False

    # Test TorchScript loading
    try:
        torchscript_models = list(model_dir.glob('*.torchscript'))
        if torchscript_models:
            start = time.time()
            model = torch.jit.load(str(torchscript_models[0]))
            load_time = time.time() - start
            print(f"✅ TorchScript model loading: {load_time:.3f}s")
        else:
            print("ℹ️  No TorchScript models found")
    except Exception as e:
        print(f"⚠️  TorchScript loading test failed: {e}")

    return True

def verify_platform_optimizations():
    """Test platform-specific optimizations"""

    print("\n🔧 Testing platform optimizations...")

    device = torch.device('cpu')

    # Test Apple MPS if available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Apple Metal Performance Shaders: Available")
        try:
            device = torch.device('mps')
            x = torch.randn(100, 100).to(device)
            start = time.time()
            y = torch.mm(x, x.t())
            mps_time = time.time() - start
            print(f"✅ MPS tensor operations: {mps_time:.3f}s")
        except Exception as e:
            print(f"⚠️  MPS test failed: {e}")
            device = torch.device('cpu')

    # Test CPU performance
    x = torch.randn(100, 100).to('cpu')
    start = time.time()
    y = torch.mm(x, x.t())
    cpu_time = time.time() - start
    print(f"✅ CPU tensor operations: {cpu_time:.3f}s")

    # Test Intel MKL if available
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mkl'):
        if torch.backends.mkl.is_available():
            print("✅ Intel MKL: Available")
        else:
            print("ℹ️  Intel MKL: Not available")

    return True

def verify_hybrid_architecture():
    """Verify complete hybrid setup"""

    print("\n🏗️  Verifying hybrid architecture...")

    # Check directory structure
    required_dirs = [
        'backend/ai_modules/models/exported',
        'backend/ai_modules/models/metadata',
        'backend/ai_modules/models/cache',
        'backend/ai_modules/utils',
        'colab_notebooks',  # For Colab integration
    ]

    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")

    # Check for model sync capability
    try:
        import subprocess
        rclone_check = subprocess.run(['which', 'rclone'], capture_output=True)
        if rclone_check.returncode == 0:
            print("✅ rclone available for Google Drive sync")
        else:
            print("ℹ️  rclone not available - manual model sync required")
    except:
        print("ℹ️  Model sync tools check skipped")

    return True

def generate_setup_report():
    """Generate complete setup report"""

    device = get_torch_device()
    model_dir = Path('backend/ai_modules/models/exported')
    model_count = len(list(model_dir.glob('*.*')))

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    HYBRID SETUP REPORT                      ║
╠══════════════════════════════════════════════════════════════╣
║ Local Inference Device: {device:<40} ║
║ Exported Models Found:  {model_count:<40} ║
║ PyTorch Version:        {torch.__version__:<40} ║
║ Setup Status:           {'✅ READY FOR INFERENCE':<40} ║
╚══════════════════════════════════════════════════════════════╝

📋 NEXT STEPS:
1. Train models in Google Colab using colab_training_notebook.ipynb
2. Export and sync trained models using sync_trained_models.sh
3. Test inference with: python test_hybrid_inference.py
4. Deploy with: python web_server.py

💡 COLAB TRAINING:
   - Use T4/V100/A100 GPUs for 10-100x faster training
   - Export models in ONNX/TorchScript formats
   - Sync to local deployment via Google Drive
"""

    print(report)

if __name__ == "__main__":
    print("🤖 Colab-Hybrid AI Setup Verification")
    print("=" * 60)

    success = True
    success &= verify_inference_dependencies()
    success &= verify_model_loading()
    success &= verify_platform_optimizations()
    success &= verify_hybrid_architecture()

    if success:
        generate_setup_report()
    else:
        print("\n❌ Hybrid setup verification failed!")
        print("Please resolve issues before proceeding.")
        sys.exit(1)
```

---

## Security & Compliance

### **Training Security** (Google Colab)
- **Data Isolation**: Training data stored in private Google Drive
- **Model Security**: Trained models remain in user's Google account
- **Code Security**: Training notebooks use pinned dependency versions
- **Access Control**: Colab notebooks require Google authentication
- **Data Retention**: Training artifacts automatically managed by user

### **Inference Security** (Local Deployment)
- **Model Integrity**: Exported models include cryptographic checksums
- **Dependency Security**: All inference dependencies pinned to specific versions
- **Network Isolation**: No network requests during inference (models cached locally)
- **Input Validation**: Comprehensive validation for all AI components
- **Fallback Mechanisms**: Graceful degradation when AI models unavailable

### **Data Privacy** (Enhanced)
- **Zero External Transmission**: No image data sent to external services during inference
- **Local Processing**: All inference performed on user's hardware
- **Training Data Control**: User maintains full control of training datasets in Colab
- **Model Ownership**: All trained models remain user's intellectual property
- **GDPR Compliance**: Training data handling complies with privacy regulations
- **Optional Telemetry**: Performance metrics collection (opt-in only)

### **Hybrid Architecture Security**
- **Model Transfer Security**: Encrypted transfer between Colab and local deployment
- **Version Control**: Model versioning and rollback capabilities
- **Integrity Verification**: Automatic validation of exported model integrity
- **Sandbox Isolation**: Inference models run in isolated Python environments

### **Resource Management**
- **Memory Limits**: Enforced limits per conversion (300MB max)
- **CPU/GPU Usage**: Monitoring and throttling for both training and inference
- **Storage Management**: Automatic cleanup of temporary training artifacts
- **Load Balancing**: Graceful degradation under high inference load
- **Health Monitoring**: Comprehensive health check endpoints for both environments

---

## Testing Requirements

### **Training Validation** (Google Colab)
- **Model Training Tests**: >95% coverage of training pipeline
- **Data Pipeline Tests**: Validation of dataset loading and preprocessing
- **GPU Utilization Tests**: Verify optimal GPU memory and compute usage
- **Export Quality Tests**: Validate model export accuracy retention (>99%)
- **Cross-Validation**: K-fold validation on training datasets
- **Ablation Studies**: Component contribution analysis

### **Inference Testing** (Local Deployment)
- **Model Loading Tests**: >98% coverage of model loading scenarios
- **Format Compatibility**: Test ONNX, TorchScript, CoreML formats
- **Platform Testing**: Verify performance on Intel x86_64 and Apple Silicon
- **Memory Usage Tests**: Validate <300MB peak memory usage
- **Performance Tests**: All tier targets met with exported models
- **Fallback Tests**: Graceful degradation when models unavailable

### **Hybrid Integration Testing**
- **Model Transfer Tests**: Validate Colab → Local model transfer pipeline
- **Version Compatibility**: Test model version updates and rollbacks
- **End-to-End Pipeline**: Complete training → export → inference workflow
- **Error Handling**: All failure modes in hybrid architecture
- **Performance Regression**: Validate inference performance vs training

### **Quality Assurance**
- **Accuracy Benchmarks**:
  - Logo classification: >97% accuracy
  - Quality prediction: <3% Mean Absolute Error
  - Parameter optimization: >92% success rate
- **Performance Benchmarks**:
  - Training time: <4 hours total (all models)
  - Inference latency: <0.5s for Tier 1
  - Memory efficiency: <300MB peak usage
- **Robustness Testing**:
  - Edge case handling (corrupted images, unusual formats)
  - Stress testing (concurrent inference requests)
  - Resource exhaustion scenarios

### **Continuous Integration**
- **Automated Training**: Scheduled Colab notebook execution
- **Model Export Pipeline**: Automated export and deployment testing
- **Performance Monitoring**: Continuous benchmarking of inference speed
- **Quality Regression Tests**: Automated validation of model improvements

This technical specification ensures the Colab-Hybrid AI pipeline can be successfully implemented with GPU-accelerated training and efficient local inference deployment, meeting enhanced performance and quality targets through the power of cloud GPU training.