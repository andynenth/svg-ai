# Day 13: Model Architecture - Quality Prediction Model

**Date**: Week 4, Day 3 (Wednesday)
**Duration**: 8 hours
**Team**: 2 developers
**Objective**: Implement ResNet-50 feature extractor and MLP predictor for SSIM quality prediction

---

## Prerequisites Checklist

Before starting, verify these are complete:
- [ ] Phase 2 Data Pipeline: Image preprocessing and feature extraction working
- [ ] PyTorch CPU environment setup and validated
- [ ] Training dataset available (1000+ images with SSIM ground truth)
- [ ] Model checkpoint storage directory structure ready
- [ ] Integration interfaces defined with optimization system

---

## Developer A Tasks (8 hours)

### Task A13.1: ResNet-50 Feature Extractor Implementation ⏱️ 4 hours

**Objective**: Build ResNet-50 based feature extraction pipeline optimized for CPU inference.

**Implementation**:
```python
# backend/ai_modules/quality_prediction/feature_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Tuple, Optional
import numpy as np
import logging

class ResNetFeatureExtractor(nn.Module):
    """ResNet-50 feature extractor for SVG quality prediction"""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        # Load ResNet-50 with pre-trained ImageNet weights
        self.backbone = models.resnet50(pretrained=pretrained)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Feature dimension: 2048 (ResNet-50 final layer)
        self.feature_dim = 2048

        # Freeze backbone if specified (for feature extraction)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # CPU optimization settings
        torch.set_num_threads(4)  # Optimize for Intel Mac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images"""
        features = self.backbone(x)
        # Flatten: (batch_size, 2048, 1, 1) -> (batch_size, 2048)
        return features.view(features.size(0), -1)
```

**Detailed Checklist**:
- [ ] Implement ResNet-50 backbone loading with pre-trained weights
  - Load ImageNet pre-trained ResNet-50 model
  - Remove final classification layer (avgpool + fc)
  - Verify feature output dimension is 2048
- [ ] Create CPU optimization configuration
  - Set torch.set_num_threads(4) for Intel Mac optimization
  - Configure CPU-only device placement
  - Disable GPU-specific optimizations
- [ ] Implement feature extraction forward pass
  - Process input tensor through ResNet backbone
  - Apply global average pooling correctly
  - Flatten output to (batch_size, 2048) dimensions
- [ ] Add backbone freezing capability
  - Implement parameter freezing for transfer learning
  - Allow unfreezing for fine-tuning scenarios
  - Track frozen/unfrozen parameter counts
- [ ] Create image preprocessing pipeline
  - Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - Resize to 224x224 for ResNet compatibility
  - Handle single channel and RGB conversion
- [ ] Implement batch processing support
  - Process multiple images efficiently
  - Handle variable batch sizes gracefully
  - Optimize memory usage for CPU processing
- [ ] Add feature caching mechanism
  - Cache extracted features to avoid recomputation
  - Implement LRU cache for memory management
  - Support persistent caching to disk
- [ ] Create comprehensive unit tests
  - Test with single image and batch inputs
  - Verify output dimensions and ranges
  - Test CPU optimization settings
- [ ] Benchmark feature extraction performance
  - Measure inference time on single images
  - Test batch processing efficiency
  - Validate memory usage patterns

**Deliverable**: Complete ResNet-50 feature extraction system

### Task A13.2: MLP Predictor Network Design ⏱️ 4 hours

**Objective**: Implement Multi-Layer Perceptron for SSIM prediction from ResNet features.

**Implementation**:
```python
# backend/ai_modules/quality_prediction/mlp_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np

class SSIMPredictor(nn.Module):
    """MLP network for SSIM prediction from ResNet features"""

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dims: List[int] = [1024, 512, 256, 128],
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build MLP layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU(inplace=True))

            # Dropout
            layers.append(nn.Dropout(dropout_rate))

            current_dim = hidden_dim

        # Final prediction layer (SSIM range: 0.0 to 1.0)
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())  # Constrain output to [0,1]

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict SSIM from ResNet features"""
        return self.mlp(features)
```

**Detailed Checklist**:
- [ ] Design progressive hidden layer architecture
  - Input layer: 2048 features from ResNet-50
  - Hidden layers: [1024, 512, 256, 128] progressive reduction
  - Output layer: Single SSIM prediction value
- [ ] Implement batch normalization layers
  - Add BatchNorm1d after each linear layer
  - Include option to disable for ablation studies
  - Initialize with proper weight/bias values
- [ ] Add dropout regularization
  - Configurable dropout rate (default: 0.3)
  - Apply dropout after each activation
  - Ensure training/evaluation mode switching
- [ ] Implement proper weight initialization
  - Use Kaiming/He initialization for ReLU networks
  - Initialize batch norm parameters correctly
  - Zero-initialize bias terms appropriately
- [ ] Add sigmoid output constraint
  - Ensure SSIM predictions are in [0,1] range
  - Handle gradient flow through sigmoid
  - Consider alternative output activations (tanh + scaling)
- [ ] Create model configuration system
  - Support different architectures via config
  - Allow easy experimentation with layer sizes
  - Save/load architecture configurations
- [ ] Implement prediction confidence estimation
  - Add optional uncertainty estimation layer
  - Implement Monte Carlo dropout for uncertainty
  - Provide confidence intervals for predictions
- [ ] Add model summary and parameter counting
  - Display model architecture summary
  - Count trainable vs frozen parameters
  - Estimate memory requirements
- [ ] Create unit tests for MLP architecture
  - Test forward pass with various input sizes
  - Verify output ranges and dimensions
  - Test training/evaluation mode switching
- [ ] Validate gradient flow and training stability
  - Check for vanishing/exploding gradients
  - Test with various initialization schemes
  - Verify batch normalization functionality

**Deliverable**: Complete MLP predictor network

---

## Developer B Tasks (8 hours)

### Task B13.1: Model Integration and Checkpoint System ⏱️ 4 hours

**Objective**: Create unified model class and checkpoint management system.

**Implementation**:
```python
# backend/ai_modules/quality_prediction/quality_model.py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json
import logging

class QualityPredictionModel(nn.Module):
    """Complete quality prediction model: ResNet-50 + MLP"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        # Feature extractor
        self.feature_extractor = ResNetFeatureExtractor(
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', True)
        )

        # MLP predictor
        self.predictor = SSIMPredictor(
            input_dim=config.get('feature_dim', 2048),
            hidden_dims=config.get('hidden_dims', [1024, 512, 256, 128]),
            dropout_rate=config.get('dropout_rate', 0.3),
            use_batch_norm=config.get('use_batch_norm', True)
        )

        # Model metadata
        self.model_version = config.get('version', '1.0.0')
        self.training_config = config.get('training', {})

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: extract features and predict SSIM"""
        features = self.feature_extractor(x)
        ssim_pred = self.predictor(features)
        return ssim_pred, features

    def predict_ssim(self, x: torch.Tensor) -> torch.Tensor:
        """Predict SSIM only (main inference method)"""
        ssim_pred, _ = self.forward(x)
        return ssim_pred

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ResNet features only"""
        return self.feature_extractor(x)
```

**Detailed Checklist**:
- [ ] Create unified model class combining ResNet + MLP
  - Integrate feature extractor and predictor
  - Support configurable model architectures
  - Handle device placement consistently
- [ ] Implement model configuration system
  - JSON-based configuration loading/saving
  - Support for hyperparameter management
  - Version tracking and compatibility checks
- [ ] Create checkpoint saving functionality
  - Save model state_dict with metadata
  - Include training configuration and metrics
  - Support incremental checkpoint saving
- [ ] Implement checkpoint loading system
  - Load pre-trained models with validation
  - Handle missing keys gracefully
  - Support partial loading for transfer learning
- [ ] Add model versioning and metadata
  - Track model version and training history
  - Include data pipeline compatibility info
  - Save training timestamp and environment details
- [ ] Create model validation utilities
  - Validate model architecture consistency
  - Check parameter counts and dimensions
  - Verify gradient flow for trainable parameters
- [ ] Implement model export functionality
  - Export to ONNX for cross-platform inference
  - Create TorchScript compilation support
  - Generate model summary reports
- [ ] Add CPU optimization configurations
  - Set optimal CPU threading settings
  - Configure memory allocation strategies
  - Optimize for Intel Mac performance
- [ ] Create comprehensive unit tests
  - Test model loading/saving functionality
  - Verify forward pass with various inputs
  - Test checkpoint compatibility
- [ ] Build model deployment utilities
  - Create production inference wrapper
  - Add batch processing optimizations
  - Implement model warming functions

**Deliverable**: Complete model integration and checkpoint system

### Task B13.2: Architecture Validation and Testing Framework ⏱️ 4 hours

**Objective**: Build comprehensive testing framework for model architecture validation.

**Implementation**:
```python
# backend/ai_modules/quality_prediction/model_validator.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import psutil
import gc
from pathlib import Path

class ModelArchitectureValidator:
    """Comprehensive model architecture validation and testing"""

    def __init__(self, model: QualityPredictionModel):
        self.model = model
        self.device = torch.device('cpu')  # CPU-only validation
        self.model.to(self.device)

    def validate_architecture(self) -> Dict[str, Any]:
        """Complete architecture validation suite"""
        results = {
            'dimension_tests': self._test_dimensions(),
            'gradient_tests': self._test_gradients(),
            'memory_tests': self._test_memory_usage(),
            'performance_tests': self._test_performance(),
            'stability_tests': self._test_numerical_stability()
        }
        return results

    def _test_dimensions(self) -> Dict[str, bool]:
        """Test input/output dimensions"""
        # Implementation here

    def _test_gradients(self) -> Dict[str, bool]:
        """Test gradient flow and backpropagation"""
        # Implementation here

    def _test_memory_usage(self) -> Dict[str, float]:
        """Test memory usage patterns"""
        # Implementation here
```

**Detailed Checklist**:
- [ ] Implement dimension validation tests
  - Test with various input sizes: (1,3,224,224), (8,3,224,224), etc.
  - Verify ResNet feature output: (batch_size, 2048)
  - Validate MLP output: (batch_size, 1) in range [0,1]
- [ ] Create gradient flow validation
  - Test backpropagation through entire model
  - Check for vanishing/exploding gradients
  - Verify frozen vs trainable parameter gradients
- [ ] Build memory usage profiling
  - Monitor memory consumption during forward/backward
  - Test with various batch sizes
  - Identify memory leaks in repeated operations
- [ ] Implement performance benchmarking
  - Measure inference time per image
  - Test batch processing efficiency
  - Compare single vs multi-threaded performance
- [ ] Add numerical stability tests
  - Test with edge case inputs (all zeros, all ones, etc.)
  - Verify output stability across multiple runs
  - Check for NaN/Inf handling
- [ ] Create model comparison utilities
  - Compare different architecture configurations
  - Benchmark parameter count vs performance
  - Generate architecture comparison reports
- [ ] Implement automated test suite
  - Create pytest-based testing framework
  - Add continuous integration test cases
  - Generate test coverage reports
- [ ] Build model profiling tools
  - Profile CPU usage and bottlenecks
  - Analyze layer-wise computation time
  - Generate optimization recommendations
- [ ] Add robustness testing
  - Test with corrupted/noisy inputs
  - Verify model behavior with out-of-distribution data
  - Test with various image preprocessing variations
- [ ] Create validation reporting system
  - Generate comprehensive validation reports
  - Include visualizations of test results
  - Export results in JSON and HTML formats

**Deliverable**: Complete architecture validation framework

---

## Integration Tasks (Both Developers - 1 hour)

### Task AB13.3: Model Architecture Integration Testing

**Objective**: Verify complete model architecture works end-to-end.

**Integration Test**:
```python
def test_day13_integration():
    """Test complete model architecture pipeline"""

    # Create model configuration
    config = {
        'pretrained': True,
        'freeze_backbone': True,
        'feature_dim': 2048,
        'hidden_dims': [1024, 512, 256, 128],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'version': '1.0.0'
    }

    # Initialize model
    model = QualityPredictionModel(config)
    model.eval()

    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        ssim_pred, features = model(dummy_input)

    # Validate outputs
    assert ssim_pred.shape == (batch_size, 1)
    assert features.shape == (batch_size, 2048)
    assert torch.all(ssim_pred >= 0.0) and torch.all(ssim_pred <= 1.0)

    print(f"✅ Model architecture integration successful")
    print(f"   SSIM predictions: {ssim_pred.squeeze().tolist()}")
    print(f"   Feature dimensions: {features.shape}")
```

**Checklist**:
- [ ] Test model initialization with configuration
- [ ] Verify forward pass with dummy data
- [ ] Validate output dimensions and ranges
- [ ] Test checkpoint saving and loading
- [ ] Run architecture validation suite
- [ ] Confirm CPU optimization settings
- [ ] Test integration with preprocessing pipeline

---

## End-of-Day Validation

### Functional Testing
- [ ] ResNet-50 feature extractor produces 2048-dimensional features
- [ ] MLP predictor outputs SSIM values in [0,1] range
- [ ] Complete model handles batch processing correctly
- [ ] Checkpoint system saves and loads models properly
- [ ] Architecture validation passes all tests

### Performance Testing
- [ ] Single image inference completes in <50ms
- [ ] Batch processing (8 images) completes in <200ms
- [ ] Memory usage remains stable over multiple inferences
- [ ] CPU utilization is optimized for Intel Mac

### Quality Verification
- [ ] Model architecture follows best practices
- [ ] Gradient flow works correctly through all layers
- [ ] Numerical stability maintained across input ranges
- [ ] Code follows established patterns and documentation standards

---

## Tomorrow's Preparation

**Day 14 Focus**: Training pipeline implementation and validation framework

**Prerequisites for Day 14**:
- [ ] Model architecture fully implemented and tested
- [ ] Checkpoint system operational
- [ ] Architecture validation framework working
- [ ] CPU optimization settings confirmed
- [ ] Integration with data pipeline verified

**Day 14 Preview**:
- Developer A: Training loop implementation and loss function design
- Developer B: Model evaluation metrics and training monitoring system

---

## Success Criteria

✅ **Day 13 Success Indicators**:
- ResNet-50 feature extractor working with pre-trained weights
- MLP predictor network producing valid SSIM predictions
- Complete model architecture integrated and tested
- Checkpoint system functional for model persistence
- Architecture validation framework operational

**Files Created**:
- `backend/ai_modules/quality_prediction/feature_extractor.py`
- `backend/ai_modules/quality_prediction/mlp_predictor.py`
- `backend/ai_modules/quality_prediction/quality_model.py`
- `backend/ai_modules/quality_prediction/model_validator.py`
- Unit tests for all components

**Key Metrics**:
- Feature extraction time: <20ms per image ✅
- SSIM prediction time: <10ms per prediction ✅
- Model parameter count: ~25M (ResNet) + ~2M (MLP) ✅
- Memory usage: <2GB for training, <1GB for inference ✅

**Interface Contracts Defined**:
- `QualityPredictionModel.predict_ssim(x)` → SSIM prediction
- `QualityPredictionModel.extract_features(x)` → ResNet features
- Model checkpoint format and metadata structure
- Configuration schema for model architecture