# Day 12: Model Architecture Implementation - Quality Prediction Model

**Date**: Week 4, Day 2
**Duration**: 8 hours
**Team**: 1 developer
**Objective**: Implement ResNet-50 + MLP architecture with PyTorch optimization for Intel Mac x86_64 CPU deployment

---

## Prerequisites Verification
- [ ] Day 11 data infrastructure completed and tested
- [ ] PyTorch CPU installation verified (torch>=1.13.0)
- [ ] torchvision with pre-trained models available
- [ ] Data pipeline from Day 11 producing training examples
- [ ] Feature extraction infrastructure operational
- [ ] Training data storage format defined

---

## Task 12.1: ResNet-50 Feature Extractor Implementation ⏱️ 4 hours
**Objective**: Build CPU-optimized ResNet-50 feature extraction system for SVG quality prediction

### Implementation Architecture
```python
# backend/ai_modules/quality_prediction/models/feature_extractor.py
class QualityFeatureExtractor(nn.Module):
    """ResNet-50 based feature extractor optimized for CPU inference"""

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # Load pre-trained ResNet-50 (ImageNet weights)
        self.backbone = models.resnet50(pretrained=True)

        # Remove final classification layers
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        # Feature dimension: 2048 from ResNet-50 global average pooling
        self.feature_dim = 2048

        # Freeze backbone for transfer learning
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # CPU optimization settings
        torch.set_num_threads(4)  # Optimal for Intel Mac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional features from input images"""
        # Input: (batch_size, 3, 224, 224)
        # Output: (batch_size, 2048)
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)
```

**Detailed Checklist**:
- [ ] **ResNet-50 Backbone Setup**
  - [ ] Load pre-trained ImageNet weights from torchvision.models
  - [ ] Remove final classification layer (avgpool + fc → avgpool only)
  - [ ] Verify output feature dimension is exactly 2048
  - [ ] Test with dummy input tensor: (1, 3, 224, 224) → (1, 2048)

- [ ] **CPU Optimization Configuration**
  - [ ] Set torch.set_num_threads(4) for Intel Mac optimization
  - [ ] Configure CPU-only device placement throughout
  - [ ] Disable CUDA/GPU related optimizations
  - [ ] Test performance with threading vs no threading

- [ ] **Transfer Learning Setup**
  - [ ] Implement backbone parameter freezing for feature extraction
  - [ ] Verify frozen parameters have requires_grad=False
  - [ ] Count trainable vs frozen parameters (should be 0 trainable when frozen)
  - [ ] Add option to unfreeze for fine-tuning scenarios

- [ ] **Image Preprocessing Pipeline**
  - [ ] Implement ImageNet preprocessing: Resize(256), CenterCrop(224)
  - [ ] Add normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  - [ ] Handle single-channel to RGB conversion for grayscale logos
  - [ ] Create preprocessing transforms for training vs inference

- [ ] **Feature Caching System**
  - [ ] Implement LRU cache for extracted features to avoid recomputation
  - [ ] Cache key based on image path + preprocessing parameters
  - [ ] Add disk-based persistent caching for large datasets
  - [ ] Memory management for cache size limits

- [ ] **Performance Optimization**
  - [ ] Batch processing support for multiple images
  - [ ] Optimize memory usage during feature extraction
  - [ ] Implement warming function for initial model loading
  - [ ] Add progress tracking for batch feature extraction

- [ ] **Unit Testing**
  - [ ] Test single image feature extraction: verify 2048 output
  - [ ] Test batch processing: (8, 3, 224, 224) → (8, 2048)
  - [ ] Test with various image sizes and formats
  - [ ] Verify preprocessing pipeline with real logo images

- [ ] **Performance Benchmarking**
  - [ ] Measure single image inference time (target: <2s on Intel Mac)
  - [ ] Test batch processing efficiency with various batch sizes
  - [ ] Monitor memory usage patterns and peak consumption
  - [ ] Compare frozen vs unfrozen backbone performance

**Expected Output**: Fully functional ResNet-50 feature extraction system ready for MLP integration

---

## Task 12.2: MLP Quality Predictor Implementation ⏱️ 3 hours
**Objective**: Implement Multi-Layer Perceptron for SSIM prediction from ResNet-50 features

### Architecture Design
```python
# backend/ai_modules/quality_prediction/models/mlp_predictor.py
class SSIMPredictor(nn.Module):
    """MLP network predicting SSIM quality from ResNet features + VTracer parameters"""

    def __init__(self,
                 image_feature_dim: int = 2048,
                 vtracer_param_dim: int = 8,
                 hidden_dims: List[int] = [1024, 512, 256],
                 dropout_rate: float = 0.3):
        super().__init__()

        # Input: 2048 image features + 8 VTracer parameters = 2056
        input_dim = image_feature_dim + vtracer_param_dim

        # Progressive MLP architecture
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim

        # Output layer: SSIM prediction [0,1]
        layers.extend([
            nn.Linear(current_dim, 1),
            nn.Sigmoid()  # Constrain to [0,1] range
        ])

        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, image_features: torch.Tensor, vtracer_params: torch.Tensor) -> torch.Tensor:
        """Predict SSIM from combined features"""
        # Concatenate features: (batch_size, 2048) + (batch_size, 8) → (batch_size, 2056)
        combined_features = torch.cat([image_features, vtracer_params], dim=1)
        ssim_prediction = self.mlp(combined_features)
        return ssim_prediction
```

**Detailed Checklist**:
- [ ] **MLP Architecture Implementation**
  - [ ] Input layer: 2056 dimensions (2048 image + 8 VTracer parameters)
  - [ ] Hidden layers: Progressive reduction [1024, 512, 256]
  - [ ] Output layer: Single SSIM value with sigmoid activation
  - [ ] Verify output range is strictly [0,1] with sigmoid

- [ ] **Regularization Components**
  - [ ] Add BatchNorm1d after each linear layer for training stability
  - [ ] Implement dropout with configurable rate (default: 0.3)
  - [ ] Test training vs evaluation mode switching
  - [ ] Verify dropout only active during training

- [ ] **Weight Initialization**
  - [ ] Use Xavier/He initialization for ReLU networks
  - [ ] Initialize BatchNorm parameters: weight=1, bias=0
  - [ ] Zero-initialize final layer bias for centered SSIM predictions
  - [ ] Test gradient flow through initialized network

- [ ] **VTracer Parameter Encoding**
  - [ ] Implement parameter normalization for 8 VTracer parameters:
    - color_precision: [1-20] → [0,1]
    - corner_threshold: [0-180] → [0,1]
    - length_threshold: [0-50] → [0,1]
    - max_iterations: [1-100] → [0,1]
    - splice_threshold: [0-180] → [0,1]
    - path_precision: [1-20] → [0,1]
    - layer_difference: [1-50] → [0,1]
    - hierarchical: [0,1] → [0,1]
  - [ ] Create parameter encoding utilities
  - [ ] Test parameter tensor creation from VTracer config dicts

- [ ] **Model Configuration System**
  - [ ] Support configurable hidden layer dimensions
  - [ ] Allow different dropout rates per layer
  - [ ] Enable/disable batch normalization option
  - [ ] Save architecture configuration with model checkpoints

- [ ] **Forward Pass Implementation**
  - [ ] Concatenate image features and VTracer parameters correctly
  - [ ] Handle variable batch sizes gracefully
  - [ ] Ensure output tensor has correct dimensions: (batch_size, 1)
  - [ ] Test with dummy inputs of various batch sizes

- [ ] **Unit Testing**
  - [ ] Test forward pass with known input dimensions
  - [ ] Verify output range [0,1] with various inputs
  - [ ] Test parameter encoding/normalization functions
  - [ ] Validate gradient computation through MLP

- [ ] **Performance Validation**
  - [ ] Measure inference time per prediction (target: <50ms)
  - [ ] Test memory usage with different batch sizes
  - [ ] Verify numerical stability with edge case inputs
  - [ ] Test with real ResNet features and VTracer parameters

**Expected Output**: Complete MLP predictor ready for training integration

---

## Task 12.3: Integrated Quality Prediction Model ⏱️ 1 hour
**Objective**: Create unified model class combining ResNet-50 + MLP with deployment interfaces

### Complete Model Integration
```python
# backend/ai_modules/quality_prediction/models/quality_predictor.py
class QualityPredictor(nn.Module):
    """Complete quality prediction model: ResNet-50 + MLP + VTracer parameter encoding"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        # Feature extractor component
        self.feature_extractor = QualityFeatureExtractor(
            freeze_backbone=config.get('freeze_backbone', True)
        )

        # SSIM predictor component
        self.ssim_predictor = SSIMPredictor(
            image_feature_dim=config.get('feature_dim', 2048),
            vtracer_param_dim=config.get('param_dim', 8),
            hidden_dims=config.get('hidden_dims', [1024, 512, 256]),
            dropout_rate=config.get('dropout_rate', 0.3)
        )

        # Parameter encoder for VTracer parameters
        self.param_encoder = VTracerParameterEncoder()

    def predict_quality(self, image_path: str, vtracer_params: Dict) -> float:
        """Main inference method: predict SSIM before conversion"""
        # Load and preprocess image
        image_tensor = self._preprocess_image(image_path)

        # Extract image features
        with torch.no_grad():
            image_features = self.feature_extractor(image_tensor.unsqueeze(0))

            # Encode VTracer parameters
            param_tensor = self.param_encoder.encode(vtracer_params).unsqueeze(0)

            # Predict SSIM
            ssim_pred = self.ssim_predictor(image_features, param_tensor)

        return ssim_pred.item()
```

**Detailed Checklist**:
- [ ] **Model Integration**
  - [ ] Combine feature extractor and MLP predictor seamlessly
  - [ ] Implement configuration-based model initialization
  - [ ] Handle device placement consistently (CPU-only)
  - [ ] Test end-to-end forward pass integration

- [ ] **Parameter Encoding System**
  - [ ] Create VTracerParameterEncoder class for 8 parameters
  - [ ] Implement normalization for each parameter type
  - [ ] Handle missing or invalid parameter values gracefully
  - [ ] Test parameter encoding with real VTracer configurations

- [ ] **Inference Interface**
  - [ ] Implement predict_quality() method for production use
  - [ ] Add image preprocessing pipeline integration
  - [ ] Handle file loading and tensor conversion
  - [ ] Return single float SSIM prediction

- [ ] **Model Persistence**
  - [ ] Implement save_model() with configuration metadata
  - [ ] Create load_model() class method for deployment
  - [ ] Save/load parameter encoder state
  - [ ] Include model versioning and training metadata

**Expected Output**: Production-ready quality prediction model

---

## End-of-Day Assessment

### Success Criteria
✅ **Day 12 Success Indicators**:
- ResNet-50 feature extractor produces 2048-dimensional features consistently
- MLP predictor accepts combined features and outputs SSIM predictions in [0,1] range
- VTracer parameter encoding system handles all 8 parameters correctly
- Integrated model supports end-to-end inference from image path to SSIM prediction
- All components optimized for Intel Mac x86_64 CPU deployment

### Performance Targets
- [ ] **Feature Extraction**: <2s per image on Intel Mac CPU
- [ ] **SSIM Prediction**: <50ms per inference
- [ ] **Memory Usage**: <1GB for inference, <4GB for training
- [ ] **Model Size**: <100MB for complete model + weights
- [ ] **Accuracy**: Architecture supports >90% correlation with actual SSIM

### Quality Gates
- [ ] All unit tests passing with >95% code coverage
- [ ] Performance benchmarks meet targets on target hardware
- [ ] Model architecture validated for gradient flow and numerical stability
- [ ] Integration tests successful with dummy data
- [ ] Code review completed following established patterns

**Files Created**:
- `/backend/ai_modules/quality_prediction/models/feature_extractor.py` - ResNet-50 feature extraction
- `/backend/ai_modules/quality_prediction/models/mlp_predictor.py` - MLP SSIM predictor
- `/backend/ai_modules/quality_prediction/models/quality_predictor.py` - Integrated model class
- `/backend/ai_modules/quality_prediction/models/parameter_encoder.py` - VTracer parameter encoding
- `/backend/ai_modules/quality_prediction/models/__init__.py` - Module exports
- `/config/quality_prediction_model_config.yaml` - Model architecture configuration
- Unit tests for all model components
- Performance benchmarking scripts

### Integration Interfaces for Agent 2 (Training)
**Model API Contracts**:
```python
# For training pipeline integration
model = QualityPredictor(config)
ssim_pred = model.predict_quality(image_path, vtracer_params)  # Single inference
features = model.feature_extractor(image_batch)  # Batch feature extraction
predictions = model.ssim_predictor(features, param_batch)  # Batch prediction
```

**Training Data Format**:
```python
training_example = {
    'image_path': str,           # Path to input image
    'vtracer_params': dict,      # 8 VTracer parameters
    'ssim_ground_truth': float,  # Target SSIM value [0,1]
    'metadata': dict             # Additional training metadata
}
```

### Integration Interfaces for Agent 3 (Deployment)
**Deployment API**:
```python
# For production deployment integration
predictor = QualityPredictor.load_model(checkpoint_path)
quality_score = predictor.predict_quality(logo_path, optimization_params)
confidence = predictor.prediction_confidence(logo_path, optimization_params)  # Optional
```

**Performance Contracts**:
- Inference latency: <50ms per prediction
- Memory footprint: <1GB resident memory
- CPU optimization: 4-thread configuration for Intel Mac
- Batch processing: Support 1-32 images per batch

---

## Risk Mitigation & Contingency Plans

**Identified Risks**:
1. **CPU Performance Bottleneck**:
   - Mitigation: Feature caching, batch optimization, model quantization
   - Contingency: Fall back to smaller ResNet variants (ResNet-18)

2. **Memory Constraints on Intel Mac**:
   - Mitigation: Streaming processing, garbage collection, batch size limits
   - Contingency: Progressive loading, feature precomputation

3. **VTracer Parameter Encoding Complexity**:
   - Mitigation: Comprehensive testing, parameter validation
   - Contingency: Simplified encoding with reduced parameter set

4. **Model Integration Complexity**:
   - Mitigation: Modular design, interface contracts, unit testing
   - Contingency: Simplified integration with manual parameter encoding

**Day 13 Preparation**:
- [ ] Model architecture components tested and validated
- [ ] Performance benchmarks established on target hardware
- [ ] Integration interfaces documented for Agent 2 coordination
- [ ] Training data format and API contracts finalized
- [ ] GPU/CPU deployment strategies determined

---

## Next Day Preview (Agent 2 Coordination)

**Agent 2 Focus (Day 13)**: Training Pipeline Implementation
- Training loop design and loss function optimization
- Validation framework and metric tracking
- Hyperparameter tuning and model selection
- Integration with Day 12 model architecture

**Coordination Points**:
- Model checkpoint format compatibility
- Training data pipeline integration
- Performance optimization strategies
- Validation metric consistency