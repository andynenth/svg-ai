# Day 2: Fix Model Loading & Architecture Mismatches

## Objective
Fix the EfficientNet model loading issues and ensure all trained models can be loaded and used for inference.

## Prerequisites
- [ ] Completed Day 1 data collection
- [ ] Access to trained models in `backend/ai_modules/models/trained/`
- [ ] PyTorch installed

## Tasks

### Task 1: Diagnose Model Loading Issues (1.5 hours)
**File**: `scripts/diagnose_model_loading.py`

- [x] List all model files in `backend/ai_modules/models/trained/`
- [x] Attempt to load each model and capture errors
- [x] Extract model architecture from checkpoint
- [x] Compare expected vs actual architecture:
  - [x] Check input dimensions
  - [x] Check number of classes
  - [x] Check layer names
  - [x] Check state dict keys
- [x] Generate diagnostic report

**Acceptance Criteria**:
- Identifies specific architecture mismatches
- Lists which models can/cannot be loaded
- Saves report to `model_diagnostic_report.json`

### Task 2: Create Model Architecture Adapter (2 hours)
**File**: `backend/ai_modules/utils/model_adapter.py`

- [x] Create flexible model loader class
- [x] Implement architecture detection
- [x] Add state dict key mapping:
  ```python
  def map_state_dict_keys(old_dict, key_mapping):
      # Map old keys to new architecture
  ```
- [x] Handle different model versions:
  - [x] 256-class to 4-class conversion
  - [x] Missing/extra layers
  - [x] Different layer names
- [x] Add fallback to random initialization for missing weights

**Acceptance Criteria**:
- Successfully loads at least one trained model
- Handles architecture mismatches gracefully
- Returns usable model for inference

### Task 3: Fix EfficientNet Classifier (2.5 hours)
**File**: `backend/ai_modules/classification/efficientnet_classifier_fixed.py`

- [x] Copy existing EfficientNet classifier
- [x] Fix class initialization:
  ```python
  def __init__(self, num_classes=4, pretrained=False):
      # Flexible architecture
  ```
- [x] Add model loading with adapter:
  - [x] Try loading checkpoint_best.pth
  - [x] Fall back to checkpoint_latest.pth
  - [x] Fall back to random weights
- [x] Implement proper preprocessing pipeline
- [x] Add inference method with error handling
- [x] Test on sample images

**Acceptance Criteria**:
- Loads without errors
- Classifies test image into one of 4 classes
- Returns confidence scores
- Works with existing router

### Task 4: Create Simple Fallback Classifier (1.5 hours)
**File**: `backend/ai_modules/classification/statistical_classifier.py`

- [x] Implement feature-based classifier using sklearn
- [x] Extract simple features:
  - [x] Color histogram
  - [x] Edge density
  - [x] Complexity score
- [x] Train on 800 existing classification images:
  ```python
  from sklearn.ensemble import RandomForestClassifier
  classifier = RandomForestClassifier(n_estimators=100)
  ```
- [x] Save trained model as pickle
- [x] Create prediction interface matching EfficientNet

**Acceptance Criteria**:
- Achieves >70% accuracy on test set
- Loads in <0.1 seconds
- Provides fallback when neural network fails

### Task 5: Integration Testing (30 minutes)
**File**: `tests/test_fixed_models.py`

- [x] Test fixed EfficientNet classifier
- [x] Test statistical fallback classifier
- [x] Test model adapter with various checkpoints
- [x] Test integration with router
- [x] Benchmark inference speed

**Acceptance Criteria**:
- All tests pass
- Models load successfully
- Classification works on test images
- Inference time <1 second per image

## Deliverables
1. **Diagnostic Report**: `model_diagnostic_report.json`
2. **Fixed Models**: Working classifiers in `backend/ai_modules/classification/`
3. **Model Adapter**: Utility for handling architecture mismatches
4. **Test Suite**: Comprehensive tests for fixed models

## Testing Commands
```bash
# Diagnose model issues
python scripts/diagnose_model_loading.py

# Test fixed EfficientNet
python -c "from backend.ai_modules.classification.efficientnet_classifier_fixed import EfficientNetClassifier; c = EfficientNetClassifier(); print(c.classify('data/raw_logos/62088.png'))"

# Test statistical classifier
python -c "from backend.ai_modules.classification.statistical_classifier import StatisticalClassifier; c = StatisticalClassifier(); print(c.classify('data/raw_logos/62088.png'))"

# Run integration tests
pytest tests/test_fixed_models.py -v
```

## Common Issues & Solutions

### Issue: "size mismatch for classifier.1.weight"
**Solution**: Model was trained with 256 classes, but code expects 4
- Use model adapter to map weights
- Or retrain final layer only

### Issue: "Unexpected key(s) in state_dict"
**Solution**: Model has extra layers not in current architecture
- Use strict=False when loading
- Filter out unexpected keys

### Issue: Model loads but predictions are random
**Solution**: Weights not properly initialized
- Check if weights actually loaded
- Verify preprocessing matches training

## Success Metrics
- [x] At least one model loads successfully
- [x] Classification accuracy >60% (better than random 25%)
- [x] Fallback classifier works when neural network fails
- [x] Integration with existing system maintained

## Notes
- Don't aim for perfect model loading - working is better than perfect
- Statistical classifier provides reliable fallback
- Document all architecture changes for future reference
- Keep both old and new classifier files during transition

## Next Day Preview
Day 3 will use the training data from Day 1 to build simple but effective statistical models for parameter optimization.