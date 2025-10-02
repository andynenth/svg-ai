# SVG-AI Training Manual

## 1. Overview
- **Objective**: Train three cooperating models that improve PNG → SVG conversion by recommending optimal VTracer parameters.
- **Models**:
  1. **Logo Classifier** (CNN) – identifies logo type (simple_geometric, text_based, gradient, complex, abstract).
  2. **Quality Predictor** (Neural Network) – predicts SSIM scores before conversion.
  3. **Parameter Optimizer** (XGBoost/RandomForest) – selects optimal VTracer parameters per logo type.
- **Unified Training**: Use `train_ai.py` wrapper for complete automated training pipeline.

## 2. Prerequisites
- **Python Environment**: Python 3.9+ with required packages:
  ```bash
  torch, torchvision, scikit-learn, xgboost, joblib, numpy, Pillow, opencv-python,
  matplotlib, seaborn, pandas, cairosvg
  ```
- **Dataset**:
  - Production: `data/raw_logos/` (2,069 real company logos, various sizes)
  - Testing: `data/logos/` (50 synthetic logos for smoke tests)
- **Auto-Classification**: No manual labeling required; scripts automatically classify logos based on visual features.

## 3. Quick Start - Unified Training

### Simplest Way (Recommended)
```bash
# Quick test with 20 logos (~2 minutes)
python train_ai.py --quick

# Standard training with 100 logos (~15 minutes)
python train_ai.py

# Full training with all 2,069 logos (~2 hours)
python train_ai.py --full

# Custom sample size
python train_ai.py --samples 500
```

The `train_ai.py` script automatically:
1. Generates training data with progress monitoring
2. Trains all three models in sequence
3. Creates comprehensive visualizations
4. Generates HTML reports with metrics
5. Saves all models to appropriate locations

## 4. Manual Training Process

### Data Generation
```bash
# With progress monitoring (recommended)
python train_with_progress.py 100         # 100 logos with detailed progress

# With raw logos and auto-classification
python train_with_raw_logos.py            # 500 logos (default)
python train_with_raw_logos.py 2000       # 2000 logos

# Original synthetic dataset
python generate_training_data.py          # Uses data/logos/
```

**Process**:
- Auto-classifies logos using visual features (colors, edges, gradients, patterns)
- Tests 6 parameter combinations per logo
- Records SSIM, MSE, PSNR, file size, and conversion time

## 5. Individual Model Training

### Logo Classifier (CNN)
```bash
python train_classifier.py
```
- **Architecture**: 3-layer neural network with dropout
- **Input Features**: 7 features including color_precision, corner_threshold, quality_score
- **Output**: 5 logo type classes
- **Success Metrics**:
  - Validation accuracy ≥95% (often achieves 100%)
  - Small gap between training and validation accuracy
- **Output**: `models/production/logo_classifier.torchscript`

### Quality Predictor (Neural Network)
```bash
python train_quality_predictor.py
```
- **Architecture**: 4-layer deep neural network
- **Predicts**: SSIM score before conversion
- **Success Metrics**:
  - Validation loss ≤0.001
  - MAE <0.05 between predicted and actual SSIM
- **Output**: `models/production/quality_predictor.torchscript`

### Parameter Optimizer (Ensemble)
```bash
python train_optimizer.py
```
- **Models**: XGBoost + RandomForest ensemble
- **Learns**: Optimal VTracer parameters per logo type
- **Parameters Optimized**:
  - `color_precision` (3-10)
  - `corner_threshold` (30-90)
  - `segment_length` (2.5-6.0)
- **Output**: `models/production/parameter_optimizer.pkl`

### Model Export & Validation
```bash
# Export models to production format
python scripts/export_models.py

# Validate model bundle
python scripts/validate_ai_models.py models/production/

# Package for deployment
tar -czf models_production.tar.gz models/production/
```

## 6. Monitoring & Visualization

### Integrated Monitoring Systems

#### Training with Full Monitoring
```bash
python train_with_monitoring.py
```
- Real-time loss and accuracy tracking
- Memory and gradient monitoring
- Automatic visualization generation
- TensorBoard integration (optional)

#### Training with Validation Framework
```bash
python train_with_validation.py
```
- Comprehensive validation metrics
- Confusion matrices
- Precision, recall, F1 scores
- Early stopping detection

#### Comprehensive Visualization
```bash
python visualize_training.py
```
Creates:
- Training curves (loss, accuracy)
- Parameter correlation matrices
- Performance by logo type
- Feature importance plots
- File size analysis
- Confusion matrices

### Result Analysis

#### View Training Results
```bash
python view_results.py
```
- Side-by-side visual comparisons
- Best/worst conversion examples
- Interactive HTML report (`results_report.html`)
- Performance statistics by logo type

#### Quick Single Logo Test
```bash
python quick_compare.py [logo.png]
# Or random logo if no argument
open comparison.html
```

### System Monitoring
```bash
# GPU usage (if available)
watch -n 1 nvidia-smi

# CPU and memory
htop

# Training logs with output capture
python train_with_progress.py 500 2>&1 | tee training.log

# Monitor background processes
ps aux | grep python
```

## 7. Metrics & Success Criteria

### Model Performance Targets

#### Logo Classifier
- **Validation Accuracy**: ≥95% (often achieves 100%)
- **Training-Validation Gap**: <5% (indicates good generalization)
- **Per-Class Precision**: >0.90 for all logo types
- **Confusion Matrix**: Minimal off-diagonal elements

#### Quality Predictor
- **Validation Loss**: ≤0.001
- **MAE**: <0.05 between predicted and actual SSIM
- **R² Score**: >0.85 on validation set
- **Prediction Time**: <10ms per image

#### Parameter Optimizer
- **SSIM Achievement by Logo Type**:
  - Simple geometric: ≥0.95
  - Text-based: ≥0.90
  - Gradient: ≥0.85
  - Complex: ≥0.80
  - Abstract: ≥0.70
- **Parameter Selection Accuracy**: >80% selecting optimal params

### System Integration Tests
```bash
# Test AI endpoint fallbacks
PYTEST_ADDOPTS='--no-cov --maxfail=1' venv39/bin/python -m pytest tests/test_ai_endpoints_fallbacks.py

# Test production model loading
python -m pytest tests/test_production_model_manager.py

# Integration test suite
python -m pytest tests/test_integration.py -v
```

## 8. Deployment

### Pre-Deployment Checklist
- [ ] All models trained with acceptable metrics
- [ ] Models exported to production format (TorchScript/ONNX)
- [ ] Models placed in `models/production/` directory
- [ ] Environment variables configured
- [ ] Integration tests passing
- [ ] Fallback mechanisms tested

### Deployment Steps

1. **Model Directory Structure**:
   ```
   models/production/
   ├── logo_classifier.torchscript
   ├── quality_predictor.torchscript
   ├── parameter_optimizer.pkl
   └── metadata.json
   ```

2. **Environment Configuration**:
   ```bash
   export MODEL_DIR=models/production
   export CLASSIFIER_MODEL=logo_classifier.torchscript
   export QUALITY_MODEL=quality_predictor.torchscript
   export OPTIMIZER_MODEL=parameter_optimizer.pkl
   ```

3. **Start Backend Server**:
   ```bash
   python -m backend.app
   # Or with uvicorn for production
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4
   ```

4. **Health Check**:
   ```bash
   curl http://localhost:8000/api/ai-health
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "models_loaded": true,
     "models": {
       "classifier": "loaded",
       "quality_predictor": "loaded",
       "parameter_optimizer": "loaded"
     }
   }
   ```

5. **Test AI Conversion**:
   ```bash
   curl -X POST http://localhost:8000/api/convert-ai \
        -F "file=@test_logo.png" \
        -F "target_quality=0.9"
   ```

### Production Monitoring
- Monitor `/api/ai-health` endpoint regularly
- Track conversion success rates
- Monitor SSIM scores achieved
- Alert on fallback activations
- Log model prediction times

## 9. Frequently Asked Questions

**Q: Do I need to label logos manually?**
A: No. Scripts automatically classify logos based on visual features (colors, edges, gradients, patterns).

**Q: How long does training take?**
A: Quick test (20 logos): ~2 minutes. Standard (100 logos): ~15 minutes. Full (2,069 logos): ~2 hours.

**Q: Where are trained models saved?**
A: Models are saved to `models/production/` in TorchScript or pickle format.

**Q: What if training fails?**
A: Check `training.log` for errors. Common issues: missing dependencies, insufficient memory, corrupted images.

**Q: Can I resume interrupted training?**
A: Yes, training data is saved to JSON files. You can re-run model training without regenerating data.

**Q: How do I improve model performance?**
A: Use more training data (`--full` flag), adjust hyperparameters in training scripts, or increase epochs.

**Q: What if `/api/ai-health` shows models not loaded?**
A: Check that models exist in `models/production/` and environment variables are set correctly.

## 10. Quick Reference Commands

### Complete Training Pipeline
```bash
# Fastest complete training (~2 min)
python train_ai.py --quick

# Standard training (~15 min)
python train_ai.py

# Full dataset training (~2 hours)
python train_ai.py --full

# Monitor existing training
python train_ai.py --monitor
```

### Individual Components
```bash
# Data generation
python train_with_progress.py 100      # With progress bars
python train_with_raw_logos.py 500     # Auto-classification
python generate_training_data.py       # Synthetic data

# Model training
python train_classifier.py             # Logo type classifier
python train_quality_predictor.py      # SSIM predictor
python train_optimizer.py              # Parameter optimizer

# Monitoring & Visualization
python train_with_monitoring.py        # Training with monitoring
python train_with_validation.py        # With validation framework
python visualize_training.py          # Generate visualizations
python view_results.py                # View results & HTML report
python quick_compare.py               # Single logo test

# Testing & Validation
python scripts/validate_ai_models.py models/production/
python -m pytest tests/test_ai_endpoints_fallbacks.py
python -m pytest tests/test_integration.py -v

# Deployment
python -m backend.app                 # Start server
curl http://localhost:8000/api/ai-health  # Health check
```

## 11. Troubleshooting

### Common Issues and Solutions

**Import errors**: Ensure all dependencies installed:
```bash
pip install torch torchvision scikit-learn xgboost matplotlib seaborn pandas
```

**CUDA/GPU errors**: Training works on CPU, set:
```bash
export CUDA_VISIBLE_DEVICES=""
```

**Memory errors**: Reduce batch size in training scripts or use fewer samples.

**Poor SSIM scores**:
- Check if images are corrupted
- Ensure VTracer is properly installed
- Try different parameter ranges

**Model not loading**:
- Verify model files exist in `models/production/`
- Check file permissions
- Ensure TorchScript version compatibility

---

This manual provides comprehensive guidance for training, monitoring, and deploying the SVG-AI model pipeline. For additional support, consult the codebase documentation or run scripts with `--help` flag.
