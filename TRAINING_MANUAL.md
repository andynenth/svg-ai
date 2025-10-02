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

## 4. Model Training Steps
1. **Logo Classifier**
   ```bash
   python train_classifier.py
   ```
   - Aim for ≥95% validation accuracy.
   - Monitor training vs validation accuracy to detect overfitting.

2. **Quality Predictor**
   ```bash
   python train_quality_predictor.py
   ```
   - Target validation loss ≈0.001 and low MAE between predicted and actual SSIM.

3. **Parameter Optimizer**
   ```bash
   python train_optimizer.py
   ```
   - Learns optimal VTracer parameters (e.g., `color_precision`, `corner_threshold`).
   - Evaluate via SSIM improvement on validation logos.

4. **Model Packaging**
   - Export trained models to `models/production/`:
     - `classifier.pth`
     - `optimizer.xgb`
     - Optional metadata (`metadata.json`).
   - Validate bundle:
     ```bash
     python scripts/validate_ai_models.py models/production/
     ```

## 5. Monitoring & Visualization
- **Progress tracking**:
  ```bash
  python train_with_progress.py 500
  ```
  Displays progress bars, ETA, and per-logo statistics during data generation.

- **Visual results**:
  ```bash
  python view_results.py
  ```
  Generates comparison grids, difference maps, and `results_report.html`.

- **Single-logo comparison**:
  ```bash
  python quick_compare.py
  open comparison.html
  ```
  Shows original vs SVG side-by-side.

- **General monitoring**:
  - `watch -n 1 nvidia-smi` (GPU usage if available).
  - `htop` (CPU/RAM).
  - Stream logs: `python train_with_progress.py 500 2>&1 | tee training.log`.

## 6. Metrics & Success Criteria
- **Classifier**: Validation accuracy ≥90%; small gap versus training accuracy.
- **Quality predictor**: Low validation loss (≈0.001).
- **Optimizer**: Achieve target SSIM thresholds:
  - Simple ≥0.95
  - Text ≥0.90
  - Gradient ≥0.97
  - Complex ≥0.85
- **Generalization**: validation metrics close to training metrics.
- **Endpoint regression tests** (fallback behavior):
  ```bash
  PYTEST_ADDOPTS='--no-cov --maxfail=1' venv39/bin/python -m pytest tests/test_ai_endpoints_fallbacks.py
  ```

## 7. Deployment Checklist
1. Confirm exported models reside in `models/production/`.
2. Ensure environment variables (`MODEL_DIR`, `CLASSIFIER_MODEL`, `OPTIMIZER_MODEL`) point to these assets.
3. Health check:
   ```bash
   curl http://localhost:8000/api/ai-health
   ```
   - Expect `models_loaded: true`; if false, follow guidance in the response.
4. Conversion endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/convert-ai -F "file=@logo.png"
   ```
   - Response should contain real `ssim`, `mse`, `psnr` and fallback metadata if AI is degraded.

## 8. FAQ
- **Do I need to label logos manually?** No. Scripts auto-classify based on extracted features.
- **How do I change sample size?** Pass desired count to `train_with_raw_logos.py` (e.g., `python train_with_raw_logos.py 1000`).
- **Where are reports saved?** `results_report.html`, `training_results_comparison.png`, and logs appear in the project root.
- **What if `/api/ai-health` reports missing models?** Copy exports into `models/production/` and rerun the command; the response includes exact guidance.

## 9. Reference Commands
```bash
# Generate data (default 500 logos)
python train_with_raw_logos.py

# Full dataset (≈2,000 logos)
python train_with_raw_logos.py 2000

# Train models
python train_classifier.py
python train_quality_predictor.py
python train_optimizer.py

# Validate bundle
python scripts/validate_ai_models.py models/production/

# View results
python view_results.py
python quick_compare.py

# Automated fallback tests
PYTEST_ADDOPTS='--no-cov --maxfail=1' venv39/bin/python -m pytest tests/test_ai_endpoints_fallbacks.py
```

This manual consolidates the end-to-end process for generating training data, training all AI components, monitoring progress, evaluating success, and deploying the trained models for SVG-AI.
