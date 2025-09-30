# Day 3: Build Statistical Models for Parameter Optimization

## Objective
Train simple but effective statistical models using the parameter-quality data collected on Day 1 to replace hardcoded correlation formulas.

## Prerequisites
- [ ] Completed Day 1 data collection (1,000+ samples)
- [ ] Training data in `data/training/parameter_quality_data.json`
- [ ] Scikit-learn and XGBoost installed

## Tasks

### Task 1: Data Preprocessing Pipeline (1.5 hours)
**File**: `scripts/preprocess_training_data.py`

- [ ] Load parameter-quality data from Day 1
- [ ] Feature engineering:
  ```python
  features = {
      'edge_density': float,
      'unique_colors': float,
      'entropy': float,
      'complexity_score': float,
      'gradient_strength': float,
      'image_size': int,
      'aspect_ratio': float
  }
  ```
- [ ] Target variable preparation:
  - [ ] SSIM as primary target
  - [ ] Parameter values as multi-output targets
- [ ] Split data (70% train, 15% val, 15% test)
- [ ] Normalize features (StandardScaler)
- [ ] Handle missing values and outliers
- [ ] Save preprocessed data

**Acceptance Criteria**:
- Preprocessed data saved to `data/training/preprocessed/`
- Train/val/test splits created
- Feature scaling applied
- No missing values in output

### Task 2: Parameter Prediction Model (2.5 hours)
**File**: `backend/ai_modules/optimization/statistical_parameter_predictor.py`

- [ ] Implement XGBoost regressor for parameter prediction:
  ```python
  from xgboost import XGBRegressor
  from sklearn.multioutput import MultiOutputRegressor

  class StatisticalParameterPredictor:
      def __init__(self):
          self.model = MultiOutputRegressor(
              XGBRegressor(n_estimators=100, max_depth=6)
          )
  ```
- [ ] Train model on features → parameters mapping
- [ ] Implement parameter bounds enforcement
- [ ] Add confidence scoring based on prediction variance
- [ ] Save trained model
- [ ] Create prediction interface matching existing optimizer

**Acceptance Criteria**:
- Model trains successfully
- Predicts all 8 VTracer parameters
- Parameters within valid bounds
- Saves to `backend/ai_modules/models/xgb_parameter_predictor.pkl`

### Task 3: Quality Prediction Model (2 hours)
**File**: `backend/ai_modules/prediction/statistical_quality_predictor.py`

- [ ] Build gradient boosting model for SSIM prediction:
  ```python
  from sklearn.ensemble import GradientBoostingRegressor

  class StatisticalQualityPredictor:
      def __init__(self):
          self.model = GradientBoostingRegressor(
              n_estimators=200,
              learning_rate=0.1,
              max_depth=4
          )
  ```
- [ ] Features = image features + parameters
- [ ] Target = SSIM score
- [ ] Train with cross-validation
- [ ] Implement prediction with uncertainty estimates
- [ ] Add caching for repeated predictions

**Acceptance Criteria**:
- Predicts SSIM within ±0.1 of actual
- Model evaluation metrics documented
- Saves to `backend/ai_modules/models/gb_quality_predictor.pkl`

### Task 4: Model Evaluation & Comparison (1.5 hours)
**File**: `scripts/evaluate_statistical_models.py`

- [ ] Load test set from Task 1
- [ ] Evaluate parameter predictor:
  - [ ] Mean absolute error per parameter
  - [ ] R² score
  - [ ] Feature importance analysis
- [ ] Evaluate quality predictor:
  - [ ] MAE for SSIM prediction
  - [ ] Correlation with actual SSIM
- [ ] Compare with hardcoded formulas:
  - [ ] Load CorrelationFormulas
  - [ ] Run same test set
  - [ ] Calculate improvement metrics
- [ ] Generate comparison visualizations
- [ ] Create evaluation report

**Acceptance Criteria**:
- Shows >15% improvement over hardcoded formulas
- Generates at least 4 visualization plots
- Saves report to `model_evaluation_report.json`

### Task 5: Integration Wrapper (30 minutes)
**File**: `backend/ai_modules/optimization/learned_optimizer.py`

- [ ] Create wrapper class that uses statistical models:
  ```python
  class LearnedOptimizer:
      def __init__(self):
          self.param_predictor = load_model('xgb_parameter_predictor.pkl')
          self.quality_predictor = load_model('gb_quality_predictor.pkl')

      def optimize(self, features: Dict) -> Dict:
          # Predict parameters
          # Predict quality
          # Return optimization result
  ```
- [ ] Match interface of existing FeatureMappingOptimizer
- [ ] Add fallback to correlation formulas if models fail
- [ ] Include confidence scoring

**Acceptance Criteria**:
- Drop-in replacement for existing optimizer
- Handles model loading failures gracefully
- Returns same output format

## Deliverables
1. **Trained Models**:
   - `xgb_parameter_predictor.pkl`
   - `gb_quality_predictor.pkl`
2. **Evaluation Report**: `model_evaluation_report.json`
3. **Integration Code**: `learned_optimizer.py`
4. **Preprocessed Data**: Train/val/test splits

## Testing Commands
```bash
# Preprocess training data
python scripts/preprocess_training_data.py

# Train parameter predictor
python -m backend.ai_modules.optimization.statistical_parameter_predictor --train

# Train quality predictor
python -m backend.ai_modules.prediction.statistical_quality_predictor --train

# Evaluate models
python scripts/evaluate_statistical_models.py

# Test integration
python -c "from backend.ai_modules.optimization.learned_optimizer import LearnedOptimizer; opt = LearnedOptimizer(); print(opt.optimize({'edge_density': 0.5, 'unique_colors': 0.3}))"
```

## Model Training Tips

### XGBoost Hyperparameters
```python
params = {
    'n_estimators': 100,      # Start small, increase if underfitting
    'max_depth': 6,           # Prevent overfitting
    'learning_rate': 0.1,     # Default is usually good
    'subsample': 0.8,         # Prevent overfitting
    'colsample_bytree': 0.8   # Feature sampling
}
```

### Handling Small Dataset
- Use cross-validation (5-fold)
- Regularization to prevent overfitting
- Ensemble multiple models if needed

## Success Metrics
- [ ] Parameter prediction MAE < 10% of parameter range
- [ ] SSIM prediction MAE < 0.1
- [ ] >15% improvement over hardcoded formulas
- [ ] Model inference time < 50ms

## Common Issues & Solutions

### Issue: Overfitting on small dataset
**Solution**:
- Reduce model complexity (fewer trees, lower depth)
- Add regularization
- Use cross-validation

### Issue: Poor parameter predictions
**Solution**:
- Check feature importance
- Add more engineered features
- Try different model types (Random Forest vs XGBoost)

## Notes
- Simple models often outperform complex ones with limited data
- Focus on reliable predictions over perfect accuracy
- Document model limitations for future improvements
- Keep models small for fast inference

## Next Day Preview
Day 4 will implement robust quality measurement systems to validate our improvements and create a feedback loop for continuous learning.