# Day 6: Replace Hardcoded Correlation Formulas

## Objective
Replace the hardcoded mathematical correlation formulas with the learned statistical models from Days 3-5, ensuring backward compatibility and improved performance.

## Prerequisites
- [ ] Trained statistical models from Day 3
- [ ] Pattern analyzer from Day 5
- [ ] Existing correlation_formulas.py to replace

## Tasks

### Task 1: Backup and Analysis of Existing System (1 hour)
**File**: `scripts/analyze_correlation_formulas.py`

- [ ] Backup existing correlation files:
  ```bash
  cp backend/ai_modules/optimization/correlation_formulas.py \
     backend/ai_modules/optimization/correlation_formulas_backup.py
  ```
- [ ] Document current formula behavior:
  - [ ] Input/output ranges
  - [ ] Mathematical relationships
  - [ ] Edge cases handled
- [ ] Create test cases from existing behavior
- [ ] Benchmark current performance

**Acceptance Criteria**:
- Backup created successfully
- Documentation of all formulas
- Test suite with 20+ test cases
- Performance baseline established

### Task 2: Create Model-Based Correlation System (2.5 hours)
**File**: `backend/ai_modules/optimization/learned_correlations.py`

- [ ] Implement new correlation class:
  ```python
  class LearnedCorrelations:
      def __init__(self):
          self.param_model = self.load_model('xgb_parameter_predictor.pkl')
          self.patterns = self.load_patterns('success_patterns.json')
          self.fallback = CorrelationFormulasBackup()  # Original formulas

      def get_parameters(self, features: Dict) -> Dict:
          try:
              # Try learned model first
              params = self.param_model.predict(features)

              # Apply pattern-based adjustments
              image_type = self.classify_image_type(features)
              if image_type in self.patterns:
                  params = self.apply_pattern(params, self.patterns[image_type])

              return self.validate_parameters(params)
          except Exception as e:
              # Fallback to original formulas
              return self.fallback.get_parameters(features)
  ```
- [ ] Add parameter validation and bounds checking
- [ ] Implement smooth fallback mechanism
- [ ] Add logging for model vs formula usage
- [ ] Create compatibility layer for existing interface

**Acceptance Criteria**:
- Drop-in replacement for correlation_formulas.py
- Uses learned models when available
- Falls back gracefully on error
- Maintains same interface

### Task 3: Integration with Feature Mapping Optimizer (1.5 hours)
**File**: `backend/ai_modules/optimization/feature_mapping_optimizer_v2.py`

- [ ] Update FeatureMappingOptimizer to use learned correlations:
  ```python
  class FeatureMappingOptimizerV2:
      def __init__(self):
          self.correlations = LearnedCorrelations()  # New
          # self.formulas = CorrelationFormulas()    # Old
  ```
- [ ] Add confidence scoring:
  - [ ] High confidence: Use learned parameters directly
  - [ ] Medium confidence: Blend learned and formula
  - [ ] Low confidence: Use formulas
- [ ] Implement parameter caching
- [ ] Add performance monitoring

**Acceptance Criteria**:
- Optimizer uses learned correlations
- Confidence-based parameter selection
- Performance equal or better than original
- Backward compatible

### Task 4: A/B Testing Implementation (2 hours)
**File**: `scripts/ab_test_correlations.py`

- [ ] Create comprehensive A/B test:
  ```python
  def compare_correlation_methods():
      results = {
          'formula_based': [],
          'learned_model': [],
          'improvements': []
      }

      for image in test_images:
          # Test with formulas
          formula_result = test_with_formulas(image)

          # Test with learned model
          learned_result = test_with_learned(image)

          # Calculate improvement
          improvement = (learned_result.ssim - formula_result.ssim) / formula_result.ssim
  ```
- [ ] Test on diverse image set:
  - [ ] 20 simple geometric
  - [ ] 20 text-based
  - [ ] 20 gradients
  - [ ] 20 complex
- [ ] Measure:
  - [ ] Quality improvement (SSIM)
  - [ ] Processing time
  - [ ] Parameter stability
  - [ ] Failure rate
- [ ] Generate comparison report

**Acceptance Criteria**:
- Tests 80+ images
- Shows measurable improvement (>10%)
- Generates visual comparison report
- Statistical significance calculated

### Task 5: Rollout Strategy Implementation (1 hour)
**File**: `backend/ai_modules/optimization/correlation_rollout.py`

- [ ] Create gradual rollout system:
  ```python
  class CorrelationRollout:
      def __init__(self, rollout_percentage=0.1):
          self.rollout_percentage = rollout_percentage
          self.use_learned = self.should_use_learned()

      def should_use_learned(self):
          # Gradual rollout based on hash
          return random.random() < self.rollout_percentage

      def get_correlations(self):
          if self.use_learned:
              return LearnedCorrelations()
          return CorrelationFormulas()
  ```
- [ ] Add feature flags
- [ ] Implement monitoring
- [ ] Create rollback mechanism
- [ ] Add performance tracking

**Acceptance Criteria**:
- Gradual rollout works (10% → 50% → 100%)
- Can rollback instantly
- Tracks performance metrics
- No service disruption

## Deliverables
1. **Learned Correlations**: New correlation system using models
2. **Updated Optimizer**: Feature mapping with learned parameters
3. **A/B Test Results**: Comparison report showing improvements
4. **Rollout System**: Safe deployment mechanism
5. **Backup**: Original formulas preserved

## Testing Commands
```bash
# Analyze existing formulas
python scripts/analyze_correlation_formulas.py

# Test learned correlations
python -c "from backend.ai_modules.optimization.learned_correlations import LearnedCorrelations; lc = LearnedCorrelations(); print(lc.get_parameters({'edge_density': 0.5}))"

# Run A/B test
python scripts/ab_test_correlations.py --images 80 --output ab_test_results.html

# Test rollout
python -c "from backend.ai_modules.optimization.correlation_rollout import CorrelationRollout; r = CorrelationRollout(0.5); print(r.should_use_learned())"

# Integration test
python -m pytest tests/test_learned_correlations.py -v
```

## Migration Strategy

### Phase 1: Shadow Mode (Day 6)
- Run both systems in parallel
- Log differences
- No user impact

### Phase 2: Gradual Rollout (Day 7)
- 10% of requests use learned
- Monitor quality metrics
- Gather feedback

### Phase 3: Full Migration (Day 8)
- 100% using learned
- Formulas as fallback only
- Continuous monitoring

## Success Metrics
- [ ] >15% quality improvement over formulas
- [ ] No increase in failure rate
- [ ] <50ms added latency
- [ ] Successful rollback tested

## Common Issues & Solutions

### Issue: Model predictions worse than formulas
**Solution**:
- Check if model is loaded correctly
- Verify feature preprocessing matches training
- Use blended approach (weighted average)

### Issue: High latency with model inference
**Solution**:
- Cache predictions for common features
- Use smaller model (fewer trees)
- Implement model warmup on startup

## Notes
- Keep formulas as permanent fallback
- Monitor closely during rollout
- Document any degradations immediately
- Learned models should enhance, not replace entirely

## Next Day Preview
Day 7 will focus on integrating all the models into a cohesive system and ensuring they work together seamlessly.