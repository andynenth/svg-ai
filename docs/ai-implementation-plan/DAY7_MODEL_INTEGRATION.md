# Day 7: Model Integration & Pipeline Unification

## Objective
Integrate all AI components (classification, optimization, quality prediction) into a unified pipeline that works seamlessly together.

## Prerequisites
- [ ] Working statistical models from Days 3-5
- [ ] Fixed classifiers from Day 2
- [ ] Learned correlations from Day 6
- [ ] Quality measurement system from Day 4

## Tasks

### Task 1: Create Unified AI Pipeline Manager (2.5 hours)
**File**: `backend/ai_modules/pipeline/unified_ai_pipeline.py`

- [x] Design pipeline architecture:
  ```python
  class UnifiedAIPipeline:
      def __init__(self):
          # Load all components
          self.classifier = self.load_classifier()
          self.optimizer = self.load_optimizer()
          self.quality_predictor = self.load_quality_predictor()
          self.router = self.load_router()

      def process(self, image_path: str, target_quality: float = 0.9):
          # 1. Extract features
          features = self.extract_features(image_path)

          # 2. Classify image type
          image_type, confidence = self.classifier.classify(image_path, features)

          # 3. Determine processing tier
          tier = self.router.select_tier(features, image_type, target_quality)

          # 4. Optimize parameters
          parameters = self.optimizer.optimize(features, image_type, tier)

          # 5. Predict quality
          predicted_quality = self.quality_predictor.predict(features, parameters)

          # 6. Convert
          result = self.convert(image_path, parameters)

          return result
  ```
- [x] Implement component loading with fallbacks
- [x] Add error handling for each stage
- [x] Create pipeline metadata tracking
- [x] Add performance timing

**Acceptance Criteria**:
- All components load successfully
- Pipeline executes end-to-end
- Graceful degradation on component failure
- Returns comprehensive result object

### Task 2: Component Interface Standardization (1.5 hours)
**File**: `backend/ai_modules/pipeline/component_interfaces.py`

- [x] Define standard interfaces:
  ```python
  from abc import ABC, abstractmethod

  class BaseClassifier(ABC):
      @abstractmethod
      def classify(self, image_path: str, features: Dict) -> Tuple[str, float]:
          pass

  class BaseOptimizer(ABC):
      @abstractmethod
      def optimize(self, features: Dict, image_type: str, tier: int) -> Dict:
          pass

  class BasePredictor(ABC):
      @abstractmethod
      def predict(self, features: Dict, parameters: Dict) -> float:
          pass
  ```
- [x] Create adapters for existing components
- [x] Implement interface validation
- [x] Add type hints and documentation

**Acceptance Criteria**:
- All components implement standard interfaces
- Type checking passes
- Components are interchangeable
- Clear documentation for each interface

### Task 3: Pipeline Configuration System (1.5 hours)
**File**: `backend/ai_modules/pipeline/pipeline_config.py`

- [x] Create configuration management:
  ```python
  pipeline_config = {
      "classifier": {
          "primary": "statistical_classifier",
          "fallback": "rule_based_classifier",
          "confidence_threshold": 0.7
      },
      "optimizer": {
          "method": "learned_optimizer",
          "fallback": "correlation_formulas",
          "cache_enabled": True
      },
      "quality_predictor": {
          "model": "xgboost_predictor",
          "threshold": 0.85
      },
      "tiers": {
          "tier1": {"max_time": 2.0, "methods": ["statistical"]},
          "tier2": {"max_time": 5.0, "methods": ["statistical", "learned"]},
          "tier3": {"max_time": 15.0, "methods": ["all"]}
      }
  }
  ```
- [x] Load configuration from JSON/YAML
- [x] Support environment-specific configs
- [x] Add configuration validation
- [x] Implement hot-reloading capability

**Acceptance Criteria**:
- Configuration loads from file
- Invalid configs are rejected
- Can switch components via config
- Changes don't require code modifications

### Task 4: Integration Testing Suite (2 hours)
**File**: `tests/test_unified_pipeline.py`

- [x] Create comprehensive tests:
  ```python
  def test_pipeline_end_to_end():
      pipeline = UnifiedAIPipeline()
      result = pipeline.process("test_image.png")

      assert result.success
      assert result.quality > 0.7
      assert result.processing_time < 5.0
      assert result.parameters is not None

  def test_component_fallback():
      # Test with failed primary component
      pipeline = UnifiedAIPipeline()
      pipeline.classifier = None  # Simulate failure
      result = pipeline.process("test_image.png")

      assert result.success  # Should use fallback
      assert "fallback" in result.metadata
  ```
- [x] Test all component combinations
- [x] Test failure scenarios
- [x] Test performance requirements
- [x] Test configuration changes

**Acceptance Criteria**:
- All tests pass
- >90% code coverage
- Tests run in <30 seconds
- Edge cases handled

### Task 5: Pipeline Monitoring & Debugging (30 minutes)
**File**: `backend/ai_modules/pipeline/pipeline_monitor.py`

- [x] Add monitoring capabilities:
  ```python
  class PipelineMonitor:
      def __init__(self):
          self.metrics = defaultdict(list)

      def record_stage(self, stage: str, duration: float, success: bool):
          self.metrics[stage].append({
              "duration": duration,
              "success": success,
              "timestamp": datetime.now()
          })

      def get_statistics(self):
          return {
              stage: {
                  "avg_duration": np.mean([m["duration"] for m in metrics]),
                  "success_rate": sum([m["success"] for m in metrics]) / len(metrics)
              }
              for stage, metrics in self.metrics.items()
          }
  ```
- [x] Track stage-by-stage execution
- [x] Log detailed debug information
- [x] Create performance profiling
- [x] Export metrics for analysis

**Acceptance Criteria**:
- Monitors all pipeline stages
- Identifies bottlenecks
- Exports metrics to JSON
- Minimal performance overhead

## Deliverables
1. **Unified Pipeline**: Complete AI processing pipeline
2. **Standard Interfaces**: Component interface definitions
3. **Configuration System**: Flexible pipeline configuration
4. **Test Suite**: Comprehensive integration tests
5. **Monitoring**: Pipeline performance monitoring

## Testing Commands
```bash
# Test unified pipeline
python -c "from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline; p = UnifiedAIPipeline(); print(p.process('test.png'))"

# Run integration tests
pytest tests/test_unified_pipeline.py -v

# Test with different configurations
python -c "from backend.ai_modules.pipeline.pipeline_config import PipelineConfig; c = PipelineConfig('config/production.json'); print(c.validate())"

# Monitor pipeline performance
python -m backend.ai_modules.pipeline.pipeline_monitor --analyze

# Benchmark pipeline
python scripts/benchmark_pipeline.py --images 100
```

## Pipeline Flow Diagram
```
Input Image
    ↓
Feature Extraction ←→ [Cache]
    ↓
Classification (Statistical/Rule-based)
    ↓
Tier Selection (1/2/3)
    ↓
Parameter Optimization (Learned/Formula)
    ↓
Quality Prediction
    ↓
VTracer Conversion
    ↓
Quality Measurement
    ↓
Result + Metadata
```

## Success Metrics
- [ ] Pipeline processes image in <5 seconds (Tier 2)
- [ ] All components integrate successfully
- [ ] Fallbacks work for all components
- [ ] Configuration changes without code edits

## Common Issues & Solutions

### Issue: Component version mismatch
**Solution**:
- Standardize interfaces
- Use adapters for legacy components
- Version lock in requirements.txt

### Issue: Pipeline too slow
**Solution**:
- Add caching between stages
- Parallel processing where possible
- Profile and optimize bottlenecks

### Issue: Difficult to debug failures
**Solution**:
- Add detailed logging at each stage
- Include stage metadata in results
- Create debug mode with verbose output

## Notes
- Keep pipeline modular for easy component swapping
- Cache aggressively but invalidate appropriately
- Monitor performance from day one
- Document data flow clearly

## Next Day Preview
Day 8 will enhance the routing system to make better tier selections based on image complexity and quality requirements.