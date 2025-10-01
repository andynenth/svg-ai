# Development Plan - Day 2: Test Coverage & API Endpoint Fixes

**Date**: Production Readiness Sprint - Day 2
**Objective**: Achieve 80% test coverage and fix remaining API endpoints
**Duration**: 8 hours
**Priority**: HIGH

## 🎯 Day 2 Success Criteria
- [ ] Test coverage increased from 3.2% to >80%
- [ ] API endpoint tests: 10/10 passing (currently 6/10)
- [ ] Complete test suite reorganization functional
- [ ] Coverage reporting infrastructure operational

---

## 📊 Day 2 Starting Point

### Prerequisites (From Day 1)
- [x] Import time <2s (fixed lazy loading)
- [x] Quality system API compatibility restored
- [x] Core integration tests passing

### Current Status
- **Test Coverage**: 3.2% (Need: 76.8% increase)
- **API Tests**: 6/10 passing (Need: 4 fixes)
- **Failed Endpoints**: convert, optimize, batch, classification-status

---

## 🚀 Task Breakdown

### Task 1: Fix API Endpoint Failures (3 hours) - HIGH PRIORITY
**Problem**: 4 API endpoints returning 400/405/500 errors preventing full API test suite passage

#### Subtask 1.1: Fix /api/convert Endpoint (1 hour)
**Files**: `backend/app.py`, `tests/test_api.py`
**Dependencies**: Day 1 completion
**Estimated Time**: 1 hour

**Current Issue**: 400 Bad Request on convert endpoint

**Investigation Steps**:
- [ ] **Step 1.1.1** (15 min): Analyze request payload format in test
- [ ] **Step 1.1.2** (30 min): Debug Flask request handling for JSON payload
- [ ] **Step 1.1.3** (15 min): Fix payload validation or processing

**Implementation Example**:
```python
# In backend/app.py
@app.route('/api/convert', methods=['POST'])
def convert_endpoint():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        # Process base64 image data
        image_data = data['image']
        # ... conversion logic

        return jsonify({
            'svg': svg_result,
            'quality': quality_metrics,
            'parameters': used_params
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

**Validation**:
```bash
python -m pytest tests/test_api.py::TestAPIEndpoints::test_convert_endpoint -v
```

#### Subtask 1.2: Fix /api/optimize Endpoint (1 hour)
**Files**: `backend/app.py`
**Dependencies**: None
**Estimated Time**: 1 hour

**Current Issue**: 405 Method Not Allowed

**Implementation Steps**:
- [ ] **Step 1.2.1** (30 min): Check if endpoint exists and accepts POST
- [ ] **Step 1.2.2** (30 min): Implement missing endpoint or fix method handling

**Expected Implementation**:
```python
@app.route('/api/optimize', methods=['POST'])
def optimize_parameters():
    data = request.get_json()
    image = data.get('image')
    target_quality = data.get('target_quality', 0.95)

    # Optimization logic
    optimized_params = optimizer.optimize_for_target(image, target_quality)

    return jsonify({'parameters': optimized_params})
```

#### Subtask 1.3: Fix /api/batch-convert Endpoint (1 hour)
**Files**: `backend/app.py`
**Dependencies**: None
**Estimated Time**: 1 hour

**Current Issue**: 405 Method Not Allowed

**Implementation Steps**:
- [ ] **Step 1.3.1** (45 min): Implement batch processing endpoint
- [ ] **Step 1.3.2** (15 min): Handle multiple image payloads

**Expected Implementation**:
```python
@app.route('/api/batch-convert', methods=['POST'])
def batch_convert():
    data = request.get_json()
    images = data.get('images', [])

    results = []
    for img_data in images:
        # Process each image
        result = process_single_image(img_data)
        results.append(result)

    return jsonify({'results': results})
```

---

### Task 2: Implement Comprehensive Test Coverage (4 hours) - CRITICAL
**Problem**: 3.2% coverage far below 80% target due to untested code paths

#### Subtask 2.1: Analyze Coverage Gaps (1 hour)
**Files**: Coverage reports, existing test files
**Dependencies**: None
**Estimated Time**: 1 hour

**Implementation Steps**:
- [ ] **Step 2.1.1** (30 min): Generate detailed coverage report
  ```bash
  python -m pytest --cov=backend --cov-report=html --cov-report=json tests/
  ```
- [ ] **Step 2.1.2** (30 min): Identify critical untested modules:
  - `backend/ai_modules/` (currently ~20% coverage)
  - `backend/converters/` (currently ~15% coverage)
  - `backend/utils/` (currently ~18% coverage)

#### Subtask 2.2: Create High-Impact Test Files (2 hours)
**Files**: New test files for untested modules
**Dependencies**: Subtask 2.1
**Estimated Time**: 2 hours

**Priority Testing Areas** (based on coverage analysis):
- [ ] **Step 2.2.1** (45 min): Create `tests/test_ai_modules.py`
  ```python
  class TestClassificationModule:
      def test_classify_simple_image(self):
          # Test basic classification

      def test_feature_extraction(self):
          # Test feature extraction

  class TestOptimizationEngine:
      def test_parameter_calculation(self):
          # Test parameter optimization
  ```

- [ ] **Step 2.2.2** (45 min): Create `tests/test_converters.py`
  ```python
  class TestAIEnhancedConverter:
      def test_convert_with_metrics(self):
          # Test conversion with metrics

      def test_parameter_optimization(self):
          # Test AI parameter optimization
  ```

- [ ] **Step 2.2.3** (30 min): Enhance `tests/test_utils.py` with missing utility tests

#### Subtask 2.3: Implement Mock-Based Testing (1 hour)
**Files**: Enhanced test files with mocks
**Dependencies**: Subtask 2.2
**Estimated Time**: 1 hour

**Implementation Strategy**:
- [ ] **Step 2.3.1** (30 min): Add mock tests for external dependencies
  ```python
  @patch('backend.ai_modules.classification.torch')
  def test_classification_without_models(self, mock_torch):
      # Test classification logic without loading actual models
  ```

- [ ] **Step 2.3.2** (30 min): Add mock tests for file operations
  ```python
  @patch('backend.utils.image_utils.Image.open')
  def test_image_processing_edge_cases(self, mock_open):
      # Test error handling without actual files
  ```

---

### Task 3: Configure Production Coverage Reporting (1 hour) - MEDIUM
**Problem**: Need reliable coverage tracking for ongoing development

#### Subtask 3.1: Setup Coverage Configuration (30 min)
**Files**: `.coveragerc`, `pytest.ini`
**Dependencies**: None
**Estimated Time**: 30 minutes

**Implementation Steps**:
- [ ] **Step 3.1.1** (15 min): Create comprehensive `.coveragerc`:
  ```ini
  [run]
  source = backend
  omit =
      */tests/*
      */venv/*
      */__pycache__/*
      */migrations/*

  [report]
  exclude_lines =
      pragma: no cover
      def __repr__
      raise AssertionError
      raise NotImplementedError

  [html]
  directory = coverage_html_report
  ```

- [ ] **Step 3.1.2** (15 min): Update `pytest.ini` for coverage requirements:
  ```ini
  [tool:pytest]
  testpaths = tests
  addopts = --cov=backend --cov-report=html --cov-report=json --cov-fail-under=80
  ```

#### Subtask 3.2: Automated Coverage Reporting (30 min)
**Files**: `scripts/coverage_report.py`
**Dependencies**: Subtask 3.1
**Estimated Time**: 30 minutes

**Implementation**:
- [ ] **Step 3.2.1** (30 min): Create automated coverage script:
  ```python
  #!/usr/bin/env python3
  """Generate and analyze coverage reports"""

  import subprocess
  import json
  from pathlib import Path

  def generate_coverage_report():
      # Run tests with coverage
      result = subprocess.run([
          'python', '-m', 'pytest',
          '--cov=backend',
          '--cov-report=json',
          'tests/'
      ], capture_output=True, text=True)

      # Analyze results
      with open('coverage.json') as f:
          coverage_data = json.load(f)

      total_coverage = coverage_data['totals']['percent_covered']
      print(f"Total Coverage: {total_coverage:.1f}%")

      # Identify low-coverage files
      for filename, data in coverage_data['files'].items():
          file_coverage = data['summary']['percent_covered']
          if file_coverage < 60:
              print(f"⚠️ Low coverage: {filename} ({file_coverage:.1f}%)")

      return total_coverage >= 80
  ```

---

## 📈 Progress Tracking

### Hourly Checkpoints
- **Hour 1**: ⏳ API endpoint debugging
- **Hour 2**: ⏳ Convert endpoint fixed
- **Hour 3**: ⏳ Optimize and batch endpoints fixed
- **Hour 4**: ⏳ Coverage gap analysis complete
- **Hour 5**: ⏳ High-impact tests created
- **Hour 6**: ⏳ Mock-based testing implemented
- **Hour 7**: ⏳ Coverage reporting configured
- **Hour 8**: ⏳ Day 2 validation complete

### Success Metrics Tracking
- [ ] API Tests Passing: ___/10 (Target: 10/10)
- [ ] Test Coverage: ___%  (Target: >80%)
- [ ] New Test Files: ___/3 (Target: 3/3)
- [ ] Coverage Report: WORKING/BROKEN

---

## 🔧 Tools & Commands

### API Testing
```bash
# Test specific failing endpoints
python -m pytest tests/test_api.py::TestAPIEndpoints::test_convert_endpoint -v
python -m pytest tests/test_api.py::TestAPIEndpoints::test_optimize_endpoint -v
python -m pytest tests/test_api.py::TestAPIEndpoints::test_batch_endpoint -v

# Full API test suite
python -m pytest tests/test_api.py -v
```

### Coverage Analysis
```bash
# Generate detailed coverage report
python -m pytest --cov=backend --cov-report=html --cov-report=json tests/

# View coverage in browser
open coverage_html_report/index.html

# Quick coverage check
python scripts/coverage_report.py
```

### Validation Pipeline
```bash
# Full validation
python -m pytest tests/ --cov=backend --cov-fail-under=80

# Performance check
python scripts/performance_regression_test.py
```

---

## 🚧 Risk Mitigation

### High Risk Items
1. **Coverage targets too aggressive**: Mitigation - Focus on high-impact areas first
2. **API changes break integration**: Mitigation - Maintain backward compatibility
3. **Mock tests don't reflect reality**: Mitigation - Combine with integration tests

### Quality Gates
- [ ] No existing tests should break during coverage improvements
- [ ] All API changes must be backward compatible
- [ ] Coverage increase must be sustainable (not just test padding)

---

## 📋 End of Day 2 Deliverables

### Required Outputs
- [ ] **API Status Report**: All 10 endpoints functional
- [ ] **Coverage Report**: Detailed HTML and JSON reports
- [ ] **New Test Files**: 3+ comprehensive test modules
- [ ] **Documentation**: Coverage setup and maintenance guide

### Quality Metrics
- [ ] Test Coverage: >80%
- [ ] API Test Success: 100%
- [ ] No regression in existing functionality
- [ ] Coverage reporting automation working

### Handoff to Day 3
- [ ] **Test Infrastructure**: Fully operational coverage tracking
- [ ] **API Stability**: All endpoints tested and working
- [ ] **Foundation**: Ready for performance optimization
- [ ] **Documentation**: Clear maintenance procedures

---

## 🎯 Day 2 Completion Criteria

**MANDATORY (All must pass)**:
✅ API tests: 10/10 passing
✅ Test coverage: >80%
✅ Coverage reporting: Functional
✅ No functionality regressions

**SUCCESS INDICATORS**:
- Coverage report shows >80% across all critical modules
- API endpoint suite: 100% passing
- Automated coverage tracking: Operational
- Test execution time: <5 minutes

**READY FOR DAY 3 IF**:
- All API endpoints stable and tested
- Coverage infrastructure operational
- Performance baseline maintained
- Ready for optimization work

---

*Day 2 focuses on establishing robust testing infrastructure and API stability - essential foundations for production deployment.*