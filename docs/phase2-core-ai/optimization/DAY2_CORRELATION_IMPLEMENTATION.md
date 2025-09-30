# Day 2: Correlation Implementation - Parameter Optimization Engine

**Date**: Week 3, Day 2 (Tuesday)
**Duration**: 8 hours
**Team**: 2 developers
**Objective**: Implement mathematical correlations and feature mapping optimizer

---

## Prerequisites Verification

Ensure Day 1 deliverables are complete:
- [ ] Parameter bounds system working
- [ ] Parameter validator functional
- [ ] VTracer test harness operational
- [ ] Test infrastructure ready
- [ ] Correlation research documented

---

## Developer A Tasks (8 hours)

### Task A2.1: Implement Core Correlation Formulas ⏱️ 4 hours

**Objective**: Convert research into working correlation formulas.

**Implementation**:
```python
# backend/ai_modules/optimization/correlation_formulas.py
import numpy as np
from typing import Dict, Any
import logging

class CorrelationFormulas:
    """Research-validated parameter correlations"""

    @staticmethod
    def edge_to_corner_threshold(edge_density: float) -> int:
        """Map edge density to corner threshold parameter

        Formula: corner_threshold = max(10, min(110, int(110 - (edge_density * 800))))
        Logic: Higher edge density → lower corner threshold for better detail
        """
        raw_value = 110 - (edge_density * 800)
        return max(10, min(110, int(raw_value)))

    @staticmethod
    def colors_to_precision(unique_colors: float) -> int:
        """Map unique colors to color precision parameter"""
        # Implementation here
```

**Detailed Checklist**:
- [x] Implement `edge_to_corner_threshold()` formula
  - Range: edge_density [0,1] → corner_threshold [10,110]
  - Test with edge cases: 0.0, 0.5, 1.0
- [x] Implement `colors_to_precision()` formula
  - Formula: `max(2, min(10, int(2 + np.log2(max(1, unique_colors)))))`
  - Handle zero/negative colors gracefully
- [x] Implement `entropy_to_path_precision()` formula
  - Formula: `max(1, min(20, int(20 * (1 - entropy))))`
  - Higher entropy → higher precision
- [x] Implement `corners_to_length_threshold()` formula
  - Formula: `max(1.0, min(20.0, 1.0 + (corner_density * 100)))`
  - More corners → shorter segments
- [x] Implement `gradient_to_splice_threshold()` formula
  - Formula: `max(10, min(100, int(10 + (gradient_strength * 90))))`
  - Stronger gradients → more splice points
- [x] Implement `complexity_to_iterations()` formula
  - Formula: `max(5, min(20, int(5 + (complexity_score * 15))))`
  - Higher complexity → more iterations
- [x] Add comprehensive input validation for all formulas
- [x] Create formula testing utility with known inputs/outputs
- [x] Write detailed docstrings with mathematical justification
- [x] Add logging for correlation decisions

**Deliverable**: Complete correlation formula implementation

### Task A2.2: Create Feature Mapping Optimizer ⏱️ 4 hours

**Objective**: Build the main Method 1 optimizer that uses correlation formulas.

**Implementation**:
```python
# backend/ai_modules/optimization/feature_mapping.py
from typing import Dict, Any, Tuple
from .correlation_formulas import CorrelationFormulas
from .parameter_bounds import VTracerParameterBounds
import logging

class FeatureMappingOptimizer:
    """Map image features to optimal VTracer parameters using correlations"""

    def __init__(self):
        self.formulas = CorrelationFormulas()
        self.bounds = VTracerParameterBounds()
        self.logger = logging.getLogger(__name__)

    def optimize(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate optimized parameters from image features"""
        # Implementation here

    def calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for optimization"""
        # Implementation here
```

**Detailed Checklist**:
- [x] Implement main `optimize()` method
  - Apply all 6 correlation formulas
  - Use bounds system for validation
  - Generate complete parameter set
- [x] Apply correlation formulas in sequence:
  - edge_density → corner_threshold
  - unique_colors → color_precision
  - entropy → path_precision
  - corner_density → length_threshold
  - gradient_strength → splice_threshold
  - complexity_score → max_iterations
- [x] Set default values for non-correlated parameters
  - layer_difference: use default (10)
  - mode: choose based on complexity ('spline' for complex, 'polygon' for simple)
- [x] Implement confidence scoring (0-1 range)
  - Base confidence on feature quality
  - Penalize extreme values
  - Higher confidence for well-distributed features
- [x] Add optimization metadata generation
  - Record which correlations were used
  - Include confidence explanations
  - Add processing timestamp
- [x] Create parameter explanation system
  - Explain why each parameter was chosen
  - Provide human-readable reasoning
- [x] Implement basic caching for repeated feature sets
- [x] Add comprehensive error handling
- [x] Write detailed unit tests covering all scenarios
- [x] Test with real image features from test dataset

**Deliverable**: Complete Method 1 optimizer

---

## Developer B Tasks (8 hours)

### Task B2.1: Implement Quality Measurement System ⏱️ 4 hours

**Objective**: Create system to measure optimization effectiveness.

**Implementation**:
```python
# backend/ai_modules/optimization/quality_metrics.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, Optional
import tempfile
import os

class OptimizationQualityMetrics:
    """Measure optimization quality improvements"""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()

    def measure_improvement(self,
                           image_path: str,
                           default_params: Dict,
                           optimized_params: Dict) -> Dict:
        """Compare quality between parameter sets"""
        # Implementation here
```

**Detailed Checklist**:
- [x] Implement SSIM comparison function
  - Convert both SVGs to PNG for comparison
  - Use same dimensions for fair comparison
  - Handle conversion errors gracefully
- [x] Add file size comparison metric
  - Compare SVG file sizes
  - Calculate compression ratio
  - Account for quality vs size tradeoff
- [x] Calculate processing time differences
  - Measure VTracer conversion time for both parameter sets
  - Account for system variability
  - Average over multiple runs if needed
- [x] Implement visual quality scorer
  - Edge preservation metric
  - Color accuracy measurement
  - Shape fidelity assessment
- [x] Create improvement percentage calculator
  - SSIM improvement: (new_ssim - old_ssim) / old_ssim * 100
  - File size change: (old_size - new_size) / old_size * 100
  - Speed improvement calculation
- [x] Add statistical significance testing
  - Use t-test for improvement significance
  - Calculate confidence intervals
  - Handle small sample sizes
- [x] Generate detailed quality report structure
  - JSON format for programmatic use
  - Human-readable summary
  - Visualization data for charts
- [x] Write comprehensive unit tests
- [x] Test with known good/bad parameter combinations
- [x] Validate metrics correlate with visual assessment

**Deliverable**: Comprehensive quality measurement system

### Task B2.2: Create Optimization Logger & Analytics ⏱️ 4 hours

**Objective**: Build logging and analytics system for optimization results.

**Implementation**:
```python
# backend/ai_modules/optimization/optimization_logger.py
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class OptimizationLogger:
    """Log and analyze optimization results"""

    def __init__(self, log_dir: str = "logs/optimization"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def log_optimization(self,
                        image_path: str,
                        features: Dict,
                        params: Dict,
                        quality_metrics: Dict):
        """Log detailed optimization results"""
        # Implementation here
```

**Detailed Checklist**:
- [x] Setup structured logging format
  - Create consistent log entry structure
  - Include timestamp, image info, parameters, results
  - Use JSON format for structured data
- [x] Implement CSV export functionality
  - Export optimization results to CSV for analysis
  - Include all relevant metrics and parameters
  - Handle large datasets efficiently
- [x] Add JSON serialization for complex results
  - Handle numpy arrays and custom objects
  - Ensure all data is JSON serializable
  - Create compact but complete format
- [x] Create performance tracking system
  - Track optimization times over sessions
  - Monitor quality improvement trends
  - Identify performance regressions
- [x] Build statistical analysis utilities
  - Calculate average improvements by logo type
  - Identify best/worst performing images
  - Generate correlation statistics
- [x] Add visualization generation helpers
  - Create data structures for plotting
  - Export chart-ready JSON format
  - Generate summary statistics
- [x] Implement log rotation and cleanup
  - Prevent log files from growing too large
  - Archive old results
  - Clean up temporary files
- [x] Create analysis dashboard template
  - HTML template for viewing results
  - JavaScript for interactive charts
  - Export functionality for reports
- [x] Write unit tests for all logging functions
- [x] Test with various optimization scenarios
- [x] Verify data integrity and completeness

**Deliverable**: Complete logging and analytics system

---

## Integration Tasks (Both Developers - 30 minutes)

### Task AB2.3: Integration Testing

**Objective**: Verify all Day 2 components work together.

**Integration Test**:
```python
def test_day2_integration():
    """Test correlation formulas + optimizer + quality measurement"""

    # Load test image features
    features = {
        'edge_density': 0.15,
        'unique_colors': 12,
        'entropy': 0.65,
        'corner_density': 0.08,
        'gradient_strength': 0.45,
        'complexity_score': 0.35
    }

    # Run optimization
    optimizer = FeatureMappingOptimizer()
    result = optimizer.optimize(features)

    # Validate results
    assert all(key in result for key in ['parameters', 'confidence', 'metadata'])
    assert result['confidence'] > 0.0

    print(f"✅ Optimization complete: {result['parameters']}")
```

**Checklist**:
- [x] Test feature mapping with real features
- [x] Verify all parameters are within bounds
- [x] Test quality measurement with sample images (Developer B)
- [x] Check logging system captures all data (Developer B)
- [x] Run integration test successfully

---

## End-of-Day Validation

### Functional Testing
- [x] Correlation formulas produce reasonable outputs
- [x] Feature mapping optimizer generates valid parameters
- [x] Quality measurement system works with test images (Developer B)
- [x] Logging system captures complete optimization data (Developer B)

### Performance Testing
- [x] Optimization completes in <0.1s
- [x] Quality measurement completes in <5s per comparison
- [x] No memory leaks in repeated operations
- [x] Logging doesn't slow down optimization significantly

### Quality Verification
- [x] Optimized parameters show visual improvement over defaults
- [x] Quality metrics correlate with visual assessment
- [x] Confidence scores reflect optimization reliability

---

## Tomorrow's Preparation

**Day 3 Focus**: Comprehensive testing and validation

**Prerequisites for Day 3**:
- [x] All correlation formulas working correctly
- [x] Feature mapping optimizer functional
- [x] Quality measurement system operational
- [x] Logging capturing complete data

**Day 3 Preview**:
- Developer A: Build comprehensive unit test suite and integration tests
- Developer B: Create benchmark system and validation pipeline

---

## Success Criteria

✅ **Day 2 Success Indicators**:
- Correlation formulas convert features to parameters correctly
- Feature mapping optimizer produces complete parameter sets
- Quality measurement can compare parameter effectiveness
- Logging system captures optimization data

**Files Created**:
- `backend/ai_modules/optimization/correlation_formulas.py`
- `backend/ai_modules/optimization/feature_mapping.py`
- `backend/ai_modules/optimization/quality_metrics.py`
- `backend/ai_modules/optimization/optimization_logger.py`
- Unit tests for all components

**Key Metrics**:
- Optimization time: <0.1s ✅
- Parameter validity: 100% ✅
- Quality measurement accuracy: Visual validation ✅