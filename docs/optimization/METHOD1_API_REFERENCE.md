# Method 1 API Reference

*Generated on 2025-09-29 11:44:02*

## Overview

Method 1 Parameter Optimization Engine provides intelligent parameter optimization for VTracer SVG conversion using correlation-based feature mapping.

## Core Components

### FeatureMappingOptimizer

The main optimization class that maps image features to optimal VTracer parameters.

```python
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

optimizer = FeatureMappingOptimizer()
result = optimizer.optimize(features)
```

#### Methods

##### `optimize(features: Dict[str, float]) -> Dict[str, Any]`

Generate optimized parameters from image features.

**Parameters:**
- `features`: Dictionary containing extracted image features:
  - `edge_density` (float): Edge density ratio [0.0, 1.0]
  - `unique_colors` (int): Number of unique colors [1, 256]
  - `entropy` (float): Image entropy [0.0, 1.0]
  - `corner_density` (float): Corner density ratio [0.0, 1.0]
  - `gradient_strength` (float): Gradient strength [0.0, 1.0]
  - `complexity_score` (float): Overall complexity [0.0, 1.0]

**Returns:**
```python
{
    "parameters": {
        "color_precision": int,      # [2, 10]
        "layer_difference": int,     # [1, 30]
        "corner_threshold": int,     # [10, 110]
        "length_threshold": float,   # [1.0, 20.0]
        "max_iterations": int,       # [5, 20]
        "splice_threshold": int,     # [10, 100]
        "path_precision": int,       # [1, 20]
        "mode": str                  # "spline" or "polygon"
    },
    "confidence": float,             # [0.0, 1.0]
    "metadata": {
        "optimization_method": str,
        "processing_timestamp": str,
        "correlations_used": List[str]
    }
}
```

### CorrelationFormulas

Static methods for converting features to parameters using validated mathematical formulas.

#### Methods

##### `edge_to_corner_threshold(edge_density: float) -> int`

Map edge density to corner threshold parameter.

**Formula:** `max(10, min(110, int(110 - (edge_density * 800))))`

**Logic:** Higher edge density → lower corner threshold for better detail capture

### VTracerParameterBounds

Parameter validation and bounds checking system.

#### Methods

##### `validate_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]`

Validate parameter set against VTracer bounds.

### OptimizationQualityMetrics

Quality measurement and comparison system.

#### Methods

##### `measure_improvement(image_path: str, default_params: Dict, optimized_params: Dict, runs: int = 3) -> Dict`

Measure quality improvement between parameter sets.

### OptimizationErrorHandler

Comprehensive error handling and recovery system.

#### Methods

##### `detect_error(exception: Exception, context: Dict = None) -> OptimizationError`

Detect and classify optimization errors.

##### `attempt_recovery(error: OptimizationError, **kwargs) -> Dict[str, Any]`

Attempt error recovery using appropriate strategy.

## Performance Characteristics

### Optimization Speed
- **Target**: <0.05s per image (50ms)
- **Typical**: 0.01-0.03s for simple images
- **Range**: 0.005-0.1s depending on complexity

### Memory Usage
- **Target**: <25MB per optimization
- **Typical**: 10-15MB for standard operations
- **Peak**: 20-30MB for complex images

### Quality Improvement
- **Simple Logos**: 95-99% SSIM (18-28% improvement)
- **Text Logos**: 90-99% SSIM (15-25% improvement)
- **Gradient Logos**: 85-97% SSIM (12-20% improvement)
- **Complex Logos**: 80-95% SSIM (8-16% improvement)

## Version Information

- **Method Version**: 1.0
- **API Version**: 1.0
- **Documentation Version**: 1.0
- **Last Updated**: 2025-09-29
