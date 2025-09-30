# Method 1 Quick Reference

*Generated on 2025-09-29 11:44:02*

## Quick Start

### Basic Optimization
```python
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.feature_extraction import ImageFeatureExtractor

# Extract features and optimize
extractor = ImageFeatureExtractor()
features = extractor.extract_features("logo.png")

optimizer = FeatureMappingOptimizer()
result = optimizer.optimize(features)

print(f"Confidence: {result['confidence']:.1%}")
print(f"Parameters: {result['parameters']}")
```

### Batch Processing
```python
for image_path in image_list:
    features = extractor.extract_features(image_path)
    result = optimizer.optimize(features)
    # Use result['parameters'] with VTracer
```

## Core Classes

### FeatureMappingOptimizer
- `optimize(features)` → optimized parameters
- `calculate_confidence(features)` → confidence score

### CorrelationFormulas
- `edge_to_corner_threshold(density)` → corner threshold
- `colors_to_precision(colors)` → color precision
- `entropy_to_path_precision(entropy)` → path precision
- `corners_to_length_threshold(density)` → length threshold
- `gradient_to_splice_threshold(strength)` → splice threshold
- `complexity_to_iterations(score)` → max iterations

### VTracerParameterBounds
- `validate_parameters(params)` → validation result
- `get_default_parameters()` → default params
- `get_bounds()` → parameter bounds

### OptimizationQualityMetrics
- `measure_improvement(image, default, optimized)` → quality comparison

### OptimizationLogger
- `log_optimization(image, features, params, metrics)` → log results
- `calculate_statistics()` → performance stats
- `export_to_csv()` → export results

### OptimizationErrorHandler
- `detect_error(exception, context)` → classified error
- `attempt_recovery(error)` → recovery attempt
- `retry_with_backoff(operation, error_type)` → retry with backoff

## Parameter Ranges

| Parameter | Min | Max | Default | Description |
|-----------|-----|-----|---------|-------------|
| color_precision | 2 | 10 | 6 | Number of colors |
| corner_threshold | 10 | 110 | 50 | Corner sensitivity |
| length_threshold | 1.0 | 20.0 | 4.0 | Min path length |
| max_iterations | 5 | 20 | 10 | Optimization cycles |
| splice_threshold | 10 | 100 | 45 | Path splicing |
| path_precision | 1 | 20 | 8 | Path accuracy |
| layer_difference | 1 | 30 | 10 | Layer separation |
| mode | - | - | "spline" | "spline" or "polygon" |

## Logo Type Recommendations

### Simple Geometric
- **Features**: Low edge density, few colors, low entropy
- **Parameters**: color_precision=3-4, corner_threshold=30-50
- **Expected SSIM**: 95-99%

### Text-Based
- **Features**: High edge density, few colors, medium entropy
- **Parameters**: color_precision=2-3, corner_threshold=20-30, path_precision=8-10
- **Expected SSIM**: 90-99%

### Gradient
- **Features**: Medium edge density, many colors, high entropy
- **Parameters**: color_precision=8-10, layer_difference=5-8
- **Expected SSIM**: 85-97%

### Complex
- **Features**: High edge density, many colors, high entropy
- **Parameters**: max_iterations=15-20, splice_threshold=60-80
- **Expected SSIM**: 80-95%

## Error Types

| Error Type | Severity | Recovery Strategy |
|------------|----------|-------------------|
| FEATURE_EXTRACTION_FAILED | Medium | Use default features |
| PARAMETER_VALIDATION_FAILED | Medium | Sanitize parameters |
| VTRACER_CONVERSION_FAILED | High | Conservative parameters |
| QUALITY_MEASUREMENT_FAILED | Low | Skip measurement |
| INVALID_INPUT_IMAGE | Medium | Manual intervention |
| MEMORY_EXHAUSTION | Critical | Memory-efficient params |
| TIMEOUT_ERROR | Medium | High-speed params |

## Performance Targets

| Metric | Target | Typical | Notes |
|--------|--------|---------|-------|
| Optimization Speed | <0.05s | 0.01-0.03s | Per image |
| Memory Usage | <25MB | 10-15MB | Per optimization |
| Quality Improvement | >18% | 15-25% | SSIM improvement |
| Error Recovery Rate | >95% | 90-98% | Successful recovery |

## Configuration Files

### Environment Variables
```bash
OPTIMIZATION_LOG_LEVEL=INFO
OPTIMIZATION_CACHE_SIZE=1000
OPTIMIZATION_MAX_WORKERS=4
QUALITY_MEASUREMENT_ENABLED=true
ERROR_NOTIFICATION_ENABLED=false
```

## Troubleshooting

### Low Confidence
- Check image quality and format
- Verify feature extraction results
- Use manual parameter override

### Performance Issues
- Enable caching
- Reduce batch size
- Use fast parameters for speed

### Memory Issues
- Process smaller batches
- Call cleanup methods
- Use memory-efficient parameters

### VTracer Failures
- Use conservative parameters
- Check VTracer installation
- Enable circuit breaker

## Support

- **API Reference**: See `METHOD1_API_REFERENCE.md`
- **User Guide**: See `METHOD1_USER_GUIDE.md`
- **Troubleshooting**: See `METHOD1_TROUBLESHOOTING.md`
- **Configuration**: See `METHOD1_CONFIGURATION.md`
- **Deployment**: See `METHOD1_DEPLOYMENT.md`
