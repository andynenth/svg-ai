# Method 1 User Guide

*Generated on 2025-09-29 11:44:02*

## Introduction

Method 1 Parameter Optimization Engine automatically optimizes VTracer parameters for SVG conversion based on image characteristics. This guide shows you how to use Method 1 for optimal SVG conversion results.

## Quick Start

### Basic Usage

```python
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.feature_extraction import ImageFeatureExtractor

# Extract features from your image
extractor = ImageFeatureExtractor()
features = extractor.extract_features("logo.png")

# Optimize parameters
optimizer = FeatureMappingOptimizer()
result = optimizer.optimize(features)

# Use optimized parameters
optimized_params = result["parameters"]
confidence = result["confidence"]

print(f"Optimization confidence: {confidence:.1%}")
print(f"Recommended parameters: {optimized_params}")
```

### Single Image Optimization

```python
import vtracer
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.feature_extraction import ImageFeatureExtractor

def optimize_single_image(image_path: str, output_path: str):
    # Extract features
    extractor = ImageFeatureExtractor()
    features = extractor.extract_features(image_path)

    # Optimize parameters
    optimizer = FeatureMappingOptimizer()
    result = optimizer.optimize(features)

    # Convert with optimized parameters
    vtracer.convert_image_to_svg_py(
        image_path,
        output_path,
        **result["parameters"]
    )

    return result

# Example usage
result = optimize_single_image("logo.png", "logo.svg")
print(f"Optimization complete with {result['confidence']:.1%} confidence")
```

## Parameter Recommendations by Logo Type

### Simple Geometric Logos
- **Best for**: Circles, squares, basic shapes
- **Typical parameters**: Low color precision (3-4), medium corner threshold (30-50)
- **Expected quality**: 95-99% SSIM

### Text-Based Logos
- **Best for**: Logos with text elements
- **Typical parameters**: Low color precision (2-3), low corner threshold (20-30), high path precision (8-10)
- **Expected quality**: 90-99% SSIM

### Gradient Logos
- **Best for**: Smooth color transitions
- **Typical parameters**: High color precision (8-10), low layer difference (5-8)
- **Expected quality**: 85-97% SSIM

### Complex Logos
- **Best for**: Detailed illustrations
- **Typical parameters**: High iterations (15-20), high splice threshold (60-80)
- **Expected quality**: 80-95% SSIM

## Error Handling

### Robust Optimization with Error Recovery

```python
from backend.ai_modules.optimization.error_handler import OptimizationErrorHandler
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

def robust_optimize(image_path: str):
    error_handler = OptimizationErrorHandler()
    optimizer = FeatureMappingOptimizer()

    try:
        # Extract features with retry
        features = error_handler.retry_with_backoff(
            lambda: ImageFeatureExtractor().extract_features(image_path),
            OptimizationErrorType.FEATURE_EXTRACTION_FAILED
        )

        # Optimize parameters
        result = optimizer.optimize(features)
        return result

    except Exception as e:
        # Detect and handle error
        error = error_handler.detect_error(e, {"image_path": image_path})
        recovery = error_handler.attempt_recovery(error)

        if recovery["success"]:
            print(f"Recovered from error: {recovery['message']}")
            return {"parameters": recovery.get("fallback_parameters", {})}
        else:
            print(f"Could not recover from error: {error.message}")
            raise
```

## Best Practices

### Image Preparation
- Use high-quality PNG images (recommended: 300+ DPI)
- Ensure clear contrast between elements
- Avoid overly complex images for best results

### Parameter Selection
- Start with Method 1 optimization
- Fine-tune based on specific requirements
- Test with representative sample images

### Quality Assurance
- Always measure quality improvements
- Compare results visually
- Test across different logo types

### Performance
- Use batch processing for multiple images
- Enable logging for analysis
- Monitor error rates and adjust accordingly

## Support

For technical support or questions:
- Review the API Reference for detailed method documentation
- Check the Troubleshooting Guide for common issues
- Examine error logs for specific error messages
