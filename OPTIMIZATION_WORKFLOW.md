# 🎯 Iterative SVG Optimization Workflow

## Overview

The optimization workflow automatically tunes VTracer parameters to achieve target quality levels for PNG to SVG conversion. It iteratively adjusts parameters based on quality feedback until the desired SSIM (Structural Similarity Index) is reached.

## Quick Start

### Basic Usage
```bash
# Simple optimization with default target (85% SSIM)
python optimize_iterative.py logo.png

# Specify quality target
python optimize_iterative.py logo.png --target-ssim 0.9

# With verbose output
python optimize_iterative.py logo.png --verbose --max-iterations 10
```

### Examples

#### High Quality Text Logo
```bash
python optimize_iterative.py data/logos/text_based/text_tech_00.png \
    --preset text \
    --target-ssim 0.95 \
    --verbose
```

#### Complex Logo with History
```bash
python optimize_iterative.py complex_logo.png \
    --target-ssim 0.8 \
    --max-iterations 15 \
    --save-history \
    --verbose
```

## How It Works

### 1. Logo Type Detection
The system automatically analyzes the input image to determine its type:
- **Simple**: Few colors (≤4), no gradients
- **Text**: Limited colors (≤8), sharp edges
- **Gradient**: Smooth color transitions detected
- **Complex**: Many colors, mixed elements

### 2. Parameter Optimization Loop

```
Start → Detect Type → Apply Preset → Convert → Measure Quality
         ↑                                            ↓
         ← Adjust Parameters ← Quality Check ←────────┘
```

### 3. Quality Measurement
- Renders SVG back to PNG
- Compares with original using SSIM
- Also tracks MSE and PSNR metrics
- Monitors file size

### 4. Parameter Adjustment Strategy

Based on quality gap from target:

**Large Gap (>0.2 SSIM)**
- Increase color_precision by 2
- Increase path_precision by 2
- Decrease corner_threshold by 10
- Increase max_iterations by 5

**Medium Gap (0.1-0.2 SSIM)**
- Increase color_precision by 1
- Increase path_precision by 1
- Decrease layer_difference by 4
- Decrease corner_threshold by 5

**Small Gap (<0.1 SSIM)**
- Fine-tune path_precision
- Adjust length_threshold

## VTracer Parameters Explained

### Core Parameters

| Parameter | Range | Impact | Best For |
|-----------|-------|--------|----------|
| **color_precision** | 1-10 | Number of color levels | Higher = more colors preserved |
| **path_precision** | 0-10 | Decimal places in coordinates | Higher = smoother curves |
| **corner_threshold** | 10-180 | Angle to detect corners (degrees) | Lower = sharper corners |
| **layer_difference** | 4-64 | Min color diff between layers | Lower = more gradients |
| **length_threshold** | 0.5-10 | Min path length to keep | Lower = more detail |
| **splice_threshold** | 10-90 | Angle to merge paths | Higher = simpler paths |
| **max_iterations** | 5-30 | Optimization passes | Higher = better quality |

### Preset Configurations

**Simple Geometric**
```python
{
    'color_precision': 4,      # Few colors needed
    'corner_threshold': 30,     # Sharp corners
    'path_precision': 8,        # High precision
    'layer_difference': 32      # Clear separation
}
```

**Text-Based**
```python
{
    'color_precision': 2,       # Usually monochrome
    'corner_threshold': 20,     # Very sharp
    'path_precision': 10,       # Maximum precision
    'length_threshold': 1.0     # Keep small details
}
```

**Gradient**
```python
{
    'color_precision': 8,       # Many color levels
    'layer_difference': 8,      # Smooth transitions
    'max_iterations': 15        # More optimization
}
```

## Quality Targets

### Recommended SSIM Targets by Use Case

| Use Case | Target SSIM | Rationale |
|----------|-------------|-----------|
| **Icons/UI** | 0.90-0.95 | Need sharp, accurate representation |
| **Logos** | 0.85-0.90 | Balance quality and file size |
| **Illustrations** | 0.75-0.85 | Artistic interpretation acceptable |
| **Complex Images** | 0.70-0.80 | Difficult to vectorize perfectly |

### Understanding SSIM Scores

- **0.95+**: Nearly perfect, pixel-level accuracy
- **0.90-0.95**: Excellent, minor differences
- **0.85-0.90**: Good, suitable for most uses
- **0.80-0.85**: Acceptable, visible differences
- **0.70-0.80**: Fair, noticeable simplification
- **<0.70**: Poor, significant loss of detail

## Advanced Features

### Save Optimization History
```bash
python optimize_iterative.py logo.png --save-history
```

Creates `logo.optimization.json` with:
- All parameter combinations tried
- Quality scores for each iteration
- Best parameters found
- Complete metrics

### Batch Optimization
```bash
# Process entire directory
for file in data/logos/*.png; do
    python optimize_iterative.py "$file" --target-ssim 0.85
done
```

### Custom Parameter Ranges
Modify `optimize_iterative.py`:
```python
self.param_ranges = {
    'color_precision': (1, 10),    # Adjust range
    'corner_threshold': (5, 90),   # More aggressive
    # ... etc
}
```

## SVG Post-Processing

After optimization, further reduce file size:

```bash
# Install SVGO
npm install -g svgo

# Optimize SVG
svgo logo.optimized.svg -o logo.final.svg
```

## Visual Comparison

Use the visual comparison tools:

```python
from utils.visual_compare import VisualComparer

comparer = VisualComparer()
grid = comparer.create_comparison_grid(
    "original.png",
    svg_content
)
grid.save("comparison.png")
```

## Performance Tips

1. **Start with lower targets**: Begin with 0.80 SSIM, increase if needed
2. **Use presets**: Let the system detect logo type automatically
3. **Limit iterations**: 5-10 is usually sufficient
4. **Cache results**: System caches conversions automatically
5. **Batch similar files**: Group by type for efficiency

## Troubleshooting

### Low Quality Results
- Increase `max_iterations`
- Lower `corner_threshold` for sharper edges
- Increase `color_precision` for better color matching

### Large File Sizes
- Reduce `path_precision` (try 4-6)
- Increase `length_threshold` to remove small paths
- Use SVG post-processor

### Slow Conversion
- Reduce `max_iterations`
- Use simpler presets
- Process smaller batches

## Integration Example

```python
from optimize_iterative import IterativeOptimizer

# Initialize optimizer
optimizer = IterativeOptimizer(
    target_ssim=0.85,
    max_iterations=10
)

# Run optimization
result = optimizer.optimize(
    "logo.png",
    preset="auto",  # Auto-detect
    verbose=True
)

# Access results
print(f"Best SSIM: {result['best_ssim']}")
print(f"Best params: {result['best_params']}")

# Save optimized SVG
with open("optimized.svg", "w") as f:
    f.write(result['best_svg'])
```

## Results Summary

Typical results from optimization:

| Logo Type | Iterations | Final SSIM | Size Reduction |
|-----------|------------|------------|----------------|
| Simple Geometric | 1-2 | 0.95-0.99 | 70-80% |
| Text-Based | 2-3 | 0.90-0.95 | 60-70% |
| Gradient | 3-5 | 0.85-0.90 | 50-60% |
| Complex | 5-10 | 0.75-0.85 | 40-50% |

## Next Steps

1. **Test on your logos**: Run optimization on actual files
2. **Tune targets**: Adjust SSIM based on visual inspection
3. **Build pipeline**: Integrate into production workflow
4. **Monitor metrics**: Track quality and performance

---

The optimization workflow provides a systematic approach to achieving high-quality PNG to SVG conversions with automatic parameter tuning. It eliminates guesswork and ensures consistent, measurable results.