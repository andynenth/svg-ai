# SVG Converter Parameter Guide

A comprehensive guide to optimizing conversion parameters for different types of images.

## Quick Reference

| Image Type | Recommended Converter | Key Settings |
|------------|----------------------|--------------|
| **Icons with transparency** | Alpha-aware | Low threshold (64), Clean edges ON |
| **Simple logos (B&W)** | Potrace | White corners, Remove noise (5), High smoothness |
| **Colorful logos** | VTracer | Color mode, 6-8 colors, High quality |
| **Complex illustrations** | VTracer | Color mode, 8-10 colors, High precision |
| **Line art/sketches** | Potrace | Black corners, Low noise removal |

---

## Alpha-aware Converter Examples

### Example 1: Clean Icon Conversion
**Image Type**: Simple icon with transparency (e.g., app icon, UI element)

**Optimal Settings**:
```json
{
  "converter": "alpha",
  "threshold": 64,
  "use_potrace": true,
  "preserve_antialiasing": false
}
```

**Why These Settings**:
- **Low threshold (64)**: Captures semi-transparent edges for smoother appearance
- **Clean edges ON**: Uses Potrace for crisp, scalable edges
- **Anti-aliasing OFF**: Keeps file size small while maintaining clarity

**Expected Results**: 95-98% SSIM, 70-80% file size reduction

### Example 2: Detailed Icon with Soft Edges
**Image Type**: Complex icon with gradients or soft shadows

**Optimal Settings**:
```json
{
  "converter": "alpha",
  "threshold": 32,
  "use_potrace": false,
  "preserve_antialiasing": true
}
```

**Why These Settings**:
- **Very low threshold (32)**: Preserves subtle transparency effects
- **Clean edges OFF**: Maintains original soft appearance
- **Anti-aliasing ON**: Preserves smooth gradients and shadows

**Expected Results**: 85-92% SSIM, larger file but higher fidelity

---

## Potrace Converter Examples

### Example 3: Corporate Logo (Clean Lines)
**Image Type**: Simple corporate logo with clean typography

**Optimal Settings**:
```json
{
  "converter": "potrace",
  "threshold": 128,
  "turnpolicy": "white",
  "turdsize": 5,
  "alphamax": 1.2,
  "opttolerance": 0.1
}
```

**Why These Settings**:
- **Standard threshold (128)**: Works well for most logos
- **White corners**: Creates smooth, professional curves
- **Noise removal (5)**: Eliminates small artifacts
- **High smoothness (1.2)**: Professional, polished appearance
- **High accuracy (0.1)**: Precise curve matching

**Expected Results**: 96-99% SSIM, excellent scalability

### Example 4: Hand-drawn Sketch
**Image Type**: Rough sketch or hand-drawn artwork

**Optimal Settings**:
```json
{
  "converter": "potrace",
  "threshold": 140,
  "turnpolicy": "black",
  "turdsize": 1,
  "alphamax": 1.0,
  "opttolerance": 0.2
}
```

**Why These Settings**:
- **Higher threshold (140)**: Captures only darker lines
- **Black corners**: Preserves sharp, sketch-like appearance
- **Minimal noise removal (1)**: Keeps texture and character
- **Standard smoothness (1.0)**: Maintains hand-drawn feel

**Expected Results**: 88-94% SSIM, preserves artistic character

### Example 5: Fine Text/Typography
**Image Type**: Text logos or fine typography

**Optimal Settings**:
```json
{
  "converter": "potrace",
  "threshold": 120,
  "turnpolicy": "white",
  "turdsize": 2,
  "alphamax": 1.1,
  "opttolerance": 0.05
}
```

**Why These Settings**:
- **Lower threshold (120)**: Captures thinner strokes
- **White corners**: Smooth letter curves
- **Small noise removal (2)**: Preserves fine details
- **Moderate smoothness (1.1)**: Readable at all sizes
- **Very high accuracy (0.05)**: Precise letterforms

**Expected Results**: 97-99% SSIM, perfect text rendering

---

## VTracer Converter Examples

### Example 6: Colorful Brand Logo
**Image Type**: Multi-color logo with distinct color areas

**Optimal Settings**:
```json
{
  "converter": "vtracer",
  "colormode": "color",
  "color_precision": 6,
  "layer_difference": 16,
  "path_precision": 5,
  "corner_threshold": 60,
  "length_threshold": 5.0,
  "max_iterations": 10,
  "splice_threshold": 45
}
```

**Why These Settings**:
- **Color mode**: Preserves original colors
- **6 colors**: Good balance of simplicity and accuracy
- **Standard settings**: Balanced quality and file size

**Expected Results**: 92-96% SSIM, good color reproduction

### Example 7: Complex Illustration
**Image Type**: Detailed artwork with many colors and gradients

**Optimal Settings**:
```json
{
  "converter": "vtracer",
  "colormode": "color",
  "color_precision": 8,
  "layer_difference": 8,
  "path_precision": 8,
  "corner_threshold": 30,
  "length_threshold": 2.0,
  "max_iterations": 20,
  "splice_threshold": 60
}
```

**Why These Settings**:
- **High color precision (8)**: Captures color nuances
- **Fine gradients (8)**: Smooth color transitions
- **High path precision (8)**: Detailed curves
- **Gentle corners (30)**: Smooth organic shapes
- **Keep short paths (2.0)**: Preserves fine details
- **More iterations (20)**: Higher quality result

**Expected Results**: 85-92% SSIM, excellent detail preservation

### Example 8: Simple Geometric Logo
**Image Type**: Logo with basic shapes and few colors

**Optimal Settings**:
```json
{
  "converter": "vtracer",
  "colormode": "color",
  "color_precision": 4,
  "layer_difference": 24,
  "path_precision": 3,
  "corner_threshold": 90,
  "length_threshold": 8.0,
  "max_iterations": 5,
  "splice_threshold": 30
}
```

**Why These Settings**:
- **Few colors (4)**: Simple, clean appearance
- **Larger color differences (24)**: Distinct color areas
- **Lower precision (3)**: Simpler paths
- **Sharp corners (90)**: Geometric appearance
- **Fast processing (5 iterations)**: Quick conversion

**Expected Results**: 94-98% SSIM, clean geometric output

---

## Parameter Interaction Guide

### How Parameters Affect Each Other

#### Potrace Interactions:
- **Higher Smoothness + Lower Accuracy**: Very smooth but less precise curves
- **Lower Threshold + Higher Noise Removal**: Captures more detail while staying clean
- **White Corners + High Smoothness**: Professional, polished look

#### VTracer Interactions:
- **More Colors + Lower Layer Difference**: Captures subtle color variations
- **Higher Path Precision + More Iterations**: Maximum quality but slower conversion
- **Gentle Corners + High Splice Threshold**: Smooth, connected shapes

#### Alpha-aware Interactions:
- **Low Threshold + Anti-aliasing ON**: Maximum detail preservation
- **High Threshold + Clean Edges ON**: Crisp, minimal file size

### Common Optimization Strategies

#### For Speed:
- Use preset "Fast" settings
- Lower iteration counts (VTracer)
- Higher thresholds
- Less noise removal (Potrace)

#### For Quality:
- Use preset "Quality" settings
- Higher iteration counts
- Lower thresholds for detail capture
- More color precision (VTracer)

#### For File Size:
- Use Alpha-aware for icons
- Lower color precision (VTracer)
- Higher layer differences
- Clean edges without anti-aliasing

---

## Troubleshooting Common Issues

### Issue: SVG appears too simple/loses detail
**Solution**:
- Lower threshold values
- Increase color precision (VTracer)
- Reduce layer difference (VTracer)
- Lower corner threshold for more corners

### Issue: SVG file too large
**Solution**:
- Use Alpha-aware for icons
- Reduce color precision
- Increase layer difference
- Turn off anti-aliasing
- Higher corner threshold for fewer points

### Issue: Curves look jagged
**Solution**:
- Increase smoothness (Potrace)
- Use "white" corner policy (Potrace)
- Increase path precision (VTracer)
- More iterations for smoother curves

### Issue: Small details disappear
**Solution**:
- Lower threshold values
- Reduce noise removal (Potrace)
- Lower length threshold (VTracer)
- Increase color precision

### Issue: Wrong colors in output
**Solution**:
- Use VTracer for color images
- Increase color precision
- Reduce layer difference
- Check if image has transparency (use Alpha-aware)

---

## API Usage Examples

### Upload and Convert with Optimal Settings

#### For Icons:
```bash
# Upload
curl -X POST http://localhost:8001/api/upload -F "file=@icon.png"
# Returns: {"file_id": "abc123..."}

# Convert
curl -X POST http://localhost:8001/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123...",
    "converter": "alpha",
    "threshold": 64,
    "use_potrace": true,
    "preserve_antialiasing": false
  }'
```

#### For Logos:
```bash
curl -X POST http://localhost:8001/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123...",
    "converter": "potrace",
    "threshold": 128,
    "turnpolicy": "white",
    "turdsize": 5,
    "alphamax": 1.2,
    "opttolerance": 0.1
  }'
```

#### For Complex Graphics:
```bash
curl -X POST http://localhost:8001/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123...",
    "converter": "vtracer",
    "colormode": "color",
    "color_precision": 8,
    "layer_difference": 8,
    "path_precision": 8,
    "corner_threshold": 30,
    "max_iterations": 20
  }'
```

---

## Advanced Tips

### 1. Converter Selection Strategy
1. **Try Alpha-aware first** for any PNG with transparency
2. **Use Potrace** for black/white content or when you need perfect curves
3. **Use VTracer** for color content or when other converters don't work well

### 2. Parameter Testing Workflow
1. Start with preset "Quality" settings
2. Adjust threshold first (most impact)
3. Fine-tune smoothness/precision
4. Optimize for file size if needed

### 3. Quality Assessment
- **SSIM > 95%**: Excellent quality
- **SSIM 90-95%**: Good quality
- **SSIM 85-90%**: Acceptable quality
- **SSIM < 85%**: Consider different converter or settings

### 4. File Size Optimization
- Alpha-aware typically produces smallest files for icons
- Potrace produces very small files for B&W content
- VTracer files scale with color complexity

---

*This guide covers the most common use cases. For specific needs, experiment with the web interface's real-time preview to find optimal settings.*