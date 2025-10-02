# How SVG-AI Works

## Overview
SVG-AI converts raster images (PNG) to scalable vector graphics (SVG) using computer vision and optimization techniques.

## Core Conversion Pipeline

```
PNG Image → VTracer Engine → SVG Output
    ↓            ↓              ↓
[Pixels]   [Vectorization]  [Paths & Curves]
```

## Step-by-Step Process

### 1. **Input Processing**
- You provide a PNG image (like a logo)
- System analyzes image properties:
  - Dimensions
  - Color complexity
  - Shape characteristics

### 2. **Vectorization (VTracer)**
The core uses VTracer, a Rust-based algorithm that:
- Traces edges and boundaries in the image
- Converts pixel regions into mathematical curves (Bézier paths)
- Groups similar colors into layers
- Optimizes path complexity

**Key Parameters VTracer Uses:**
- `color_precision`: How many colors to preserve (1-10)
- `corner_threshold`: How sharp corners should be (0-180°)
- `segment_length`: How detailed curves are
- `path_precision`: Accuracy of path tracing

### 3. **Quality Assessment**
After conversion, the system:
- Renders the SVG back to PNG
- Compares with original using SSIM (Structural Similarity)
- Calculates quality metrics:
  - **SSIM**: 0-1 score (higher = better match)
  - **MSE**: Mean Squared Error (lower = better)
  - **PSNR**: Peak Signal-to-Noise Ratio (higher = better)

### 4. **Optimization**
The system can iteratively:
- Adjust VTracer parameters
- Test different settings
- Find optimal balance between quality and file size

## Example Workflow

```python
# Simple usage
from backend.converter import convert_image

# Convert a logo
result = convert_image("company_logo.png")

# What you get back:
{
    "success": True,
    "svg": "<svg>...</svg>",  # The actual SVG code
    "ssim": 0.99,             # 99% similarity to original
    "mse": 50.0,              # Error measurement
    "psnr": 31.0,             # Quality ratio
    "file_size": 2048         # Size in bytes
}
```

## Why Convert PNG to SVG?

**PNG (Raster):**
- Fixed resolution
- Gets pixelated when scaled
- Larger file sizes
- Good for photos

**SVG (Vector):**
- Infinitely scalable
- Crystal clear at any size
- Smaller file sizes (usually)
- Perfect for logos, icons, graphics

## Real-World Example

**Input:** 100x100px company logo (PNG, 15KB)
**Output:** Scalable SVG (3KB)

Benefits:
- Works on retina displays
- Scales from favicon to billboard
- 80% smaller file size
- Can be edited in vector programs

## Quality Levels

The system adapts to different logo types:

1. **Simple Geometric** (circles, squares)
   - Achieves 98-99% SSIM
   - Very small file size

2. **Text-Based Logos**
   - Achieves 99%+ SSIM
   - Preserves letter shapes perfectly

3. **Complex Logos** (gradients, photos)
   - Achieves 85-95% SSIM
   - May need manual optimization

## API Endpoints

When running the web server:

- `GET /health` - Check if service is running
- `POST /api/convert` - Convert an image
- `POST /api/upload` - Upload image first, then convert
- `GET /api/ai-health` - Check AI component status

## The AI Enhancement (Optional)

The project includes AI modules that can:
- Classify logo types automatically
- Predict optimal parameters
- Route to best conversion strategy

These are optional - the core conversion works without them.

## File Structure

```
backend/
  converter.py           # Main conversion logic
  ai_modules/
    quality.py          # SSIM/MSE/PSNR calculation
    classification.py   # Logo type detection
  converters/
    vtracer_converter.py # VTracer integration
```

## Try It Yourself

1. **Quick Test:**
```bash
python -c "from backend.converter import convert_image; print(convert_image('data/logos/simple_geometric/circle_00.png')['ssim'])"
```

2. **Start Web Server:**
```bash
python -m backend.app
# Opens on http://localhost:8001
```

3. **Convert via API:**
```bash
curl -X POST http://localhost:8001/api/convert \
  -F "file=@your_logo.png"
```

## Common Use Cases

- **Web Development**: Convert logos for responsive websites
- **App Development**: Generate icons for multiple resolutions
- **Print Design**: Create scalable graphics for any size
- **Performance**: Reduce bandwidth with smaller SVG files
- **Accessibility**: SVGs can include semantic information

## Technical Details

The magic happens in the VTracer algorithm which:
1. Applies edge detection
2. Traces contours
3. Fits Bézier curves to paths
4. Optimizes control points
5. Groups into color layers

This creates a mathematical description of your image that scales perfectly!