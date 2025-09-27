# SVG AI Converter - PNG to SVG Conversion Tool

A comprehensive PNG to SVG conversion tool with advanced parameter controls and multiple conversion algorithms.

## Features

- ✅ **Three Specialized Converters**: Potrace (B&W), VTracer (Color), Alpha-aware (Icons)
- ✅ **Web Interface**: Interactive parameter controls with real-time preview
- ✅ **Dynamic Parameter System**: Converter-specific controls with tooltips and presets
- ✅ **Quality Metrics**: SSIM scoring and file size optimization
- ✅ **Parameter Validation**: Comprehensive input validation and error handling
- ✅ **Preset System**: Quality, Fast, and Reset presets for optimal settings
- ✅ **Responsive Design**: Mobile-friendly interface with touch controls

## Quick Start

### Installation

1. **Clone the repository:**
```bash
cd /Users/nrw/python/svg-ai
```

2. **Set up Python virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install pillow numpy click requests vtracer
```

### Web Interface (Recommended)

1. **Start the web server:**
```bash
cd backend
python app.py
```

2. **Open your browser:**
```
http://localhost:8001
```

3. **Convert images:**
   - Drag & drop PNG/JPEG files
   - Select converter: Alpha-aware (icons), Potrace (B&W), or VTracer (color)
   - Adjust parameters using intuitive controls
   - Download SVG results

### CLI Usage (Advanced)

1. **Test the installation:**
```bash
python test_vtracer.py
```

2. **Convert with CLI tool:**
```bash
python convert.py image.png --optimize-logo
```

## Project Structure

```
svg-ai/
├── backend/            # Flask API server
│   ├── converters/     # Conversion algorithms
│   │   ├── base.py             # Base converter class
│   │   ├── potrace_converter.py # Potrace (B&W) implementation
│   │   ├── vtracer_converter.py # VTracer (color) implementation
│   │   └── alpha_converter.py   # Alpha-aware (icons) implementation
│   ├── utils/          # Utility functions
│   │   ├── quality_metrics.py  # SSIM and quality analysis
│   │   └── cache.py            # Conversion caching
│   ├── app.py          # Flask API server
│   └── converter.py    # Main conversion logic
├── frontend/           # Web interface
│   ├── index.html      # Main web interface
│   ├── script.js       # Parameter controls and API calls
│   └── style.css       # Responsive UI styling
├── data/               # Test data
│   └── logos/          # Sample PNG files
└── uploads/            # Uploaded files storage
```

## Converters & Parameters

### Alpha-aware Converter (Best for Icons)
**Optimized for transparent PNG icons and logos with alpha channels.**

#### Parameters:
- **Alpha Level** (0-255): Minimum opacity to include. Lower = more semi-transparent areas
- **Clean Edges**: Use Potrace for sharper edges (recommended: ON)
- **Anti-aliasing**: Preserve smooth edges. Larger file but smoother (recommended: OFF for clean icons)

#### Best For:
- Icons with transparency
- Simple logos with clean edges
- PNG files with alpha channels

#### Presets:
- **Quality**: Lower threshold (64), clean edges, anti-aliasing enabled
- **Fast**: Standard threshold (128), clean edges, no anti-aliasing

### Potrace Converter (Black & White)
**Specialized for high-contrast, monochromatic images with perfect curves.**

#### Parameters:
- **Black Level** (0-255): How dark pixels must be to become black. Lower = more black areas
- **Corner Style**: How to handle ambiguous corners
  - Black (Sharp): Creates sharp, angular corners
  - White (Smooth): Creates smooth, rounded corners
- **Remove Noise** (0-100): Removes spots smaller than this many pixels
- **Smoothness** (0.0-1.34): How rounded corners should be. Higher = smoother curves
- **Accuracy** (0.01-1.0): How closely curves match original. Lower = more precise

#### Best For:
- Black and white logos
- Text and typography
- Line art and sketches
- Images needing perfect curves

#### Presets:
- **Quality**: Smooth corners, noise removal (5), high smoothness (1.2), high accuracy (0.1)
- **Fast**: Sharp corners, minimal noise removal (1), standard smoothness (1.0), standard accuracy (0.2)

### VTracer Converter (Color)
**Advanced color tracing with gradient support and path optimization.**

#### Parameters:
- **Mode**: Color or Black & White processing
- **Colors** (1-10): Number of colors to detect. Lower = simpler image
- **Color Diff** (0-256): Minimum difference between colors. Higher = fewer colors
- **Smoothness** (0-10): Path curve smoothness. Higher = smoother paths
- **Corner Angle** (0-180°): Angle that defines a corner. Higher = fewer corners
- **Min Path**: Ignore paths shorter than this length
- **Quality** (1-50): Number of refinement passes. More = better but slower
- **Join Paths** (0-180°): Angle for connecting paths. Higher = more connected

#### Best For:
- Colorful logos and graphics
- Images with gradients
- Complex illustrations
- Photographic content

#### Presets:
- **Quality**: Color mode, more colors (8), fine gradients (8), high precision (8), gentle corners (30)
- **Fast**: Color mode, fewer colors (4), basic gradients (16), lower precision (3), more corners (60)

## CLI Options

```bash
python convert.py [OPTIONS] INPUT_PATH

Options:
  -o, --output PATH        Output SVG file path
  --optimize-logo          Use logo-optimized settings
  --color-precision INT    Color precision (1-10, default: 6)
  --preprocess            Apply preprocessing to image
  -v, --verbose           Show detailed output
  --help                  Show this message and exit
```

## Examples

### Simple Conversion
```bash
python convert.py image.png
# Output: image.svg
```

### Custom Output Path
```bash
python convert.py input.png -o output.svg
```

### Logo Optimization
```bash
python convert.py logo.png --optimize-logo
```

### With Preprocessing
```bash
python convert.py complex.png --preprocess --verbose
```

## API Endpoints

The Flask backend provides REST endpoints for programmatic access:

### Upload Image
```bash
POST /api/upload
Content-Type: multipart/form-data

curl -X POST http://localhost:8001/api/upload \
  -F "file=@image.png"
```
**Response:**
```json
{
  "file_id": "abc123...",
  "filename": "image.png",
  "path": "/uploads/abc123....png"
}
```

### Convert Image
```bash
POST /api/convert
Content-Type: application/json

curl -X POST http://localhost:8001/api/convert \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123...",
    "converter": "alpha",
    "threshold": 128,
    "use_potrace": true,
    "preserve_antialiasing": false
  }'
```

**Response:**
```json
{
  "success": true,
  "svg": "<svg>...</svg>",
  "ssim": 0.95,
  "size": 1024
}
```

### Parameter Validation
All parameters are validated with specific ranges:
- **Alpha**: `threshold` (0-255), `use_potrace` (boolean), `preserve_antialiasing` (boolean)
- **Potrace**: `threshold` (0-255), `turnpolicy` (black/white/left/right/minority/majority), `turdsize` (0-100), `alphamax` (0-1.34), `opttolerance` (0.01-1.0)
- **VTracer**: `colormode` (color/binary), `color_precision` (1-10), `layer_difference` (0-256), `path_precision` (0-10), `corner_threshold` (0-180), `length_threshold` (0-100), `max_iterations` (1-50), `splice_threshold` (0-180)

## Performance

- **Simple logos**: 0.5-1s conversion time, 90%+ quality
- **Complex images**: 1-2s conversion time, 70-85% quality
- **File size**: Typically 50-80% reduction from PNG

## Requirements

- Python 3.8+
- macOS, Linux, or Windows
- 4GB RAM minimum
- No GPU required (CPU-based conversion)

## Development Roadmap

### Phase 1: Foundation ✅ COMPLETED
- ✅ Basic PNG to SVG conversion
- ✅ CLI tool
- ✅ Quality metrics (SSIM scoring)
- ✅ Test dataset creation
- ✅ Benchmark system

### Phase 2: Advanced Features ✅ COMPLETED
- ✅ Multiple converter support (Potrace, VTracer, Alpha-aware)
- ✅ Dynamic parameter system with validation
- ✅ Caching system for conversions
- ✅ Web interface with responsive design
- ✅ REST API endpoints

### Phase 3: Future Enhancements
- [ ] OmniSVG integration for AI-powered conversion
- [ ] Batch processing interface
- [ ] Cloud deployment and scaling
- [ ] Advanced parameter optimization
- [ ] Visual difference highlighting

## Troubleshooting

### VTracer Installation Issues

If `pip install vtracer` fails, try:

```bash
# On macOS
brew install rust
pip install --no-binary :all: vtracer

# Or build from source
git clone https://github.com/visioncortex/vtracer
cd vtracer
cargo build --release
pip install .
```

### Low Quality Results

Adjust parameters:
```bash
python convert.py image.png --color-precision 8
```

### Memory Issues

Process smaller batches or resize images:
```bash
# Preprocessing will resize to 512x512 max
python convert.py large_image.png --preprocess
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - feel free to use for personal or commercial projects.

## Acknowledgments

- VTracer by VisionCortex for the core tracing engine
- Pillow for image processing
- Click for CLI framework