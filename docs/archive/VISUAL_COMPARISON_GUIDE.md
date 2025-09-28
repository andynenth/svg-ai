# Visual Parameter Effects Guide

A guide to understanding how different parameter settings affect the visual output of SVG conversions.

## How to Use This Guide

1. **Use the Web Interface**: Start the server (`python backend/app.py`) and open http://localhost:8001
2. **Upload a Test Image**: Use the drag-and-drop interface
3. **Experiment with Settings**: Follow the comparisons below using the real-time preview
4. **Compare Results**: The interface shows original vs converted side-by-side

---

## Alpha-aware Converter Visual Effects

### Threshold Parameter (0-255)

#### Low Threshold (32-64): "Soft Edges"
**Visual Effect**: Captures subtle transparency and semi-opaque areas
- **What You'll See**: Smoother edges, preserved gradients, more detail in transparency
- **Best For**: Icons with soft shadows, anti-aliased edges, gradient transparency
- **Example**: App icons with drop shadows retain their soft appearance

#### Medium Threshold (128): "Balanced"
**Visual Effect**: Standard transparency cutoff
- **What You'll See**: Clean separation between opaque and transparent areas
- **Best For**: Most icons and logos with clear transparency
- **Example**: Simple UI icons convert cleanly without artifacts

#### High Threshold (200-255): "Hard Edges"
**Visual Effect**: Only fully opaque areas are preserved
- **What You'll See**: Very sharp edges, loss of anti-aliasing, smaller file size
- **Best For**: Icons that need crisp, pixel-perfect edges
- **Example**: Pixel art icons maintain their sharp, digital appearance

### Clean Edges Setting

#### Clean Edges ON (use_potrace: true)
**Visual Effect**: Sharp, scalable vector edges
- **What You'll See**: Smooth curves, perfect scalability, professional appearance
- **File Size**: Smaller, more efficient paths
- **Example**: Corporate logos look crisp at any size

#### Clean Edges OFF (use_potrace: false)
**Visual Effect**: Preserves original pixel-level detail
- **What You'll See**: More faithful to original, maintains texture
- **File Size**: Larger, more complex paths
- **Example**: Artistic icons retain their hand-crafted feel

---

## Potrace Converter Visual Effects

### Threshold Parameter (0-255)

#### Low Threshold (64-100): "More Black Areas"
**Visual Effect**: More pixels become black in the final SVG
- **What You'll See**: Thicker lines, filled areas, bold appearance
- **Best For**: Thin text that needs to be more readable
- **Example**: Handwritten text becomes bolder and more legible

#### Medium Threshold (128): "Balanced"
**Visual Effect**: Standard balance between black and white
- **What You'll See**: Faithful reproduction of original contrast
- **Best For**: Most logos and clean line art
- **Example**: Logo maintains original proportions and weight

#### High Threshold (180-220): "More White Areas"
**Visual Effect**: Only very dark pixels become black
- **What You'll See**: Thinner lines, more white space, delicate appearance
- **Best For**: Fine line art, detailed sketches
- **Example**: Pencil sketches maintain their light, sketchy quality

### Corner Style (turnpolicy)

#### Black Corners (turnpolicy: "black")
**Visual Effect**: Sharp, angular corners at path junctions
- **What You'll See**: Crisp angles, geometric appearance, technical look
- **Best For**: Architectural drawings, technical diagrams, pixel art
- **Example**: A square logo maintains perfectly sharp 90-degree corners

#### White Corners (turnpolicy: "white")
**Visual Effect**: Smooth, rounded corners at path junctions
- **What You'll See**: Flowing curves, organic appearance, polished look
- **Best For**: Brand logos, artistic content, professional materials
- **Example**: A circular logo has perfectly smooth curves

### Smoothness (alphamax: 0.0-1.34)

#### Low Smoothness (0.5-0.8): "Angular Paths"
**Visual Effect**: More straight line segments, fewer curves
- **What You'll See**: Geometric appearance, smaller file size, digital look
- **Best For**: Technical drawings, pixel art, geometric designs
- **Example**: A curved logo appears more faceted and geometric

#### Medium Smoothness (1.0): "Balanced"
**Visual Effect**: Good balance of curves and file size
- **What You'll See**: Natural-looking curves that scale well
- **Best For**: Most logos and graphics
- **Example**: Typography looks natural and readable

#### High Smoothness (1.2-1.34): "Maximum Curves"
**Visual Effect**: Maximum curve smoothness, very organic appearance
- **What You'll See**: Flowing, artistic curves, larger file size
- **Best For**: Artistic content, handwritten text, organic shapes
- **Example**: A hand-drawn logo maintains its flowing, artistic quality

### Noise Removal (turdsize: 0-100)

#### No Noise Removal (0-2): "Preserve All Detail"
**Visual Effect**: All small elements are preserved
- **What You'll See**: Texture, artifacts, imperfections maintained
- **Best For**: Textured artwork, intentional grain effects
- **Example**: A distressed logo keeps its weathered appearance

#### Medium Noise Removal (5-10): "Clean but Detailed"
**Visual Effect**: Small artifacts removed, detail preserved
- **What You'll See**: Clean appearance while keeping important small elements
- **Best For**: Most professional logos
- **Example**: Scanning artifacts removed but design details remain

#### High Noise Removal (20-100): "Ultra Clean"
**Visual Effect**: Only large, significant shapes remain
- **What You'll See**: Very simplified, clean appearance
- **Best For**: Logos that need maximum simplification
- **Example**: A complex logo becomes simplified and iconic

---

## VTracer Converter Visual Effects

### Color Precision (1-10)

#### Low Precision (1-3): "Simplified Colors"
**Visual Effect**: Very few distinct colors, poster-like appearance
- **What You'll See**: Bold, simplified color areas, graphic design look
- **File Size**: Smaller, fewer color layers
- **Best For**: Simple graphics, poster designs, bold logos
- **Example**: A gradient becomes 2-3 distinct color bands

#### Medium Precision (4-6): "Balanced Colors"
**Visual Effect**: Good color reproduction with manageable complexity
- **What You'll See**: Natural color representation, good detail
- **File Size**: Moderate, good balance
- **Best For**: Most colorful logos and graphics
- **Example**: A logo maintains its color scheme without oversimplification

#### High Precision (7-10): "Maximum Colors"
**Visual Effect**: Very detailed color reproduction
- **What You'll See**: Subtle color variations, smooth gradients
- **File Size**: Larger, more complex
- **Best For**: Complex illustrations, photographic content
- **Example**: A detailed illustration maintains subtle color nuances

### Layer Difference (0-256)

#### Small Difference (4-12): "Smooth Gradients"
**Visual Effect**: Colors that are close together are treated as gradients
- **What You'll See**: Smooth color transitions, many color layers
- **Best For**: Images with gradients, subtle shading
- **Example**: A logo with a gradient maintains smooth color flow

#### Medium Difference (16-32): "Distinct Colors"
**Visual Effect**: Clear separation between different color areas
- **What You'll See**: Well-defined color regions, clean appearance
- **Best For**: Most logos with distinct colors
- **Example**: A multi-color logo has clear, separate color areas

#### Large Difference (64-128): "Bold Separation"
**Visual Effect**: Only very different colors are treated as separate
- **What You'll See**: High contrast, poster-like effect
- **Best For**: Images that need bold, simplified appearance
- **Example**: A colorful illustration becomes a bold, graphic design

### Corner Threshold (0-180°)

#### Sharp Corners (15-45°): "Angular Appearance"
**Visual Effect**: Many corners are preserved as sharp angles
- **What You'll See**: Geometric, precise appearance
- **Best For**: Technical drawings, logos with sharp design elements
- **Example**: A star logo maintains all its sharp points

#### Medium Corners (45-90°): "Balanced"
**Visual Effect**: Natural balance of curves and corners
- **What You'll See**: Realistic representation of original shapes
- **Best For**: Most graphics and illustrations
- **Example**: A logo looks natural and proportional

#### Smooth Corners (120-180°): "Flowing Curves"
**Visual Effect**: Most angles become smooth curves
- **What You'll See**: Organic, flowing appearance
- **Best For**: Artistic content, natural objects
- **Example**: A geometric logo becomes more organic and flowing

---

## Comparative Testing Workflow

### Step 1: Upload Test Image
Use these sample image types for testing:
- **Simple Icon**: Basic shape with transparency (test Alpha-aware)
- **Text Logo**: Black text on white background (test Potrace)
- **Colorful Logo**: Multi-color brand logo (test VTracer)

### Step 2: Baseline Conversion
Start with "Quality" presets for each converter to establish baseline quality.

### Step 3: Parameter Experimentation
Test one parameter at a time to see individual effects:

#### For Alpha-aware:
1. Try threshold: 32, 128, 200 (observe edge treatment)
2. Toggle Clean Edges (observe sharpness vs. fidelity)
3. Toggle Anti-aliasing (observe smoothness vs. file size)

#### For Potrace:
1. Try threshold: 80, 128, 180 (observe line weight)
2. Try corners: black vs. white (observe sharpness vs. smoothness)
3. Try smoothness: 0.8, 1.0, 1.3 (observe curve quality)

#### For VTracer:
1. Try colors: 3, 6, 9 (observe color complexity)
2. Try layer difference: 8, 16, 32 (observe gradient handling)
3. Try corners: 30°, 60°, 120° (observe shape character)

### Step 4: Quality Assessment
For each test, check:
- **SSIM Score**: Higher is better (aim for >90%)
- **File Size**: Balance quality vs. efficiency
- **Visual Appeal**: Does it match your intended use?
- **Scalability**: Zoom in to check curve quality

---

## Common Visual Issues and Solutions

### Issue: Edges Look Jagged or Pixelated
**What You See**: Stair-step appearance, rough curves
**Solutions**:
- Increase smoothness (Potrace: higher alphamax)
- Use white corners instead of black (Potrace)
- Increase path precision (VTracer)
- Try Alpha-aware with clean edges

### Issue: Colors Are Wrong or Missing
**What You See**: Different colors than original, lost color details
**Solutions**:
- Increase color precision (VTracer)
- Decrease layer difference (VTracer)
- Check if image has transparency (use Alpha-aware)
- Ensure using color mode, not binary (VTracer)

### Issue: Too Much Detail/File Too Large
**What You See**: Overly complex SVG, large file size
**Solutions**:
- Reduce color precision (VTracer)
- Increase layer difference (VTracer)
- Increase noise removal (Potrace)
- Use higher thresholds to simplify

### Issue: Lost Important Details
**What You See**: Missing fine elements, over-simplified
**Solutions**:
- Lower threshold values
- Reduce noise removal (Potrace)
- Lower corner threshold (VTracer)
- Increase color precision (VTracer)

### Issue: File Won't Scale Well
**What You See**: Looks good at one size, poor at others
**Solutions**:
- Use Potrace for maximum scalability
- Enable clean edges (Alpha-aware)
- Increase smoothness for better curves
- Avoid excessive detail that doesn't scale

---

## Advanced Visual Testing

### A/B Testing Different Converters
For the same image, try:
1. **Alpha-aware** with quality preset
2. **Potrace** with quality preset
3. **VTracer** with quality preset

Compare:
- Visual fidelity
- File size
- Scalability
- Conversion time

### Optimization Workflow
1. **Start with the best converter** for your image type
2. **Use quality preset** as baseline
3. **Optimize for your specific needs**:
   - Smaller file size: reduce precision/colors
   - Better quality: increase precision/smoothness
   - Faster conversion: use fast presets

### Real-World Testing
Test your SVGs in actual use cases:
- **Web browsers**: Check how they render online
- **Print**: Ensure they scale for print use
- **Mobile**: Test on small screens
- **Vector editors**: Import into Illustrator/Inkscape

---

*Use the web interface at http://localhost:8001 to see these effects in real-time. The side-by-side comparison makes it easy to understand how each parameter affects the final output.*