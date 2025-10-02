#!/usr/bin/env python3
"""Quick visual comparison of a single logo conversion"""

import sys
from pathlib import Path
from backend.converter import convert_image

if len(sys.argv) < 2:
    # Use a random logo
    import random
    logos = list(Path("data/raw_logos").glob("*.png"))
    if logos:
        logo_path = random.choice(logos)
        print(f"Using random logo: {logo_path.name}")
    else:
        print("No logos found in data/raw_logos/")
        sys.exit(1)
else:
    logo_path = Path(sys.argv[1])

# Convert with AI-optimized parameters
result = convert_image(str(logo_path),
                      color_precision=6,
                      corner_threshold=60)

# Save SVG
svg_file = f"{logo_path.stem}_converted.svg"
with open(svg_file, 'w') as f:
    f.write(result['svg'])

# Create HTML comparison
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Conversion Result: {logo_path.name}</title>
    <style>
        body {{ font-family: Arial; background: #f0f0f0; padding: 20px; }}
        .container {{ display: flex; gap: 20px; justify-content: center; }}
        .panel {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        img {{ max-width: 400px; height: auto; border: 1px solid #ddd; }}
        h2 {{ color: #333; margin-top: 0; }}
        .metrics {{ background: #f9f9f9; padding: 10px; border-radius: 4px; margin-top: 10px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 5px 0; }}
        .score {{ font-weight: bold; color: #4CAF50; }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">🔄 PNG to SVG Conversion Result</h1>
    <div class="container">
        <div class="panel">
            <h2>Original PNG</h2>
            <img src="{logo_path}" alt="Original">
            <div class="metrics">
                <div class="metric"><span>Format:</span> <span>PNG</span></div>
                <div class="metric"><span>File:</span> <span>{logo_path.name}</span></div>
            </div>
        </div>
        <div class="panel">
            <h2>Converted SVG</h2>
            <img src="{svg_file}" alt="Converted">
            <div class="metrics">
                <div class="metric"><span>Quality (SSIM):</span> <span class="score">{result['ssim']:.3f}</span></div>
                <div class="metric"><span>Size:</span> <span>{len(result['svg'])} bytes</span></div>
                <div class="metric"><span>Format:</span> <span>SVG (scalable)</span></div>
            </div>
        </div>
    </div>
    <div style="text-align: center; margin-top: 30px; background: white; padding: 20px; border-radius: 8px;">
        <h3>Conversion Metrics</h3>
        <p>✅ Success: {result.get('success', False)}</p>
        <p>📊 SSIM Score: {result.get('ssim', 0):.3f} (1.0 = perfect)</p>
        <p>📏 MSE: {result.get('mse', 0):.1f}</p>
        <p>🎯 PSNR: {result.get('psnr', 0):.1f} dB</p>
    </div>
</body>
</html>
"""

# Save comparison
comparison_file = "comparison.html"
with open(comparison_file, 'w') as f:
    f.write(html)

print(f"\n✅ Conversion complete!")
print(f"📊 SSIM Score: {result['ssim']:.3f}")
print(f"📁 SVG saved to: {svg_file}")
print(f"🌐 View comparison: file://{Path(comparison_file).absolute()}")
print(f"\nOpen in browser: open {comparison_file}")