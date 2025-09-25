#!/usr/bin/env python3
"""Test VTracer installation and basic functionality."""

import sys
import numpy as np
from pathlib import Path

print("Testing VTracer installation...")
print("-" * 40)

try:
    import vtracer
    print("✓ VTracer imported successfully")
    print(f"  Version info available: {hasattr(vtracer, '__version__')}")
except ImportError as e:
    print(f"✗ Failed to import VTracer: {e}")
    print("\nPlease install VTracer with:")
    print("  pip install vtracer")
    sys.exit(1)

# Test with a simple array
print("\n📝 Testing with simple array...")
try:
    # Create a simple 4x4 black and white image
    test_array = [
        [0, 0, 0, 0],
        [0, 255, 255, 0],
        [0, 255, 255, 0],
        [0, 0, 0, 0]
    ]

    svg_result = vtracer.convert_pixels_to_svg(
        test_array,
        colormode="binary"
    )

    print(f"✓ Array conversion successful!")
    print(f"  SVG length: {len(svg_result)} characters")
    print(f"  Contains SVG tag: {'<svg' in svg_result}")

except Exception as e:
    print(f"✗ Array conversion failed: {e}")

# Create a test PNG if it doesn't exist
test_png_path = Path("data/logos/test_square.png")
test_png_path.parent.mkdir(parents=True, exist_ok=True)

if not test_png_path.exists():
    print("\n📝 Creating test PNG...")
    try:
        from PIL import Image

        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        pixels = img.load()

        # Draw a red square in the center
        for i in range(30, 70):
            for j in range(30, 70):
                pixels[i, j] = (255, 0, 0)

        img.save(test_png_path)
        print(f"✓ Created test image: {test_png_path}")

    except Exception as e:
        print(f"✗ Failed to create test image: {e}")

# Test with actual PNG file
if test_png_path.exists():
    print("\n📝 Testing with PNG file...")
    try:
        svg_result = vtracer.convert_image_to_svg_py(
            str(test_png_path),
            colormode="color",
            color_precision=6
        )

        output_path = test_png_path.with_suffix('.svg')
        with open(output_path, 'w') as f:
            f.write(svg_result)

        print(f"✓ PNG conversion successful!")
        print(f"  Input: {test_png_path}")
        print(f"  Output: {output_path}")
        print(f"  SVG size: {len(svg_result)} bytes")

        # Show first 200 chars of SVG
        print(f"\n  SVG preview:")
        print(f"  {svg_result[:200]}...")

    except Exception as e:
        print(f"✗ PNG conversion failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 40)
print("✅ VTracer is working correctly!" if 'svg_result' in locals() else "⚠️ VTracer needs troubleshooting")
print("=" * 40)