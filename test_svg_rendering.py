#!/usr/bin/env python3
"""Test SVG rendering to diagnose quality issues."""

import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO

def analyze_image(array, name):
    """Analyze image array properties."""
    print(f"\n{name}:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")

    if len(array.shape) == 3:
        for i, channel in enumerate(['R', 'G', 'B', 'A']):
            if i < array.shape[2]:
                ch = array[:,:,i]
                non_zero = np.count_nonzero(ch)
                print(f"  {channel}: [{ch.min()}, {ch.max()}], non-zero pixels: {non_zero}/{ch.size}")

    # Check transparency
    if len(array.shape) == 3 and array.shape[2] == 4:
        alpha = array[:,:,3]
        transparent = np.sum(alpha == 0)
        opaque = np.sum(alpha == 255)
        partial = array.size // 4 - transparent - opaque
        print(f"  Transparency: {transparent} transparent, {opaque} opaque, {partial} partial")

def test_rendering():
    """Test SVG rendering with different methods."""

    png_path = "data/logos/text_based/text_data_02.png"
    svg_path = "data/logos/text_based/text_data_02.baseline.svg"

    # Load original PNG
    png_img = Image.open(png_path).convert('RGBA')
    png_array = np.array(png_img)
    analyze_image(png_array, "Original PNG")

    # Method 1: Default cairosvg rendering
    print("\n--- Method 1: Default CairoSVG ---")
    png_bytes = cairosvg.svg2png(url=svg_path)
    svg_img1 = Image.open(BytesIO(png_bytes)).convert('RGBA')
    svg_array1 = np.array(svg_img1)
    analyze_image(svg_array1, "SVG (default size)")

    # Method 2: With explicit size
    print("\n--- Method 2: With explicit size ---")
    height, width = png_array.shape[:2]
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=width, output_height=height)
    svg_img2 = Image.open(BytesIO(png_bytes)).convert('RGBA')
    svg_array2 = np.array(svg_img2)
    analyze_image(svg_array2, f"SVG ({width}x{height})")

    # Method 3: Composite on white background
    print("\n--- Method 3: Composite on white ---")
    white_bg = Image.new('RGBA', svg_img2.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, svg_img2)
    composite_array = np.array(composite)
    analyze_image(composite_array, "SVG on white background")

    # Compare pixel statistics
    print("\n--- Pixel Comparison ---")
    png_gray = np.mean(png_array[:,:,:3], axis=2)
    svg_gray = np.mean(svg_array2[:,:,:3], axis=2)
    comp_gray = np.mean(composite_array[:,:,:3], axis=2)

    print(f"PNG mean brightness: {png_gray.mean():.1f}")
    print(f"SVG mean brightness: {svg_gray.mean():.1f}")
    print(f"Composite mean brightness: {comp_gray.mean():.1f}")

    # Simple difference
    diff = np.abs(png_gray - comp_gray).mean()
    print(f"\nAverage pixel difference (PNG vs Composite): {diff:.1f}")

if __name__ == "__main__":
    test_rendering()