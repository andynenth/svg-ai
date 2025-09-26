#!/usr/bin/env python3
"""Debug SSIM calculation to find the issue."""

import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO

def debug_ssim():
    """Debug SSIM calculation step by step."""

    # Load images
    png_path = "data/logos/text_based/text_data_02.png"
    svg_path = "data/logos/text_based/text_data_02.baseline.svg"

    print("Loading images...")
    # Load PNG
    png_img = Image.open(png_path).convert('RGBA')
    png_array = np.array(png_img)
    print(f"PNG shape: {png_array.shape}, dtype: {png_array.dtype}")
    print(f"PNG value range: [{png_array.min()}, {png_array.max()}]")

    # Load SVG
    height, width = png_array.shape[:2]
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=width, output_height=height)
    svg_img = Image.open(BytesIO(png_bytes)).convert('RGBA')
    svg_array = np.array(svg_img)
    print(f"SVG shape: {svg_array.shape}, dtype: {svg_array.dtype}")
    print(f"SVG value range: [{svg_array.min()}, {svg_array.max()}]")

    # Check if shapes match
    print(f"\nShapes match: {png_array.shape == svg_array.shape}")

    # Convert to grayscale for SSIM
    png_gray = np.mean(png_array[:,:,:3], axis=2)  # Ignore alpha
    svg_gray = np.mean(svg_array[:,:,:3], axis=2)
    print(f"\nGrayscale PNG shape: {png_gray.shape}")
    print(f"Grayscale SVG shape: {svg_gray.shape}")

    # Normalize
    png_norm = png_gray.astype(np.float64) / 255.0
    svg_norm = svg_gray.astype(np.float64) / 255.0
    print(f"\nNormalized PNG range: [{png_norm.min():.3f}, {png_norm.max():.3f}]")
    print(f"Normalized SVG range: [{svg_norm.min():.3f}, {svg_norm.max():.3f}]")

    # Simple SSIM calculation
    k1, k2 = 0.01, 0.03
    C1 = (k1 * 1) ** 2
    C2 = (k2 * 1) ** 2

    mu1 = np.mean(png_norm)
    mu2 = np.mean(svg_norm)

    sigma1_sq = np.var(png_norm)
    sigma2_sq = np.var(svg_norm)
    sigma12 = np.mean((png_norm - mu1) * (svg_norm - mu2))

    print(f"\nStatistics:")
    print(f"mu1: {mu1:.4f}, mu2: {mu2:.4f}")
    print(f"sigma1_sq: {sigma1_sq:.4f}, sigma2_sq: {sigma2_sq:.4f}")
    print(f"sigma12: {sigma12:.4f}")

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    print(f"\nSSIM components:")
    print(f"numerator: {numerator:.6f}")
    print(f"denominator: {denominator:.6f}")

    ssim = numerator / denominator
    print(f"\nSimple SSIM: {ssim:.4f}")

    # Check for issues
    if ssim < 0:
        print("\n⚠️ SSIM is negative! This suggests the images are very different.")
        print("Possible reasons:")
        print("- Images have inverted colors")
        print("- Covariance (sigma12) is negative")
        print("- Structural differences are significant")

if __name__ == "__main__":
    debug_ssim()