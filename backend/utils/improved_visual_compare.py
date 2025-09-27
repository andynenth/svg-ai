#!/usr/bin/env python3
"""
Improved visual comparison that properly detects shape differences.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
import io
from pathlib import Path
from typing import Tuple, Dict, Optional

try:
    import cairosvg
except ImportError:
    print("Warning: cairosvg not installed. SVG comparison will not work.")
    cairosvg = None


class ImprovedVisualComparer:
    """Improved visual comparison with better difference detection."""

    def __init__(self, comparison_size: Tuple[int, int] = (512, 512)):
        self.comparison_size = comparison_size

    def svg_to_png(self, svg_path_or_content: str, output_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Convert SVG to PNG for comparison."""
        size = output_size or self.comparison_size

        # Check if it's a file path or SVG content
        if svg_path_or_content.startswith('<?xml') or svg_path_or_content.startswith('<svg'):
            # It's SVG content
            png_data = cairosvg.svg2png(
                bytestring=svg_path_or_content.encode('utf-8'),
                output_width=size[0],
                output_height=size[1]
            )
        else:
            # It's a file path
            png_data = cairosvg.svg2png(
                url=svg_path_or_content,
                output_width=size[0],
                output_height=size[1]
            )

        return Image.open(io.BytesIO(png_data))

    def calculate_shape_difference(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """
        Calculate differences focusing on shape coverage, not just pixel values.

        This method detects:
        - Different stroke widths (same color, different coverage)
        - Shape alignment differences
        - Anti-aliasing differences
        """
        # Handle images with alpha channel properly
        # Use alpha channel as the shape if RGB is all black
        if len(img1.shape) == 3 and img1.shape[2] >= 4:
            # Check if RGB channels are all black/same
            rgb1 = img1[:, :, :3]
            if np.std(rgb1) < 10:  # Very low variation in RGB
                # Use alpha channel as the shape
                gray1 = 255 - img1[:, :, 3]  # Invert alpha so shape is dark
            else:
                gray1 = np.mean(rgb1, axis=2)
        elif len(img1.shape) == 3:
            gray1 = np.mean(img1[:, :, :3], axis=2)
        else:
            gray1 = img1

        if len(img2.shape) == 3 and img2.shape[2] >= 4:
            rgb2 = img2[:, :, :3]
            if np.std(rgb2) < 10:
                gray2 = 255 - img2[:, :, 3]
            else:
                gray2 = np.mean(rgb2, axis=2)
        elif len(img2.shape) == 3:
            gray2 = np.mean(img2[:, :, :3], axis=2)
        else:
            gray2 = img2

        # Normalize to 0-255 range
        gray1 = gray1.astype(float)
        gray2 = gray2.astype(float)

        # Calculate absolute difference
        abs_diff = np.abs(gray1 - gray2)

        # Count pixels with different intensities
        # Even if both are black, different anti-aliasing will show up
        any_diff_pixels = np.sum(abs_diff > 1)  # Very low threshold
        small_diff_pixels = np.sum(abs_diff > 5)
        medium_diff_pixels = np.sum(abs_diff > 20)
        large_diff_pixels = np.sum(abs_diff > 50)

        total_pixels = gray1.shape[0] * gray1.shape[1]

        # Calculate coverage difference (how many pixels are "filled")
        threshold = 128  # Middle gray
        coverage1 = np.sum(gray1 < threshold)
        coverage2 = np.sum(gray2 < threshold)
        coverage_diff = abs(coverage1 - coverage2)
        coverage_diff_percent = (coverage_diff / total_pixels) * 100

        # Edge detection for shape boundary differences
        from scipy import ndimage
        edges1 = ndimage.sobel(gray1)
        edges2 = ndimage.sobel(gray2)
        edge_diff = np.abs(edges1 - edges2)
        edge_diff_pixels = np.sum(edge_diff > 10)

        return {
            'any_diff_pixels': int(any_diff_pixels),
            'small_diff_pixels': int(small_diff_pixels),
            'medium_diff_pixels': int(medium_diff_pixels),
            'large_diff_pixels': int(large_diff_pixels),
            'total_pixels': int(total_pixels),
            'diff_percentage': (any_diff_pixels / total_pixels) * 100,
            'coverage_diff_percent': coverage_diff_percent,
            'edge_diff_pixels': int(edge_diff_pixels),
            'edge_diff_percent': (edge_diff_pixels / total_pixels) * 100,
            'max_diff': float(np.max(abs_diff)),
            'mean_diff': float(np.mean(abs_diff)),
            'coverage1_pixels': int(coverage1),
            'coverage2_pixels': int(coverage2)
        }

    def create_enhanced_diff_image(self, original_path: str, svg_content: str) -> Tuple[Image.Image, Dict]:
        """Create an enhanced difference visualization."""
        # Load and resize original
        original = Image.open(original_path).convert('RGBA')
        original = original.resize(self.comparison_size, Image.Resampling.LANCZOS)

        # Convert SVG to PNG
        converted = self.svg_to_png(svg_content)
        if converted.mode != 'RGBA':
            converted = converted.convert('RGBA')

        # Convert to arrays
        orig_array = np.array(original)
        conv_array = np.array(converted)

        # Calculate shape-aware differences
        metrics = self.calculate_shape_difference(orig_array, conv_array)

        # Create diff visualization
        diff_vis = Image.new('RGBA', self.comparison_size, (255, 255, 255, 255))
        diff_array = np.zeros((*self.comparison_size[::-1], 4), dtype=np.uint8)
        diff_array[:, :, 3] = 255  # Full opacity

        # Calculate per-pixel differences (handle alpha channel)
        # Check if we should use alpha channel
        if orig_array.shape[2] >= 4 and np.std(orig_array[:, :, :3]) < 10:
            gray1 = 255 - orig_array[:, :, 3]  # Use inverted alpha
        else:
            gray1 = np.mean(orig_array[:, :, :3], axis=2)

        if conv_array.shape[2] >= 4 and np.std(conv_array[:, :, :3]) < 10:
            gray2 = 255 - conv_array[:, :, 3]  # Use inverted alpha
        else:
            gray2 = np.mean(conv_array[:, :, :3], axis=2)

        abs_diff = np.abs(gray1 - gray2)

        # Color code the differences
        for y in range(abs_diff.shape[0]):
            for x in range(abs_diff.shape[1]):
                diff_val = abs_diff[y, x]

                if diff_val > 1:  # Any difference
                    if diff_val < 5:
                        # Very small - light blue
                        diff_array[y, x] = [200, 200, 255, 100]
                    elif diff_val < 20:
                        # Small - green
                        diff_array[y, x] = [0, 255, 0, 150]
                    elif diff_val < 50:
                        # Medium - yellow
                        diff_array[y, x] = [255, 255, 0, 200]
                    else:
                        # Large - red
                        diff_array[y, x] = [255, 0, 0, 255]

        diff_vis = Image.fromarray(diff_array, 'RGBA')
        return diff_vis, metrics

    def create_comparison_grid(self, original_path: str, svg_content: str) -> Image.Image:
        """Create a comparison grid with accurate difference reporting."""
        # Load images
        original = Image.open(original_path).convert('RGBA')
        original = original.resize(self.comparison_size, Image.Resampling.LANCZOS)

        converted = self.svg_to_png(svg_content)
        if converted.mode != 'RGBA':
            converted = converted.convert('RGBA')

        # Get difference visualization and metrics
        diff_vis, metrics = self.create_enhanced_diff_image(original_path, svg_content)

        # Create grid
        grid_width = self.comparison_size[0] * 3 + 40
        grid_height = self.comparison_size[1] + 100

        grid = Image.new('RGBA', (grid_width, grid_height), (240, 240, 240, 255))
        draw = ImageDraw.Draw(grid)

        # Paste images
        grid.paste(original, (10, 50))
        grid.paste(converted, (self.comparison_size[0] + 20, 50))
        grid.paste(diff_vis, (self.comparison_size[0] * 2 + 30, 50))

        # Add labels
        try:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), "Original PNG", fill=(0, 0, 0), font=font)
            draw.text((self.comparison_size[0] + 20, 10), "Converted SVG", fill=(0, 0, 0), font=font)

            # More informative difference label
            diff_text = f"Difference ({metrics['diff_percentage']:.1f}%)"
            if metrics['coverage_diff_percent'] > 0.1:
                diff_text += f" Coverage: {metrics['coverage_diff_percent']:.1f}%"

            draw.text((self.comparison_size[0] * 2 + 30, 10), diff_text, fill=(0, 0, 0), font=font)

            # Add metrics info at bottom
            info_y = self.comparison_size[1] + 60
            info_text = f"Shape pixels diff: {metrics['any_diff_pixels']:,} | Edge diff: {metrics['edge_diff_percent']:.1f}% | Mean diff: {metrics['mean_diff']:.1f}"
            draw.text((10, info_y), info_text, fill=(50, 50, 50), font=font)

        except Exception as e:
            print(f"Font error: {e}")

        return grid


def test_improved_comparison():
    """Test the improved comparison on the heart image."""
    print("Testing improved comparison...")

    comparer = ImprovedVisualComparer()

    # Load heart files
    png_path = Path("data/images/heart.png")
    svg_path = Path("data/images/heart.optimized.svg")

    if not png_path.exists() or not svg_path.exists():
        print("Heart files not found")
        return

    with open(svg_path, 'r') as f:
        svg_content = f.read()

    # Create comparison
    comparison = comparer.create_comparison_grid(str(png_path), svg_content)
    comparison.save("heart_improved_comparison.png")

    # Get detailed metrics
    _, metrics = comparer.create_enhanced_diff_image(str(png_path), svg_content)

    print("\nDetailed metrics:")
    print(f"  Total pixels: {metrics['total_pixels']:,}")
    print(f"  Pixels with any difference: {metrics['any_diff_pixels']:,} ({metrics['diff_percentage']:.2f}%)")
    print(f"  Coverage difference: {metrics['coverage_diff_percent']:.2f}%")
    print(f"  Edge difference: {metrics['edge_diff_percent']:.2f}%")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.0f}")
    print(f"  Original coverage: {metrics['coverage1_pixels']:,} pixels")
    print(f"  Converted coverage: {metrics['coverage2_pixels']:,} pixels")

    print(f"\n✅ Saved improved comparison to heart_improved_comparison.png")


if __name__ == "__main__":
    test_improved_comparison()