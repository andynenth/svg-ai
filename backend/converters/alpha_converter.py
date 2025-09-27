#!/usr/bin/env python3
"""
Alpha-aware converter for icons with transparency.

This converter is optimized for images where the shape is defined
by the alpha channel, common in modern icons.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from converters.base import BaseConverter


class AlphaConverter(BaseConverter):
    """Converter optimized for alpha-based icons."""

    def __init__(self):
        """Initialize alpha converter."""
        super().__init__()
        self.potrace_cmd = self._find_potrace()

    def get_name(self) -> str:
        """Get converter name."""
        return "Alpha Converter"

    def _find_potrace(self) -> Optional[str]:
        """Find potrace command."""
        for cmd in ['potrace', '/usr/local/bin/potrace', '/opt/homebrew/bin/potrace']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                continue
        return None

    def convert(self, image_path: str, **kwargs) -> str:
        """
        Convert alpha-based image to SVG.

        Args:
            image_path: Path to input image
            **kwargs: Additional parameters
                - threshold: Alpha threshold for shape detection (0-255)
                - use_potrace: Use potrace for cleaner output
                - preserve_antialiasing: Keep smooth edges

        Returns:
            SVG content string
        """
        img = Image.open(image_path)

        # Detect if this is an alpha-based image
        if img.mode == 'RGBA':
            arr = np.array(img)
            rgb_std = np.std(arr[:, :, :3])
            alpha_variation = np.std(arr[:, :, 3])

            if rgb_std < 10:  # RGB channels are uniform (likely all black or white)
                return self._convert_alpha_icon(img, arr, **kwargs)

        # Fall back to standard conversion for non-alpha images
        return self._convert_standard(image_path, **kwargs)

    def _convert_alpha_icon(self, img: Image.Image, arr: np.ndarray, **kwargs) -> str:
        """Convert an alpha-based icon to SVG."""
        threshold = kwargs.get('threshold', 128)
        use_potrace = kwargs.get('use_potrace', True) and self.potrace_cmd
        preserve_antialiasing = kwargs.get('preserve_antialiasing', False)

        # Extract alpha channel
        alpha = arr[:, :, 3]

        # Get the RGB color (should be uniform)
        rgb_color = arr[alpha > 128, :3].mean(axis=0) if np.any(alpha > 128) else [0, 0, 0]

        if use_potrace and not preserve_antialiasing:
            # Use potrace for perfect conversion
            return self._convert_with_potrace(alpha, rgb_color, threshold)
        else:
            # Use direct path generation for antialiased edges
            return self._convert_with_paths(alpha, rgb_color, threshold, preserve_antialiasing)

    def _convert_with_potrace(self, alpha: np.ndarray, rgb_color: np.ndarray, threshold: int) -> str:
        """Convert using potrace for perfect edges."""
        # Stage 1: Data Preservation - analyze data loss first
        inclusion_threshold = max(16, threshold // 8)  # Much lower threshold for data retention
        significant_pixels = alpha > inclusion_threshold

        # Debug info to track data retention
        total_pixels = alpha.size
        significant_count = np.sum(significant_pixels)
        quality_count = np.sum(alpha > threshold)
        data_retention_ratio = significant_count / total_pixels

        # Stage 2: Choose conversion strategy based on data loss
        if data_retention_ratio < 0.6:  # If losing >40% of data
            print(f"[Alpha] High data loss detected, using direct alpha conversion (preserving {100*data_retention_ratio:.1f}% of image)")
            binary = alpha.astype(np.uint8)  # Use alpha channel as grayscale
            binary = 255 - binary  # Invert for potrace
        else:
            # Quality Control - apply user threshold only within significant areas
            if data_retention_ratio < 0.8:
                print(f"[Alpha] WARNING: Moderate data loss detected ({100*(1-data_retention_ratio):.1f}% lost), consider lower threshold")

            binary = np.zeros_like(alpha, dtype=np.uint8)
            binary[significant_pixels] = (alpha[significant_pixels] > threshold).astype(np.uint8) * 255
            binary = 255 - binary  # Invert for potrace

        # Save as PBM for potrace
        img_binary = Image.fromarray(binary, mode='L').convert('1')

        with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp_pbm:
            img_binary.save(tmp_pbm.name)

            # Convert with potrace
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                try:
                    result = subprocess.run(
                        [self.potrace_cmd, '-s', '--flat', tmp_pbm.name, '-o', tmp_svg.name],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        with open(tmp_svg.name, 'r') as f:
                            svg_content = f.read()

                        # Replace black fill with original color
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
                        )
                        svg_content = svg_content.replace('fill="#000000"', f'fill="{hex_color}"')

                        return svg_content
                    else:
                        raise Exception(f"Potrace failed: {result.stderr}")

                finally:
                    os.unlink(tmp_pbm.name)
                    if os.path.exists(tmp_svg.name):
                        os.unlink(tmp_svg.name)

    def _convert_with_paths(self, alpha: np.ndarray, rgb_color: np.ndarray,
                           threshold: int, preserve_antialiasing: bool) -> str:
        """Generate SVG paths directly with optional antialiasing."""
        height, width = alpha.shape

        # Start SVG
        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        ]

        if preserve_antialiasing:
            # Stage 1: Data Preservation - analyze data loss first
            inclusion_threshold = max(16, threshold // 8)  # Much lower threshold for data retention
            significant_pixels = alpha > inclusion_threshold

            # Debug info to track data retention
            total_pixels = alpha.size
            significant_count = np.sum(significant_pixels)
            data_retention_ratio = significant_count / total_pixels

            # Stage 2: Apply user threshold within significant areas for anti-aliasing
            if data_retention_ratio < 0.8:
                print(f"[Alpha-AA] WARNING: Moderate data loss detected ({100*(1-data_retention_ratio):.1f}% lost), consider lower threshold")

            # Create multiple opacity levels for antialiasing
            opacity_levels = {}
            for y in range(height):
                for x in range(width):
                    a = alpha[y, x]
                    if a > inclusion_threshold:  # Use inclusion threshold instead of > 0
                        opacity = round(a / 255, 2)
                        if opacity not in opacity_levels:
                            opacity_levels[opacity] = []
                        opacity_levels[opacity].append((x, y))

            # Generate rectangles for each opacity level
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
            )

            for opacity, pixels in opacity_levels.items():
                # Consolidate adjacent pixels into larger rectangles
                rectangles = self._consolidate_pixels_to_rectangles(pixels, width, height)
                for rect in rectangles:
                    svg_parts.append(
                        f'<rect x="{rect["x"]}" y="{rect["y"]}" width="{rect["width"]}" height="{rect["height"]}" '
                        f'fill="{hex_color}" opacity="{opacity}"/>'
                    )
        else:
            # Simple binary conversion
            # Use contour detection for cleaner paths
            from scipy import ndimage
            binary = (alpha > threshold).astype(np.uint8)

            # Find contours
            labeled, num_features = ndimage.label(binary)

            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
            )

            # For now, use simple rectangle approach
            # (Could be enhanced with actual contour tracing)
            svg_parts.append(f'<g fill="{hex_color}">')

            for y in range(height):
                for x in range(width):
                    if binary[y, x]:
                        # Draw ALL pixels above threshold, not just edges
                        # This ensures filled shapes instead of hollow outlines
                        svg_parts.append(f'<rect x="{x}" y="{y}" width="1" height="1"/>')

            svg_parts.append('</g>')

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    def _convert_standard(self, image_path: str, **kwargs) -> str:
        """Fallback to standard conversion."""
        # Use vtracer as fallback
        import vtracer

        # Filter parameters for VTracer (exclude Alpha-specific parameters)
        vtracer_params = {k: v for k, v in kwargs.items()
                         if k not in ['threshold', 'use_potrace', 'preserve_antialiasing']}

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            try:
                vtracer.convert_image_to_svg_py(
                    image_path,
                    tmp.name,
                    **vtracer_params
                )
                with open(tmp.name, 'r') as f:
                    return f.read()
            finally:
                os.unlink(tmp.name)

    def convert_with_params(self, input_path: str, output_path: str, **params) -> Dict:
        """
        Convert with specific parameters.

        Args:
            input_path: Input PNG file path
            output_path: Output SVG file path
            **params: Conversion parameters

        Returns:
            Dictionary with success status and metadata
        """
        try:
            svg_content = self.convert(input_path, **params)

            with open(output_path, 'w') as f:
                f.write(svg_content)

            return {
                'success': True,
                'output_path': output_path,
                'svg_size': len(svg_content)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _consolidate_pixels_to_rectangles(self, pixels, width, height):
        """
        Consolidate a list of pixels into larger rectangles for optimization.

        Args:
            pixels: List of (x, y) tuples representing pixels with same opacity
            width: Image width
            height: Image height

        Returns:
            List of rectangle dictionaries with x, y, width, height
        """
        if not pixels:
            return []

        # Convert to set for faster lookup
        pixel_set = set(pixels)
        rectangles = []
        processed = set()

        for x, y in pixels:
            if (x, y) in processed:
                continue

            # Find the largest rectangle starting at this pixel
            rect = self._find_largest_rectangle(x, y, pixel_set, processed, width, height)
            if rect:
                rectangles.append(rect)

        return rectangles

    def _find_largest_rectangle(self, start_x, start_y, pixel_set, processed, width, height):
        """
        Find the largest rectangle starting at (start_x, start_y).
        Uses a greedy algorithm to find rectangular regions.
        """
        if (start_x, start_y) in processed or (start_x, start_y) not in pixel_set:
            return None

        # Start with a 1x1 rectangle
        rect_width = 1
        rect_height = 1

        # Expand width as much as possible
        while start_x + rect_width < width:
            # Check if we can extend the width
            can_extend = True
            for y in range(start_y, start_y + rect_height):
                if (start_x + rect_width, y) not in pixel_set or (start_x + rect_width, y) in processed:
                    can_extend = False
                    break
            if not can_extend:
                break
            rect_width += 1

        # Expand height as much as possible
        while start_y + rect_height < height:
            # Check if we can extend the height
            can_extend = True
            for x in range(start_x, start_x + rect_width):
                if (x, start_y + rect_height) not in pixel_set or (x, start_y + rect_height) in processed:
                    can_extend = False
                    break
            if not can_extend:
                break
            rect_height += 1

        # Mark all pixels in this rectangle as processed
        for y in range(start_y, start_y + rect_height):
            for x in range(start_x, start_x + rect_width):
                processed.add((x, y))

        return {
            'x': start_x,
            'y': start_y,
            'width': rect_width,
            'height': rect_height
        }


def test_alpha_converter():
    """Test the alpha converter on sample icons."""
    converter = AlphaConverter()

    test_images = [
        'data/images/heart.png',
        'data/images/calendar.png'
    ]

    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"Test image not found: {img_path}")
            continue

        print(f"\nTesting {img_path}:")

        # Test with potrace
        result = converter.convert_with_params(
            img_path,
            f"{img_path}.alpha_potrace.svg",
            use_potrace=True,
            threshold=128
        )

        if result['success']:
            print(f"  ✓ Potrace conversion successful")
            print(f"    Size: {result['svg_size']} bytes")
        else:
            print(f"  ✗ Potrace conversion failed: {result.get('error')}")

        # Test with antialiasing preservation
        result = converter.convert_with_params(
            img_path,
            f"{img_path}.alpha_aa.svg",
            use_potrace=False,
            preserve_antialiasing=True,
            threshold=128
        )

        if result['success']:
            print(f"  ✓ Antialiased conversion successful")
            print(f"    Size: {result['svg_size']} bytes")
        else:
            print(f"  ✗ Antialiased conversion failed: {result.get('error')}")


if __name__ == "__main__":
    test_alpha_converter()