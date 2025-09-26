"""
Visual comparison utilities for PNG to SVG conversion.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageChops
from pathlib import Path
from typing import Tuple, Dict, Optional
import cairosvg
import io


class VisualComparer:
    """Compare original PNG with converted SVG visually."""

    def __init__(self):
        self.comparison_size = (512, 512)  # Standard size for comparisons

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

    def create_diff_image(self, original_path: str, svg_content: str) -> Tuple[Image.Image, Dict]:
        """
        Create a visual diff between original and converted images.

        Returns:
            Tuple of (diff_image, metrics)
        """
        # Load and resize original
        original = Image.open(original_path).convert('RGBA')
        original = original.resize(self.comparison_size, Image.Resampling.LANCZOS)

        # Convert SVG to PNG
        converted = self.svg_to_png(svg_content)

        # Ensure both images are RGBA
        if converted.mode != 'RGBA':
            converted = converted.convert('RGBA')

        # Calculate difference
        diff = ImageChops.difference(original, converted)

        # Create enhanced diff visualization
        diff_enhanced = Image.new('RGBA', self.comparison_size, (255, 255, 255, 255))
        diff_data = np.array(diff)

        # Calculate diff metrics - handle alpha channel properly
        total_pixels = diff_data.shape[0] * diff_data.shape[1]

        # Check if we need to compare alpha channels instead of RGB
        orig_array = np.array(original)
        conv_array = np.array(converted)

        # If RGB is all black/same, use alpha channel for comparison
        if orig_array.shape[2] >= 4 and np.std(orig_array[:, :, :3]) < 10:
            # Compare using alpha channels
            alpha_diff = np.abs(orig_array[:, :, 3].astype(float) - conv_array[:, :, 3].astype(float))
            significant_diff = np.sum(alpha_diff > 3)
            diff_percentage = (significant_diff / total_pixels) * 100

            # Update diff_data to use alpha comparison for visualization
            for i in range(3):
                diff_data[:, :, i] = alpha_diff
        else:
            # Standard RGB comparison
            diff_threshold = 3  # Lower threshold for detecting differences
            any_diff = np.sum(np.max(diff_data[:, :, :3], axis=2) > 0)
            significant_diff = np.sum(np.max(diff_data[:, :, :3], axis=2) > diff_threshold)
            diff_percentage = (significant_diff / total_pixels) * 100

        # Create heatmap visualization
        for y in range(diff_data.shape[0]):
            for x in range(diff_data.shape[1]):
                pixel_diff = np.max(diff_data[y, x, :3])
                if pixel_diff > 0:
                    # Color based on difference intensity - more sensitive thresholds
                    if pixel_diff < 3:
                        color = (100, 100, 255, 50)  # Light blue for tiny differences
                    elif pixel_diff < 10:
                        color = (0, 255, 0, 100)  # Green for small differences
                    elif pixel_diff < 30:
                        color = (255, 255, 0, 150)  # Yellow for moderate
                    else:
                        color = (255, 0, 0, 200)  # Red for large differences

                    diff_enhanced.putpixel((x, y), color)

        # Calculate metrics
        metrics = {
            'diff_percentage': diff_percentage,
            'max_diff': np.max(diff_data[:, :, :3]),
            'mean_diff': np.mean(diff_data[:, :, :3]),
            'significant_pixels': int(significant_diff),
            'total_pixels': total_pixels
        }

        return diff_enhanced, metrics

    def create_comparison_grid(self, original_path: str, svg_content: str) -> Image.Image:
        """
        Create a grid showing original, converted, and diff side by side.
        """
        # Load images
        original = Image.open(original_path).convert('RGBA')
        original = original.resize(self.comparison_size, Image.Resampling.LANCZOS)

        converted = self.svg_to_png(svg_content)
        if converted.mode != 'RGBA':
            converted = converted.convert('RGBA')

        diff, metrics = self.create_diff_image(original_path, svg_content)

        # Create grid
        grid_width = self.comparison_size[0] * 3 + 40  # 20px padding between
        grid_height = self.comparison_size[1] + 100  # Extra space for labels

        grid = Image.new('RGBA', (grid_width, grid_height), (240, 240, 240, 255))
        draw = ImageDraw.Draw(grid)

        # Paste images
        grid.paste(original, (10, 50))
        grid.paste(converted, (self.comparison_size[0] + 20, 50))
        grid.paste(diff, (self.comparison_size[0] * 2 + 30, 50))

        # Add labels (basic text, could be enhanced with proper font)
        try:
            from PIL import ImageFont
            # Try to use a better font if available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), "Original PNG", fill=(0, 0, 0), font=font)
            draw.text((self.comparison_size[0] + 20, 10), "Converted SVG", fill=(0, 0, 0), font=font)
            draw.text((self.comparison_size[0] * 2 + 30, 10), f"Difference ({metrics['diff_percentage']:.1f}%)",
                     fill=(0, 0, 0), font=font)
        except:
            pass

        return grid

    def analyze_conversion_quality(self, original_path: str, svg_content: str) -> Dict:
        """
        Comprehensive quality analysis with visual metrics.
        """
        # Get basic diff metrics
        diff_image, diff_metrics = self.create_diff_image(original_path, svg_content)

        # Load images for additional analysis
        original = Image.open(original_path).convert('RGBA')
        original = original.resize(self.comparison_size, Image.Resampling.LANCZOS)
        converted = self.svg_to_png(svg_content)

        # Color analysis
        original_colors = self._count_unique_colors(original)
        converted_colors = self._count_unique_colors(converted)

        # Edge preservation analysis
        edge_score = self._calculate_edge_preservation(original, converted)

        # Compile full analysis
        analysis = {
            'visual_metrics': diff_metrics,
            'color_analysis': {
                'original_colors': original_colors,
                'converted_colors': converted_colors,
                'color_reduction': original_colors - converted_colors,
                'color_reduction_percent': ((original_colors - converted_colors) / original_colors * 100)
                                         if original_colors > 0 else 0
            },
            'edge_preservation': edge_score,
            'quality_rating': self._calculate_quality_rating(diff_metrics, edge_score)
        }

        return analysis

    def _count_unique_colors(self, image: Image.Image) -> int:
        """Count unique colors in image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        colors = image.getcolors(maxcolors=100000)
        return len(colors) if colors else 0

    def _calculate_edge_preservation(self, original: Image.Image, converted: Image.Image) -> float:
        """Calculate how well edges are preserved (0-100 score)."""
        try:
            from PIL import ImageFilter

            # Apply edge detection
            original_edges = original.convert('L').filter(ImageFilter.FIND_EDGES)
            converted_edges = converted.convert('L').filter(ImageFilter.FIND_EDGES)

            # Compare edge maps
            diff = ImageChops.difference(original_edges, converted_edges)
            diff_array = np.array(diff)

            # Calculate preservation score
            preservation = 100 - (np.mean(diff_array) / 255 * 100)
            return max(0, min(100, preservation))
        except:
            return 0

    def _calculate_quality_rating(self, diff_metrics: Dict, edge_score: float) -> str:
        """Calculate overall quality rating."""
        diff_pct = diff_metrics['diff_percentage']
        mean_diff = diff_metrics['mean_diff']

        score = (100 - diff_pct) * 0.5 + edge_score * 0.3 + (100 - mean_diff/2.55) * 0.2

        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Acceptable"
        elif score >= 40:
            return "Poor"
        else:
            return "Unacceptable"