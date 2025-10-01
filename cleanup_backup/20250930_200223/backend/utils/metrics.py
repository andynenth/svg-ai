import os
import time
from PIL import Image
from typing import Dict, Any, Optional
import numpy as np


class ConversionMetrics:
    """Calculate metrics for PNG to SVG conversion."""

    @staticmethod
    def calculate_basic(
        png_path: str,
        svg_path: str,
        conversion_time: float
    ) -> Dict[str, Any]:
        """
        Calculate basic conversion metrics.

        Args:
            png_path: Path to original PNG
            svg_path: Path to generated SVG
            conversion_time: Time taken for conversion

        Returns:
            Dictionary with basic metrics
        """
        png_size = os.path.getsize(png_path) if os.path.exists(png_path) else 0
        svg_size = os.path.getsize(svg_path) if os.path.exists(svg_path) else 0

        return {
            'png_size_kb': round(png_size / 1024, 2),
            'svg_size_kb': round(svg_size / 1024, 2),
            'compression_ratio': round(svg_size / png_size, 3) if png_size > 0 else 0,
            'size_reduction_pct': round((1 - svg_size / png_size) * 100, 1) if png_size > 0 else 0,
            'conversion_time_s': round(conversion_time, 3),
            'success': svg_size > 0
        }

    @staticmethod
    def estimate_svg_complexity(svg_content: str) -> Dict[str, int]:
        """
        Estimate SVG complexity from content.

        Args:
            svg_content: SVG file content

        Returns:
            Dictionary with complexity metrics
        """
        if not svg_content:
            return {'paths': 0, 'commands': 0, 'groups': 0}

        # Count SVG elements
        return {
            'paths': svg_content.count('<path'),
            'commands': sum(svg_content.count(cmd) for cmd in ['M', 'L', 'C', 'Q', 'A', 'Z']),
            'groups': svg_content.count('<g'),
            'colors': len(set(import_re.findall(r'fill="([^"]+)"', svg_content)))
        }


import re as import_re