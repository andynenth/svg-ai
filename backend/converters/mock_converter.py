"""
Mock converter for testing without VTracer.
Creates a simple SVG representation for testing the pipeline.
"""

import os

from PIL import Image

from backend.converters.base import BaseConverter


class MockConverter(BaseConverter):
    """Mock converter that creates simple SVG without VTracer."""

    def __init__(self):
        super().__init__(name="MockConverter")

    def convert(self, image_path: str, **kwargs) -> str:
        """
        Create a simple SVG representation without actual vectorization.

        This is for testing the pipeline when VTracer is not available.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open image to get dimensions
        img = Image.open(image_path)
        width, height = img.size

        # Get dominant color (simplified)
        img_rgb = img.convert('RGB')
        img_small = img_rgb.resize((1, 1))
        dominant_color = img_small.getpixel((0, 0))
        hex_color = '#{:02x}{:02x}{:02x}'.format(*dominant_color)

        # Create a simple SVG with a rectangle
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <!-- Mock SVG conversion (VTracer not installed) -->
  <rect x="0" y="0" width="{width}" height="{height}" fill="{hex_color}" opacity="0.1"/>
  <rect x="{width//4}" y="{height//4}" width="{width//2}" height="{height//2}"
        fill="{hex_color}" rx="10" ry="10"/>
  <text x="{width//2}" y="{height//2}" text-anchor="middle"
        font-family="Arial" font-size="14" fill="black">
    Mock SVG
  </text>
  <text x="{width//2}" y="{height//2 + 20}" text-anchor="middle"
        font-family="Arial" font-size="10" fill="gray">
    (Install VTracer for real conversion)
  </text>
</svg>'''

        return svg

    def get_name(self) -> str:
        """Get converter name."""
        return "MockConverter (Testing Only)"