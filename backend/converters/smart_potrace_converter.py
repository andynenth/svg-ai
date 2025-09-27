#!/usr/bin/env python3
"""
Smart Potrace converter that automatically handles both transparent and opaque images.

Combines the best of PotraceConverter and AlphaConverter:
- Auto-detects transparency and chooses the appropriate method
- Uses alpha channel for transparent PNGs
- Uses standard grayscale for opaque images
- Single converter that "just works" for all image types
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from converters.base import BaseConverter


class SmartPotraceConverter(BaseConverter):
    """Smart Potrace converter that auto-detects the best conversion method."""

    def __init__(self):
        """Initialize Smart Potrace converter."""
        super().__init__()
        self.potrace_cmd = self._find_potrace()

        # Default Potrace parameters
        self.turnpolicy = "minority"  # "black", "white", "minority", "majority", "random"
        self.turdsize = 2  # Suppress speckles of up to this size
        self.alphamax = 1.0  # Corner threshold parameter
        self.opttolerance = 0.2  # Curve optimization tolerance

    def get_name(self) -> str:
        """Get converter name."""
        return "Smart Potrace"

    def _find_potrace(self) -> Optional[str]:
        """Find potrace command in common locations."""
        potrace_paths = [
            'potrace',  # System PATH
            '/usr/local/bin/potrace',  # Homebrew on Intel Mac
            '/opt/homebrew/bin/potrace',  # Homebrew on Apple Silicon
            '/usr/bin/potrace',  # Linux standard
        ]

        for path in potrace_paths:
            try:
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def _has_significant_transparency(self, img: Image.Image) -> bool:
        """
        Check if image has significant transparent areas.

        Returns True if:
        - Image has alpha channel AND
        - More than 5% of pixels have alpha < 250
        """
        if img.mode != 'RGBA':
            return False

        alpha = np.array(img.split()[3])
        transparent_ratio = np.sum(alpha < 250) / alpha.size
        return transparent_ratio > 0.05

    def convert(self, image_path: str, **kwargs) -> str:
        """
        Smart conversion that auto-selects the best method.

        Args:
            image_path: Path to input image
            **kwargs: Additional parameters including threshold, potrace options

        Returns:
            SVG content as string
        """
        if not self.potrace_cmd:
            raise Exception("Potrace not found. Please install potrace first.")

        # Open image and detect transparency
        img = Image.open(image_path)

        # Auto-select conversion method based on image characteristics
        if self._has_significant_transparency(img):
            print(f"[Smart Potrace] Detected transparent image, using alpha-aware mode")
            return self._convert_with_alpha(img, **kwargs)
        else:
            print(f"[Smart Potrace] Detected opaque image, using standard mode")
            return self._convert_standard(img, **kwargs)

    def _convert_with_alpha(self, img: Image.Image, **kwargs) -> str:
        """
        Convert using alpha channel for transparent images.
        Similar to AlphaConverter's approach.
        """
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Extract alpha channel
        arr = np.array(img)
        alpha = arr[:, :, 3]

        # Get RGB color (for icons, usually uniform)
        # Sample from high-alpha pixels to get the actual color
        high_alpha_mask = alpha > 200
        if np.any(high_alpha_mask):
            rgb_color = arr[high_alpha_mask][:, :3].mean(axis=0)
        else:
            rgb_color = [0, 0, 0]  # Default to black if no solid pixels

        # Apply threshold to alpha channel
        threshold = kwargs.get('threshold', 128)
        print(f"[Smart Potrace] Using alpha threshold: {threshold}")

        # Create binary image from alpha
        binary = (alpha > threshold).astype(np.uint8)

        # Invert for Potrace (Potrace traces white pixels as black)
        binary = 255 - (binary * 255)

        # Convert to PIL Image
        img_binary = Image.fromarray(binary, mode='L').convert('1')

        # Save as PBM and process with Potrace
        return self._run_potrace(img_binary, rgb_color, **kwargs)

    def _convert_standard(self, img: Image.Image, **kwargs) -> str:
        """
        Convert using standard grayscale for opaque images.
        Similar to original PotraceConverter's approach.
        """
        # Handle different image modes
        if img.mode == 'RGBA':
            # Composite on white background (shouldn't happen often due to detection)
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
            print(f"[Smart Potrace] Composited RGBA on white background")
        elif img.mode == 'P':
            img = img.convert('RGB')
            print(f"[Smart Potrace] Converted palette to RGB")

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
            print(f"[Smart Potrace] Converted to grayscale")

        # Apply threshold
        threshold = kwargs.get('threshold', 128)
        print(f"[Smart Potrace] Using grayscale threshold: {threshold}")

        # Create binary image
        img_binary = img.point(lambda x: 255 if x > threshold else 0, mode='1')

        # Use black as the color for standard mode
        rgb_color = [0, 0, 0]

        return self._run_potrace(img_binary, rgb_color, **kwargs)

    def _run_potrace(self, img_binary: Image.Image, rgb_color: list, **kwargs) -> str:
        """
        Run Potrace on binary image and return SVG with specified color.

        Args:
            img_binary: Binary PIL Image
            rgb_color: RGB color values [r, g, b]
            **kwargs: Potrace parameters
        """
        # Extract Potrace parameters
        turnpolicy = kwargs.get('turnpolicy', self.turnpolicy)
        turdsize = kwargs.get('turdsize', self.turdsize)
        alphamax = kwargs.get('alphamax', self.alphamax)
        opttolerance = kwargs.get('opttolerance', self.opttolerance)

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp_pbm:
            img_binary.save(tmp_pbm.name)

            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                try:
                    # Build Potrace command
                    cmd = [
                        self.potrace_cmd,
                        '-s',  # SVG output
                        tmp_pbm.name,
                        '-o', tmp_svg.name
                    ]

                    # Add optional parameters
                    if turnpolicy != "minority":
                        cmd.extend(['-z', turnpolicy])
                    if turdsize != 2:
                        cmd.extend(['-t', str(turdsize)])
                    if alphamax != 1.0:
                        cmd.extend(['-a', str(alphamax)])
                    if opttolerance != 0.2:
                        # Potrace requires opttolerance > 0, so use minimum 0.001 for slider position 0
                        actual_opttolerance = max(0.001, opttolerance)
                        cmd.extend(['-O', str(actual_opttolerance)])

                    # Debug: Log the actual Potrace command
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"[Smart Potrace] Running command: {' '.join(map(str, cmd))}")
                    logger.info(f"[Smart Potrace] Parameters - threshold: {kwargs.get('threshold')}, turnpolicy: {turnpolicy}, turdsize: {turdsize}, alphamax: {alphamax}, opttolerance: {opttolerance}")
                    print(f"[Smart Potrace DEBUG] Command: {' '.join(map(str, cmd))}")
                    print(f"[Smart Potrace DEBUG] opttolerance={opttolerance}, actual flag: {'-O' if opttolerance != 0.2 else 'default'}")

                    # Run Potrace
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode != 0:
                        raise Exception(f"Potrace failed: {result.stderr}")

                    # Read SVG and replace color
                    with open(tmp_svg.name, 'r') as f:
                        svg_content = f.read()

                    # Replace black fill with specified color
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(rgb_color[0]),
                        int(rgb_color[1]),
                        int(rgb_color[2])
                    )

                    if hex_color != '#000000':
                        svg_content = svg_content.replace('fill="#000000"', f'fill="{hex_color}"')

                    # Analyze SVG for optimization metrics
                    path_count = svg_content.count('<path')
                    svg_size = len(svg_content)

                    logger.info(f"[Smart Potrace] SVG generated - Size: {svg_size} bytes, Paths: {path_count}, opttolerance: {opttolerance}")
                    print(f"[Smart Potrace DEBUG] SVG metrics - Size: {svg_size}B, Paths: {path_count}, Color: {hex_color}")

                    return svg_content

                finally:
                    # Clean up temp files
                    os.unlink(tmp_pbm.name)
                    if os.path.exists(tmp_svg.name):
                        os.unlink(tmp_svg.name)

    def convert_with_params(self, input_path: str, output_path: str, **params) -> Dict:
        """
        Convert with specific parameters.

        Args:
            input_path: Input image file path
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


def test_smart_potrace():
    """Test the Smart Potrace converter on various image types."""
    converter = SmartPotraceConverter()

    if not converter.potrace_cmd:
        print("Potrace not found. Please install potrace to test.")
        return

    test_images = [
        # Add your test images here
        'test_transparent.png',  # Should use alpha mode
        'test_opaque.jpg',       # Should use standard mode
    ]

    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"Test image not found: {img_path}")
            continue

        print(f"\nTesting {img_path}:")
        result = converter.convert_with_params(
            img_path,
            f"{img_path}.smart.svg",
            threshold=128
        )

        if result['success']:
            print(f"  ✓ Conversion successful")
            print(f"    Size: {result['svg_size']} bytes")
        else:
            print(f"  ✗ Conversion failed: {result.get('error')}")


if __name__ == "__main__":
    test_smart_potrace()