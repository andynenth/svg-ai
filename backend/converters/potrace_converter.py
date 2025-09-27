"""
Potrace-based converter using pypotrace or subprocess.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from .base import BaseConverter


class PotraceConverter(BaseConverter):
    """Potrace-based PNG to SVG converter."""

    def __init__(self):
        super().__init__(name="Potrace")
        self.potrace_cmd = self._find_potrace()

    def _find_potrace(self):
        """Find potrace command."""
        # Check if potrace is installed
        try:
            result = subprocess.run(['which', 'potrace'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        # Check common locations
        common_paths = [
            '/usr/local/bin/potrace',
            '/opt/homebrew/bin/potrace',
            '/usr/bin/potrace'
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def convert(self, image_path: str, **kwargs) -> str:
        """Convert PNG to SVG using Potrace."""
        if not self.potrace_cmd:
            # Install instructions
            return self._get_install_instructions()

        # Potrace only works with bitmap (PBM) format
        # Convert PNG to PBM first
        img = Image.open(image_path)

        # Handle RGBA images by compositing on white background
        if img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Composite the RGBA image onto white background
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
            print(f"[Potrace] Converted RGBA to RGB with white background")
        elif img.mode == 'P':
            # Convert palette images to RGB first
            img = img.convert('RGB')
            print(f"[Potrace] Converted palette image to RGB")

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
            print(f"[Potrace] Converted to grayscale")

        # Apply threshold to create bitmap
        threshold = kwargs.get('threshold', 128)
        print(f"[Potrace] Applying threshold: {threshold}")

        # Debug: Check image stats before threshold
        import numpy as np
        img_array = np.array(img)
        print(f"[Potrace] Image stats - Min: {img_array.min()}, Max: {img_array.max()}, Mean: {img_array.mean():.1f}")

        # Create binary image: pixels > threshold become white (255), else black (0)
        img = img.point(lambda x: 255 if x > threshold else 0, mode='1')

        # Debug: Check how many pixels are black vs white after threshold
        binary_array = np.array(img)
        black_pixels = np.sum(binary_array == False)  # In mode '1', black is False
        white_pixels = np.sum(binary_array == True)   # In mode '1', white is True
        total_pixels = binary_array.size
        print(f"[Potrace] After threshold: {black_pixels} black pixels ({100*black_pixels/total_pixels:.1f}%), {white_pixels} white pixels ({100*white_pixels/total_pixels:.1f}%)")

        # Save as PBM
        with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp_pbm:
            img.save(tmp_pbm.name)
            print(f"[Potrace] Saved PBM to: {tmp_pbm.name}")

            # DEBUG: Save a copy for inspection
            import shutil
            debug_pbm = f"/tmp/potrace_debug_{threshold}.pbm"
            shutil.copy(tmp_pbm.name, debug_pbm)
            print(f"[Potrace] DEBUG: Saved copy to {debug_pbm} for inspection")

            # Convert PBM to SVG using potrace
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                try:
                    result = subprocess.run(
                        [self.potrace_cmd, '-s', tmp_pbm.name, '-o', tmp_svg.name],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        with open(tmp_svg.name, 'r') as f:
                            svg_content = f.read()

                        print(f"[Potrace] Original SVG length: {len(svg_content)} chars")
                        print(f"[Potrace] SVG preview: {svg_content[:200]}...")

                        # Check what the SVG contains
                        if 'fill="#000000"' in svg_content:
                            print("[Potrace] SVG contains black fill")
                        if 'fill="white"' in svg_content:
                            print("[Potrace] SVG already contains white fill")

                        # Count path elements
                        path_count = svg_content.count('<path')
                        print(f"[Potrace] SVG contains {path_count} path elements")

                        # DEBUG: Save original SVG for inspection
                        debug_svg = f"/tmp/potrace_debug_{threshold}_original.svg"
                        with open(debug_svg, 'w') as f:
                            f.write(svg_content)
                        print(f"[Potrace] DEBUG: Saved original SVG to {debug_svg}")

                        # Note: SVG maintains transparency like other converters (alpha-aware, vtracer)
                        # No white background is added to preserve transparency

                        return svg_content
                    else:
                        raise Exception(f"Potrace failed: {result.stderr}")

                finally:
                    # Clean up temp files
                    os.unlink(tmp_pbm.name)
                    if os.path.exists(tmp_svg.name):
                        os.unlink(tmp_svg.name)

        return svg_content

    def _get_install_instructions(self):
        """Return SVG with install instructions."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="400" height="200" fill="#f0f0f0"/>
  <text x="200" y="50" text-anchor="middle" font-family="Arial" font-size="16" fill="black">
    Potrace not installed
  </text>
  <text x="200" y="80" text-anchor="middle" font-family="Arial" font-size="12" fill="gray">
    Install with: brew install potrace
  </text>
  <text x="200" y="110" text-anchor="middle" font-family="Arial" font-size="12" fill="gray">
    or visit: http://potrace.sourceforge.net
  </text>
</svg>'''

    def get_name(self) -> str:
        """Get converter name."""
        return "Potrace"


class AutoTraceConverter(BaseConverter):
    """AutoTrace-based converter (alternative to Potrace)."""

    def __init__(self):
        super().__init__(name="AutoTrace")

    def convert(self, image_path: str, **kwargs) -> str:
        """Convert using autotrace."""
        try:
            # Check if autotrace is available
            result = subprocess.run(
                ['autotrace', '--version'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Run autotrace
                result = subprocess.run(
                    ['autotrace', '-output-format', 'svg', image_path],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    return result.stdout
        except:
            pass

        return self._get_install_instructions()

    def _get_install_instructions(self):
        """Return SVG with install instructions."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="400" height="200" fill="#f0f0f0"/>
  <text x="200" y="50" text-anchor="middle" font-family="Arial" font-size="16" fill="black">
    AutoTrace not installed
  </text>
  <text x="200" y="80" text-anchor="middle" font-family="Arial" font-size="12" fill="gray">
    Install with: brew install autotrace
  </text>
</svg>'''

    def get_name(self) -> str:
        """Get converter name."""
        return "AutoTrace"