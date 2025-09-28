"""
SVG post-processing and optimization utilities.
"""

import re
import logging
from typing import Dict, Optional
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class SVGOptimizer:
    """Post-process and optimize SVG output."""

    def __init__(self):
        self.svgo_available = self._check_svgo()

    def _check_svgo(self) -> bool:
        """Check if SVGO is installed."""
        try:
            result = subprocess.run(['which', 'svgo'], capture_output=True)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Failed to check for svgo availability: {e}")
            return False

    def optimize(self, svg_content: str, aggressive: bool = False) -> str:
        """
        Optimize SVG content.

        Args:
            svg_content: Raw SVG string
            aggressive: If True, apply more aggressive optimizations

        Returns:
            Optimized SVG string
        """
        # Apply manual optimizations first
        svg_content = self.clean_coordinates(svg_content)
        svg_content = self.merge_paths(svg_content)
        svg_content = self.remove_redundant_attributes(svg_content)

        if aggressive:
            svg_content = self.simplify_paths(svg_content)
            svg_content = self.quantize_colors(svg_content)

        # Use SVGO if available
        if self.svgo_available:
            svg_content = self.run_svgo(svg_content)

        return svg_content

    def clean_coordinates(self, svg_content: str, precision: int = 2) -> str:
        """Round coordinates to reduce file size."""
        # Find all numeric values in paths
        def round_match(match):
            value = float(match.group())
            return f"{value:.{precision}f}".rstrip('0').rstrip('.')

        # Pattern for numbers in SVG paths
        pattern = r'-?\d+\.?\d*'
        svg_content = re.sub(pattern, round_match, svg_content)

        return svg_content

    def merge_paths(self, svg_content: str) -> str:
        """Merge paths with identical styles."""
        try:
            root = ET.fromstring(svg_content)

            # Group paths by fill color
            path_groups = {}
            for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
                fill = path.get('fill', 'none')
                if fill not in path_groups:
                    path_groups[fill] = []
                path_groups[fill].append(path)

            # Merge paths with same fill
            for fill, paths in path_groups.items():
                if len(paths) > 1:
                    # Combine path data
                    combined_d = ' '.join(p.get('d', '') for p in paths)

                    # Keep first path, update its d attribute
                    paths[0].set('d', combined_d)

                    # Remove other paths
                    for p in paths[1:]:
                        root.remove(p)

            return ET.tostring(root, encoding='unicode')
        except (ET.ParseError, AttributeError, ValueError) as e:
            logger.warning(f"SVG parsing/optimization failed: {e}")
            logger.debug("Returning original SVG content without optimization")
            # If parsing fails, return original
            return svg_content

    def remove_redundant_attributes(self, svg_content: str) -> str:
        """Remove unnecessary attributes."""
        # Remove default values
        svg_content = re.sub(r'\s+stroke-width="1"', '', svg_content)
        svg_content = re.sub(r'\s+stroke-opacity="1"', '', svg_content)
        svg_content = re.sub(r'\s+fill-opacity="1"', '', svg_content)

        # Remove generator comments
        svg_content = re.sub(r'<!--.*?-->', '', svg_content, flags=re.DOTALL)

        # Clean whitespace
        svg_content = re.sub(r'>\s+<', '><', svg_content)

        return svg_content

    def simplify_paths(self, svg_content: str, tolerance: float = 0.5) -> str:
        """Simplify path commands (aggressive)."""
        # Convert absolute to relative commands where beneficial
        svg_content = re.sub(r'([MLC])\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)',
                            lambda m: f"{m.group(1).lower()} {m.group(2)} {m.group(3)}"
                            if float(m.group(2)) < 50 and float(m.group(3)) < 50
                            else m.group(0), svg_content)

        return svg_content

    def quantize_colors(self, svg_content: str, levels: int = 32) -> str:
        """Reduce color precision (aggressive)."""
        def quantize_hex(match):
            hex_color = match.group(1)
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            # Quantize to fewer levels
            step = 256 // levels
            r = (r // step) * step
            g = (g // step) * step
            b = (b // step) * step

            return f"#{r:02x}{g:02x}{b:02x}"

        svg_content = re.sub(r'#([0-9a-fA-F]{6})', quantize_hex, svg_content)
        return svg_content

    def run_svgo(self, svg_content: str) -> str:
        """Run SVGO optimizer if available."""
        if not self.svgo_available:
            return svg_content

        try:
            import tempfile
            import os

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp_in:
                tmp_in.write(svg_content)
                tmp_in_path = tmp_in.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp_out:
                tmp_out_path = tmp_out.name

            # Run SVGO
            result = subprocess.run(
                ['svgo', tmp_in_path, '-o', tmp_out_path, '--multipass'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                with open(tmp_out_path, 'r') as f:
                    optimized = f.read()
                svg_content = optimized

            # Clean up temp files
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

        except Exception as e:
            print(f"SVGO optimization failed: {e}")

        return svg_content

    def get_stats(self, svg_content: str) -> Dict:
        """Get statistics about SVG content."""
        stats = {
            'size_bytes': len(svg_content),
            'size_kb': len(svg_content) / 1024,
            'num_paths': len(re.findall(r'<path', svg_content)),
            'num_groups': len(re.findall(r'<g', svg_content)),
            'num_colors': len(set(re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg_content))),
            'has_gradients': 'linearGradient' in svg_content or 'radialGradient' in svg_content,
            'has_transforms': 'transform=' in svg_content
        }

        # Count total path commands
        path_commands = re.findall(r'd="([^"]*)"', svg_content)
        if path_commands:
            total_commands = sum(len(re.findall(r'[MLHVCSQTAZmlhvcsqtaz]', p)) for p in path_commands)
            stats['total_path_commands'] = total_commands

        return stats