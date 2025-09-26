#!/usr/bin/env python3
"""
SVG post-processing pipeline for optimization.

This module simplifies paths, merges colors, and reduces file size.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import xml.etree.ElementTree as ET
from collections import Counter
import math

logger = logging.getLogger(__name__)


class SVGPostProcessor:
    """Post-process SVG files for optimization."""

    def __init__(self, precision: int = 2):
        """
        Initialize post-processor.

        Args:
            precision: Decimal precision for coordinates
        """
        self.precision = precision

    def process_file(self, svg_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process an SVG file with all optimizations.

        Args:
            svg_path: Path to input SVG
            output_path: Path to output SVG (default: overwrite input)

        Returns:
            Processing statistics
        """
        if output_path is None:
            output_path = svg_path

        stats = {
            'original_size': 0,
            'final_size': 0,
            'reduction_percent': 0,
            'paths_simplified': 0,
            'colors_merged': 0,
            'redundant_removed': 0
        }

        try:
            # Read original file
            with open(svg_path, 'r') as f:
                svg_content = f.read()
            stats['original_size'] = len(svg_content)

            # Parse SVG
            tree = ET.fromstring(svg_content)

            # Apply optimizations
            self._simplify_paths(tree, stats)
            self._merge_similar_colors(tree, stats)
            self._remove_redundant_elements(tree, stats)
            self._optimize_transforms(tree, stats)
            self._round_coordinates(tree, stats)

            # Write optimized file
            optimized_content = ET.tostring(tree, encoding='unicode')

            # Additional text-based optimizations
            optimized_content = self._compress_path_data(optimized_content)
            optimized_content = self._remove_unnecessary_whitespace(optimized_content)

            with open(output_path, 'w') as f:
                f.write(optimized_content)

            stats['final_size'] = len(optimized_content)
            stats['reduction_percent'] = (1 - stats['final_size'] / stats['original_size']) * 100

            return stats

        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return stats

    def _simplify_paths(self, tree: ET.Element, stats: Dict):
        """Simplify SVG paths by removing redundant points."""
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}

        for path in tree.findall('.//svg:path', namespaces) or tree.findall('.//path'):
            d = path.get('d', '')
            if not d:
                continue

            simplified = self._simplify_path_data(d)
            if len(simplified) < len(d):
                path.set('d', simplified)
                stats['paths_simplified'] += 1

    def _simplify_path_data(self, path_data: str) -> str:
        """Simplify path data by removing redundant commands."""
        # Convert absolute commands to relative where beneficial
        simplified = path_data

        # Remove redundant line segments
        simplified = re.sub(r'([lL])\s*0+\.?0*\s+0+\.?0*\s*', '', simplified)

        # Combine consecutive moves
        simplified = re.sub(r'[mM]\s*0+\.?0*\s+0+\.?0*\s*', '', simplified)

        # Remove trailing zeros
        simplified = re.sub(r'(\d+)\.0+(?=\s|[a-zA-Z]|$)', r'\1', simplified)

        # Remove unnecessary leading zeros
        simplified = re.sub(r'\b0+(\d)', r'\1', simplified)

        return simplified.strip()

    def _merge_similar_colors(self, tree: ET.Element, stats: Dict):
        """Merge similar colors to reduce variety."""
        color_map = {}
        color_counts = Counter()

        # Collect all colors
        for elem in tree.iter():
            fill = elem.get('fill', '')
            stroke = elem.get('stroke', '')

            if fill and fill != 'none':
                color_counts[fill] += 1
            if stroke and stroke != 'none':
                color_counts[stroke] += 1

        # Find similar colors and create mapping
        processed = set()
        for color1 in color_counts:
            if color1 in processed:
                continue

            similar_colors = [color1]
            for color2 in color_counts:
                if color2 != color1 and color2 not in processed:
                    if self._colors_similar(color1, color2):
                        similar_colors.append(color2)
                        processed.add(color2)

            # Map all similar colors to the most common one
            if len(similar_colors) > 1:
                dominant = max(similar_colors, key=lambda c: color_counts[c])
                for color in similar_colors:
                    if color != dominant:
                        color_map[color] = dominant
                        stats['colors_merged'] += 1

        # Apply color mapping
        for elem in tree.iter():
            fill = elem.get('fill', '')
            stroke = elem.get('stroke', '')

            if fill in color_map:
                elem.set('fill', color_map[fill])
            if stroke in color_map:
                elem.set('stroke', color_map[stroke])

    def _colors_similar(self, color1: str, color2: str, threshold: int = 10) -> bool:
        """Check if two colors are similar."""
        try:
            # Convert hex to RGB
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                if len(hex_color) == 3:
                    hex_color = ''.join([c*2 for c in hex_color])
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            if color1.startswith('#') and color2.startswith('#'):
                rgb1 = hex_to_rgb(color1)
                rgb2 = hex_to_rgb(color2)

                # Calculate color distance
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
                return distance < threshold

        except:
            pass

        return False

    def _remove_redundant_elements(self, tree: ET.Element, stats: Dict):
        """Remove redundant or invisible elements."""
        to_remove = []

        for elem in tree.iter():
            # Remove invisible elements
            if elem.get('display') == 'none' or elem.get('visibility') == 'hidden':
                to_remove.append(elem)
                stats['redundant_removed'] += 1
                continue

            # Remove elements with opacity 0
            if elem.get('opacity') == '0' or elem.get('fill-opacity') == '0':
                to_remove.append(elem)
                stats['redundant_removed'] += 1
                continue

            # Remove empty groups
            if elem.tag.endswith('g') and len(elem) == 0:
                to_remove.append(elem)
                stats['redundant_removed'] += 1

        # Remove marked elements
        for elem in to_remove:
            parent = self._find_parent(tree, elem)
            if parent is not None:
                parent.remove(elem)

    def _find_parent(self, tree: ET.Element, child: ET.Element) -> Optional[ET.Element]:
        """Find parent of an element."""
        for parent in tree.iter():
            if child in parent:
                return parent
        return None

    def _optimize_transforms(self, tree: ET.Element, stats: Dict):
        """Optimize transform attributes."""
        for elem in tree.iter():
            transform = elem.get('transform', '')
            if not transform:
                continue

            # Remove identity transforms
            if 'scale(1)' in transform or 'scale(1,1)' in transform:
                transform = transform.replace('scale(1,1)', '').replace('scale(1)', '')

            if 'rotate(0)' in transform:
                transform = transform.replace('rotate(0)', '')

            if 'translate(0,0)' in transform or 'translate(0)' in transform:
                transform = transform.replace('translate(0,0)', '').replace('translate(0)', '')

            # Clean up whitespace
            transform = ' '.join(transform.split())

            if transform:
                elem.set('transform', transform)
            else:
                elem.attrib.pop('transform', None)

    def _round_coordinates(self, tree: ET.Element, stats: Dict):
        """Round coordinates to specified precision."""
        for elem in tree.iter():
            # Round numeric attributes
            for attr in ['x', 'y', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry',
                        'x1', 'y1', 'x2', 'y2', 'points', 'd']:
                value = elem.get(attr, '')
                if value:
                    rounded = self._round_numbers_in_string(value)
                    if rounded != value:
                        elem.set(attr, rounded)

    def _round_numbers_in_string(self, s: str) -> str:
        """Round all numbers in a string to specified precision."""
        def round_match(match):
            num = float(match.group())
            if self.precision == 0:
                return str(int(round(num)))
            else:
                return f"{num:.{self.precision}f}".rstrip('0').rstrip('.')

        # Match floating point numbers
        return re.sub(r'-?\d+\.?\d*', round_match, s)

    def _compress_path_data(self, svg_content: str) -> str:
        """Compress path data by removing unnecessary spaces."""
        # Remove spaces between commands and numbers
        svg_content = re.sub(r'([mlhvcsqtaz])\s+', r'\1', svg_content, flags=re.I)

        # Remove spaces around commas
        svg_content = re.sub(r'\s*,\s*', ',', svg_content)

        # Remove multiple spaces
        svg_content = re.sub(r'\s+', ' ', svg_content)

        return svg_content

    def _remove_unnecessary_whitespace(self, svg_content: str) -> str:
        """Remove unnecessary whitespace from SVG."""
        # Remove whitespace between tags
        svg_content = re.sub(r'>\s+<', '><', svg_content)

        # Remove leading/trailing whitespace
        lines = [line.strip() for line in svg_content.split('\n')]
        svg_content = ''.join(lines)

        return svg_content


def test_post_processing():
    """Test SVG post-processing."""
    print("="*60)
    print("SVG POST-PROCESSING TEST")
    print("="*60)

    processor = SVGPostProcessor(precision=2)

    # Find test SVG files
    test_dir = Path("comparison_results")
    svg_files = list(test_dir.glob("*.svg"))[:3]

    if not svg_files:
        print("No SVG files found for testing")
        return

    total_original = 0
    total_final = 0

    for svg_path in svg_files:
        print(f"\n📄 {svg_path.name}:")

        # Create output path
        output_path = test_dir / f"{svg_path.stem}_optimized.svg"

        # Process file
        stats = processor.process_file(str(svg_path), str(output_path))

        print(f"  Original: {stats['original_size']} bytes")
        print(f"  Final: {stats['final_size']} bytes")
        print(f"  Reduction: {stats['reduction_percent']:.1f}%")
        print(f"  Paths simplified: {stats['paths_simplified']}")
        print(f"  Colors merged: {stats['colors_merged']}")
        print(f"  Redundant removed: {stats['redundant_removed']}")

        total_original += stats['original_size']
        total_final += stats['final_size']

    if total_original > 0:
        total_reduction = (1 - total_final / total_original) * 100
        print(f"\n📊 Total reduction: {total_reduction:.1f}%")


if __name__ == "__main__":
    test_post_processing()