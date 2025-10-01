#!/usr/bin/env python3
"""
SVG validation and processing utilities.

This module provides centralized SVG validation and processing functions
to eliminate code duplication across converters.
"""

import re
import logging
from typing import Tuple, Optional, Dict, Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class SVGValidator:
    """
    Centralized SVG validation and processing utilities.

    This class consolidates SVG processing operations that were
    previously duplicated across multiple converter classes.
    """

    @staticmethod
    def add_viewbox_if_missing(svg_content: str) -> str:
        """
        Add viewBox attribute to SVG if missing for proper scaling.

        Args:
            svg_content: SVG content as string

        Returns:
            SVG content with viewBox added if it was missing
        """
        try:
            if 'viewBox' in svg_content:
                logger.debug("SVG already has viewBox attribute")
                return svg_content

            # Extract width and height from SVG tag (not child elements)
            svg_tag_match = re.search(r'<svg[^>]*?>', svg_content)
            if svg_tag_match:
                svg_tag = svg_tag_match.group(0)
                width_match = re.search(r'width="(\d+(?:\.\d+)?)"', svg_tag)
                height_match = re.search(r'height="(\d+(?:\.\d+)?)"', svg_tag)
            else:
                width_match = height_match = None

            if width_match and height_match:
                width = width_match.group(1)
                height = height_match.group(1)

                # Add viewBox attribute to the svg tag
                svg_content = re.sub(
                    r'<svg([^>]*?)width="(\d+(?:\.\d+)?)"([^>]*?)height="(\d+(?:\.\d+)?)"',
                    rf'<svg\1width="\2"\3height="\4" viewBox="0 0 \2 \4"',
                    svg_content
                )

                logger.debug(f"Added viewBox=\"0 0 {width} {height}\" for proper scaling")
            else:
                logger.warning("Could not extract width/height to add viewBox")

            return svg_content

        except Exception as e:
            logger.error(f"Failed to add viewBox: {e}")
            return svg_content

    @staticmethod
    def validate_svg_structure(svg_content: str) -> bool:
        """
        Validate basic SVG structure and syntax.

        Args:
            svg_content: SVG content as string

        Returns:
            True if SVG structure is valid, False otherwise
        """
        try:
            if not svg_content or not svg_content.strip():
                logger.warning("SVG content is empty")
                return False

            # Check for required SVG elements
            if '<svg' not in svg_content:
                logger.warning("SVG content missing <svg> tag")
                return False

            if 'xmlns' not in svg_content:
                logger.warning("SVG content missing xmlns declaration")
                return False

            # Try to parse as XML
            try:
                ET.fromstring(svg_content)
                logger.debug("SVG structure validation passed")
                return True
            except ET.ParseError as e:
                logger.warning(f"SVG XML parsing failed: {e}")
                return False

        except Exception as e:
            logger.error(f"SVG validation failed: {e}")
            return False

    @staticmethod
    def extract_dimensions(svg_content: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract width and height dimensions from SVG content.

        Args:
            svg_content: SVG content as string

        Returns:
            Tuple of (width, height) as floats, or (None, None) if not found
        """
        try:
            # Extract width and height from SVG tag only (not child elements)
            svg_tag_match = re.search(r'<svg[^>]*?>', svg_content)
            if svg_tag_match:
                svg_tag = svg_tag_match.group(0)
                width_match = re.search(r'width="(\d+(?:\.\d+)?)"', svg_tag)
                height_match = re.search(r'height="(\d+(?:\.\d+)?)"', svg_tag)
            else:
                width_match = height_match = None

            width = float(width_match.group(1)) if width_match else None
            height = float(height_match.group(1)) if height_match else None

            if width and height:
                logger.debug(f"Extracted SVG dimensions: {width}x{height}")
            else:
                logger.warning("Could not extract SVG dimensions")

            return width, height

        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to extract dimensions: {e}")
            return None, None

    @staticmethod
    def sanitize_svg_content(svg_content: str) -> str:
        """
        Sanitize SVG content by removing potentially harmful elements.

        Args:
            svg_content: SVG content as string

        Returns:
            Sanitized SVG content
        """
        try:
            # Remove script tags and javascript
            svg_content = re.sub(r'<script[^>]*>.*?</script>', '', svg_content, flags=re.DOTALL | re.IGNORECASE)
            svg_content = re.sub(r'javascript:', '', svg_content, flags=re.IGNORECASE)

            # Remove onclick and other event handlers
            svg_content = re.sub(r'\son\w+\s*=\s*["\'][^"\']*["\']', '', svg_content, flags=re.IGNORECASE)

            # Remove external references (for security)
            svg_content = re.sub(r'href\s*=\s*["\']https?://[^"\']*["\']', '', svg_content, flags=re.IGNORECASE)

            logger.debug("SVG content sanitized")
            return svg_content

        except Exception as e:
            logger.error(f"Failed to sanitize SVG: {e}")
            return svg_content

    @staticmethod
    def replace_fill_color(svg_content: str, old_color: str, new_color: str) -> str:
        """
        Replace fill color in SVG content.

        Args:
            svg_content: SVG content as string
            old_color: Color to replace (e.g., "#000000")
            new_color: New color (e.g., "#ff0000")

        Returns:
            SVG content with color replaced
        """
        try:
            if old_color == new_color:
                logger.debug("Old and new colors are the same, no replacement needed")
                return svg_content

            # Replace exact fill color matches
            pattern = f'fill="{old_color}"'
            replacement = f'fill="{new_color}"'

            updated_content = svg_content.replace(pattern, replacement)

            # Also try case-insensitive replacement
            pattern_upper = f'fill="{old_color.upper()}"'
            replacement_upper = f'fill="{new_color}"'
            updated_content = updated_content.replace(pattern_upper, replacement_upper)

            if updated_content != svg_content:
                logger.debug(f"Replaced fill color {old_color} with {new_color}")
            else:
                logger.debug(f"No instances of fill color {old_color} found")

            return updated_content

        except Exception as e:
            logger.error(f"Failed to replace fill color: {e}")
            return svg_content

    @staticmethod
    def create_svg_header(width: float, height: float, xmlns: bool = True) -> str:
        """
        Create standard SVG header with proper declarations.

        Args:
            width: SVG width
            height: SVG height
            xmlns: Whether to include xmlns declaration

        Returns:
            SVG header string
        """
        try:
            header_parts = ['<?xml version="1.0" encoding="UTF-8"?>']

            svg_attrs = [f'width="{width}"', f'height="{height}"']

            if xmlns:
                svg_attrs.append('xmlns="http://www.w3.org/2000/svg"')

            svg_attrs.append(f'viewBox="0 0 {width} {height}"')

            svg_tag = f'<svg {" ".join(svg_attrs)}>'
            header_parts.append(svg_tag)

            logger.debug(f"Created SVG header for {width}x{height}")
            return '\n'.join(header_parts)

        except Exception as e:
            logger.error(f"Failed to create SVG header: {e}")
            return f'<svg width="{width}" height="{height}">'

    @staticmethod
    def get_svg_info(svg_content: str) -> Dict[str, Any]:
        """
        Get comprehensive information about SVG content.

        Args:
            svg_content: SVG content as string

        Returns:
            Dictionary with SVG information
        """
        try:
            width, height = SVGValidator.extract_dimensions(svg_content)

            info = {
                'width': width,
                'height': height,
                'size_bytes': len(svg_content.encode('utf-8')),
                'has_viewbox': 'viewBox' in svg_content,
                'has_xmlns': 'xmlns' in svg_content,
                'is_valid': SVGValidator.validate_svg_structure(svg_content),
                'path_count': svg_content.count('<path'),
                'rect_count': svg_content.count('<rect'),
                'circle_count': svg_content.count('<circle'),
                'total_elements': svg_content.count('<') - svg_content.count('</') - 2  # Subtract closing tags and xml declaration
            }

            logger.debug(f"Generated SVG info: {info}")
            return info

        except Exception as e:
            logger.error(f"Failed to get SVG info: {e}")
            return {'error': str(e)}

    @staticmethod
    def optimize_svg_structure(svg_content: str) -> str:
        """
        Apply basic optimizations to SVG structure.

        Args:
            svg_content: SVG content as string

        Returns:
            Optimized SVG content
        """
        try:
            # Add viewBox if missing
            optimized = SVGValidator.add_viewbox_if_missing(svg_content)

            # Remove unnecessary whitespace between tags
            optimized = re.sub(r'>\s+<', '><', optimized)

            # Remove comments
            optimized = re.sub(r'<!--.*?-->', '', optimized, flags=re.DOTALL)

            # Ensure proper formatting
            optimized = optimized.strip()

            logger.debug("Applied basic SVG optimizations")
            return optimized

        except Exception as e:
            logger.error(f"Failed to optimize SVG: {e}")
            return svg_content


def validate_svg_file(file_path: str) -> bool:
    """
    Validate SVG file by reading and checking its structure.

    Args:
        file_path: Path to SVG file

    Returns:
        True if file is valid SVG, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return SVGValidator.validate_svg_structure(content)

    except Exception as e:
        logger.error(f"Failed to validate SVG file {file_path}: {e}")
        return False


def add_viewbox_to_file(input_path: str, output_path: str) -> bool:
    """
    Add viewBox to SVG file if missing.

    Args:
        input_path: Input SVG file path
        output_path: Output SVG file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        updated_content = SVGValidator.add_viewbox_if_missing(content)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        logger.info(f"Added viewBox to {input_path} -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to add viewBox to file: {e}")
        return False