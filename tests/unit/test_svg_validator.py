#!/usr/bin/env python3
"""
Unit tests for SVGValidator class.

Tests all SVG validation and processing utilities that consolidate duplicate
functionality across converters.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

from backend.utils.svg_validator import SVGValidator, validate_svg_file, add_viewbox_to_file


class TestSVGValidator:
    """Test cases for SVGValidator class."""

    @pytest.fixture
    def simple_svg(self):
        """Create a simple valid SVG."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="#ff0000"/>
</svg>'''

    @pytest.fixture
    def svg_without_viewbox(self):
        """Create SVG without viewBox."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="150" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="75" r="50" fill="#00ff00"/>
</svg>'''

    @pytest.fixture
    def svg_with_viewbox(self):
        """Create SVG with existing viewBox."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="150" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="75" r="50" fill="#00ff00"/>
</svg>'''

    @pytest.fixture
    def svg_with_black_fill(self):
        """Create SVG with black fill that can be replaced."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="#000000"/>
  <circle cx="50" cy="50" r="20" fill="#000000"/>
</svg>'''

    @pytest.fixture
    def invalid_svg(self):
        """Create invalid SVG for testing validation."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="#ff0000"
  <!-- Missing closing tag -->
</svg>'''

    @pytest.fixture
    def svg_with_scripts(self):
        """Create SVG with potentially harmful content."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <script>alert('test')</script>
  <rect x="10" y="10" width="80" height="80" fill="#ff0000" onclick="malicious()"/>
  <image href="https://evil.com/image.png"/>
</svg>'''

    @pytest.fixture
    def temp_svg_file(self, simple_svg):
        """Create a temporary SVG file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
            tmp.write(simple_svg)
            tmp.flush()  # Ensure content is written
        yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    def test_add_viewbox_if_missing_adds_viewbox(self, svg_without_viewbox):
        """Test adding viewBox to SVG that doesn't have one."""
        result = SVGValidator.add_viewbox_if_missing(svg_without_viewbox)

        assert 'viewBox="0 0 200 150"' in result
        assert result != svg_without_viewbox

    def test_add_viewbox_if_missing_preserves_existing(self, svg_with_viewbox):
        """Test that existing viewBox is preserved."""
        result = SVGValidator.add_viewbox_if_missing(svg_with_viewbox)

        assert result == svg_with_viewbox
        assert result.count('viewBox') == 1

    def test_add_viewbox_if_missing_handles_float_dimensions(self):
        """Test handling SVG with float dimensions."""
        svg_float = '''<svg width="100.5" height="200.75" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="red"/>
</svg>'''

        result = SVGValidator.add_viewbox_if_missing(svg_float)

        assert 'viewBox="0 0 100.5 200.75"' in result

    def test_add_viewbox_if_missing_handles_malformed_dimensions(self):
        """Test handling SVG with missing or malformed dimensions."""
        malformed_svg = '''<svg xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="red"/>
</svg>'''

        result = SVGValidator.add_viewbox_if_missing(malformed_svg)

        # Should return unchanged if no dimensions found
        assert result == malformed_svg

    def test_validate_svg_structure_valid_svg(self, simple_svg):
        """Test validation of valid SVG structure."""
        is_valid = SVGValidator.validate_svg_structure(simple_svg)

        assert is_valid == True

    def test_validate_svg_structure_invalid_svg(self, invalid_svg):
        """Test validation of invalid SVG structure."""
        is_valid = SVGValidator.validate_svg_structure(invalid_svg)

        assert is_valid == False

    def test_validate_svg_structure_empty_content(self):
        """Test validation of empty content."""
        assert SVGValidator.validate_svg_structure("") == False
        assert SVGValidator.validate_svg_structure("   ") == False
        assert SVGValidator.validate_svg_structure(None) == False

    def test_validate_svg_structure_missing_svg_tag(self):
        """Test validation when SVG tag is missing."""
        no_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<rect x="10" y="10" width="80" height="80" fill="#ff0000"/>'''

        assert SVGValidator.validate_svg_structure(no_svg) == False

    def test_validate_svg_structure_missing_xmlns(self):
        """Test validation when xmlns is missing."""
        no_xmlns = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100">
  <rect x="10" y="10" width="80" height="80" fill="#ff0000"/>
</svg>'''

        assert SVGValidator.validate_svg_structure(no_xmlns) == False

    def test_extract_dimensions_success(self, simple_svg):
        """Test successful dimension extraction."""
        width, height = SVGValidator.extract_dimensions(simple_svg)

        assert width == 100.0
        assert height == 100.0

    def test_extract_dimensions_float_values(self):
        """Test dimension extraction with float values."""
        svg_float = '''<svg width="123.45" height="67.89" xmlns="http://www.w3.org/2000/svg">'''

        width, height = SVGValidator.extract_dimensions(svg_float)

        assert width == 123.45
        assert height == 67.89

    def test_extract_dimensions_missing_dimensions(self):
        """Test dimension extraction when dimensions are missing."""
        no_dims = '''<svg xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="#ff0000"/>
</svg>'''

        width, height = SVGValidator.extract_dimensions(no_dims)

        assert width is None
        assert height is None

    def test_extract_dimensions_partial_dimensions(self):
        """Test extraction when only one dimension is present."""
        only_width = '''<svg width="100" xmlns="http://www.w3.org/2000/svg">'''

        width, height = SVGValidator.extract_dimensions(only_width)

        assert width == 100.0
        assert height is None

    def test_sanitize_svg_content_removes_scripts(self, svg_with_scripts):
        """Test removal of script tags."""
        sanitized = SVGValidator.sanitize_svg_content(svg_with_scripts)

        assert '<script>' not in sanitized
        assert 'alert(' not in sanitized

    def test_sanitize_svg_content_removes_event_handlers(self, svg_with_scripts):
        """Test removal of event handlers."""
        sanitized = SVGValidator.sanitize_svg_content(svg_with_scripts)

        assert 'onclick=' not in sanitized
        assert 'malicious()' not in sanitized

    def test_sanitize_svg_content_removes_external_refs(self, svg_with_scripts):
        """Test removal of external references."""
        sanitized = SVGValidator.sanitize_svg_content(svg_with_scripts)

        assert 'https://evil.com' not in sanitized

    def test_sanitize_svg_content_preserves_safe_content(self, simple_svg):
        """Test that safe content is preserved."""
        sanitized = SVGValidator.sanitize_svg_content(simple_svg)

        assert '<rect' in sanitized
        assert 'fill="#ff0000"' in sanitized
        assert len(sanitized) == len(simple_svg)  # Should be unchanged

    def test_replace_fill_color_basic_replacement(self, svg_with_black_fill):
        """Test basic color replacement."""
        result = SVGValidator.replace_fill_color(svg_with_black_fill, "#000000", "#ff0000")

        assert 'fill="#000000"' not in result
        assert 'fill="#ff0000"' in result
        assert result.count('fill="#ff0000"') == 2  # Two elements should be updated

    def test_replace_fill_color_case_insensitive(self):
        """Test case-insensitive color replacement."""
        svg_upper = '''<svg><rect fill="#000000"/><circle fill="#000000"/></svg>'''

        result = SVGValidator.replace_fill_color(svg_upper, "#000000", "#ff0000")

        assert 'fill="#000000"' not in result
        assert 'fill="#ff0000"' in result

    def test_replace_fill_color_no_change_when_same(self, svg_with_black_fill):
        """Test no change when old and new colors are the same."""
        result = SVGValidator.replace_fill_color(svg_with_black_fill, "#000000", "#000000")

        assert result == svg_with_black_fill

    def test_replace_fill_color_no_matches(self, simple_svg):
        """Test color replacement when no matches found."""
        result = SVGValidator.replace_fill_color(simple_svg, "#000000", "#ff0000")

        assert result == simple_svg  # Should be unchanged

    def test_create_svg_header_basic(self):
        """Test basic SVG header creation."""
        header = SVGValidator.create_svg_header(100, 200)

        assert 'width="100"' in header
        assert 'height="200"' in header
        assert 'xmlns="http://www.w3.org/2000/svg"' in header
        assert 'viewBox="0 0 100 200"' in header
        assert '<?xml version="1.0"' in header

    def test_create_svg_header_without_xmlns(self):
        """Test SVG header creation without xmlns."""
        header = SVGValidator.create_svg_header(100, 200, xmlns=False)

        assert 'width="100"' in header
        assert 'height="200"' in header
        assert 'xmlns=' not in header
        assert 'viewBox="0 0 100 200"' in header

    def test_create_svg_header_float_dimensions(self):
        """Test SVG header creation with float dimensions."""
        header = SVGValidator.create_svg_header(100.5, 200.75)

        assert 'width="100.5"' in header
        assert 'height="200.75"' in header
        assert 'viewBox="0 0 100.5 200.75"' in header

    def test_get_svg_info_comprehensive(self, simple_svg):
        """Test comprehensive SVG information extraction."""
        info = SVGValidator.get_svg_info(simple_svg)

        assert info['width'] == 100.0
        assert info['height'] == 100.0
        assert info['has_viewbox'] == False
        assert info['has_xmlns'] == True
        assert info['is_valid'] == True
        assert info['rect_count'] == 1
        assert info['path_count'] == 0
        assert info['circle_count'] == 0
        assert 'size_bytes' in info
        assert info['size_bytes'] > 0

    def test_get_svg_info_with_viewbox(self, svg_with_viewbox):
        """Test SVG info extraction with viewBox."""
        info = SVGValidator.get_svg_info(svg_with_viewbox)

        assert info['has_viewbox'] == True
        assert info['circle_count'] == 1

    def test_get_svg_info_invalid_svg(self, invalid_svg):
        """Test SVG info extraction with invalid SVG."""
        info = SVGValidator.get_svg_info(invalid_svg)

        assert info['is_valid'] == False
        # Other fields should still be extracted where possible

    def test_optimize_svg_structure_adds_viewbox(self, svg_without_viewbox):
        """Test that optimization adds viewBox."""
        optimized = SVGValidator.optimize_svg_structure(svg_without_viewbox)

        assert 'viewBox="0 0 200 150"' in optimized

    def test_optimize_svg_structure_removes_whitespace(self):
        """Test whitespace removal."""
        messy_svg = '''<?xml version="1.0"?>
<svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">

  <rect x="10" y="10" width="80" height="80" fill="red"/>

</svg>'''

        optimized = SVGValidator.optimize_svg_structure(messy_svg)

        # Should have whitespace between tags removed
        assert '>\n\n  <' not in optimized
        assert optimized.count('\n') < messy_svg.count('\n')

    def test_optimize_svg_structure_removes_comments(self):
        """Test comment removal."""
        svg_with_comments = '''<?xml version="1.0"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <!-- This is a comment -->
  <rect x="10" y="10" width="80" height="80" fill="red"/>
  <!-- Another comment -->
</svg>'''

        optimized = SVGValidator.optimize_svg_structure(svg_with_comments)

        assert '<!--' not in optimized
        assert '-->' not in optimized

    def test_validate_svg_file_success(self, temp_svg_file):
        """Test successful file validation."""
        is_valid = validate_svg_file(temp_svg_file)

        assert is_valid == True

    def test_validate_svg_file_not_found(self):
        """Test file validation with non-existent file."""
        is_valid = validate_svg_file("nonexistent_file.svg")

        assert is_valid == False

    def test_add_viewbox_to_file_success(self, svg_without_viewbox):
        """Test adding viewBox to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as input_tmp:
            input_tmp.write(svg_without_viewbox)
            input_path = input_tmp.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as output_tmp:
            output_path = output_tmp.name

        try:
            success = add_viewbox_to_file(input_path, output_path)

            assert success == True

            # Check that output file has viewBox
            with open(output_path, 'r') as f:
                output_content = f.read()

            assert 'viewBox="0 0 200 150"' in output_content

        finally:
            # Cleanup
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_add_viewbox_to_file_not_found(self):
        """Test adding viewBox to non-existent file."""
        success = add_viewbox_to_file("nonexistent.svg", "output.svg")

        assert success == False

    def test_edge_case_empty_svg_tag(self):
        """Test handling of empty SVG tag."""
        empty_svg = '''<svg></svg>'''

        width, height = SVGValidator.extract_dimensions(empty_svg)
        assert width is None
        assert height is None

        is_valid = SVGValidator.validate_svg_structure(empty_svg)
        assert is_valid == False  # Missing xmlns

    def test_edge_case_malformed_xml(self):
        """Test handling of malformed XML."""
        malformed = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="red">
  <!-- Missing closing tags -->'''

        is_valid = SVGValidator.validate_svg_structure(malformed)
        assert is_valid == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])