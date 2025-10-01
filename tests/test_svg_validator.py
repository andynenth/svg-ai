#!/usr/bin/env python3
"""
Comprehensive tests for SVG validation and processing utilities.
Tests SVG validation, sanitization, optimization, and file operations.
"""

import pytest
import numpy as np
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from xml.etree import ElementTree as ET

from backend.utils.svg_validator import SVGValidator, validate_svg_file, add_viewbox_to_file


class TestSVGValidator:
    """Comprehensive test suite for SVGValidator class"""

    @pytest.fixture
    def simple_svg(self):
        """Create a simple valid SVG"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="10" width="80" height="80" fill="#ff0000"/>
</svg>'''

    @pytest.fixture
    def svg_without_viewbox(self):
        """Create SVG without viewBox"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="150" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="75" r="50" fill="#00ff00"/>
</svg>'''

    @pytest.fixture
    def svg_with_viewbox(self):
        """Create SVG with existing viewBox"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="150" viewBox="0 0 200 150" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="75" r="50" fill="#00ff00"/>
</svg>'''

    @pytest.fixture
    def complex_svg(self):
        """Create complex SVG with multiple elements"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
  <path d="M10,10 L50,50 C75,25 100,50 125,25" stroke="#000"/>
  <rect x="10" y="10" width="80" height="80" fill="#ff0000"/>
  <circle cx="100" cy="100" r="30" fill="#00ff00"/>
  <polygon points="200,10 220,50 180,50" fill="#0000ff"/>
  <ellipse cx="250" cy="250" rx="30" ry="20" fill="#ffff00"/>
</svg>'''

    @pytest.fixture
    def svg_with_malicious_content(self):
        """Create SVG with potentially harmful content"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <script type="text/javascript">alert('XSS')</script>
  <rect x="10" y="10" width="80" height="80" fill="#ff0000" onclick="alert('click')" onmouseover="steal()"/>
  <image href="https://malicious.com/track.gif"/>
  <use href="javascript:alert('evil')"/>
</svg>'''

    @pytest.fixture
    def temp_svg_file(self, simple_svg):
        """Create temporary SVG file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
            tmp.write(simple_svg)
            tmp_path = tmp.name
        yield tmp_path
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def test_add_viewbox_if_missing_success(self, svg_without_viewbox):
        """Test adding viewBox to SVG without one"""
        result = SVGValidator.add_viewbox_if_missing(svg_without_viewbox)

        assert 'viewBox="0 0 200 150"' in result
        assert result != svg_without_viewbox

    def test_add_viewbox_if_missing_preserves_existing(self, svg_with_viewbox):
        """Test that existing viewBox is preserved"""
        result = SVGValidator.add_viewbox_if_missing(svg_with_viewbox)

        assert result == svg_with_viewbox
        assert result.count('viewBox') == 1

    def test_add_viewbox_if_missing_float_dimensions(self):
        """Test handling float dimensions"""
        svg_float = '''<svg width="100.5" height="200.75" xmlns="http://www.w3.org/2000/svg">
<rect fill="red"/>
</svg>'''

        result = SVGValidator.add_viewbox_if_missing(svg_float)
        assert 'viewBox="0 0 100.5 200.75"' in result

    def test_add_viewbox_if_missing_no_dimensions(self):
        """Test SVG without width/height"""
        svg_no_dims = '''<svg xmlns="http://www.w3.org/2000/svg">
<rect fill="red"/>
</svg>'''

        result = SVGValidator.add_viewbox_if_missing(svg_no_dims)
        assert result == svg_no_dims  # Should be unchanged

    def test_add_viewbox_if_missing_no_svg_tag(self):
        """Test content without SVG tag"""
        no_svg = '''<rect fill="red"/>'''

        result = SVGValidator.add_viewbox_if_missing(no_svg)
        assert result == no_svg

    def test_add_viewbox_if_missing_error_handling(self):
        """Test error handling in viewBox addition"""
        # Test with invalid content that might cause regex errors
        invalid_content = None

        with patch('backend.utils.svg_validator.logger') as mock_logger:
            result = SVGValidator.add_viewbox_if_missing(invalid_content)
            assert result == invalid_content
            mock_logger.error.assert_called()

    def test_validate_svg_structure_valid(self, simple_svg):
        """Test validation of valid SVG"""
        assert SVGValidator.validate_svg_structure(simple_svg) is True

    def test_validate_svg_structure_empty_content(self):
        """Test validation of empty/None content"""
        assert SVGValidator.validate_svg_structure("") is False
        assert SVGValidator.validate_svg_structure("   ") is False
        assert SVGValidator.validate_svg_structure(None) is False

    def test_validate_svg_structure_missing_svg_tag(self):
        """Test validation without SVG tag"""
        no_svg = '''<?xml version="1.0"?><rect fill="red"/>'''
        assert SVGValidator.validate_svg_structure(no_svg) is False

    def test_validate_svg_structure_missing_xmlns(self):
        """Test validation without xmlns"""
        no_xmlns = '''<svg width="100" height="100"><rect fill="red"/></svg>'''
        assert SVGValidator.validate_svg_structure(no_xmlns) is False

    def test_validate_svg_structure_malformed_xml(self):
        """Test validation of malformed XML"""
        malformed = '''<svg xmlns="http://www.w3.org/2000/svg"><rect><invalid</svg>'''
        assert SVGValidator.validate_svg_structure(malformed) is False

    def test_validate_svg_structure_error_handling(self):
        """Test error handling in SVG validation"""
        with patch('xml.etree.ElementTree.fromstring', side_effect=Exception("Parse error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                result = SVGValidator.validate_svg_structure('<svg xmlns="test">content</svg>')
                assert result is False
                mock_logger.error.assert_called()

    def test_extract_dimensions_success(self, simple_svg):
        """Test successful dimension extraction"""
        width, height = SVGValidator.extract_dimensions(simple_svg)
        assert width == 100.0
        assert height == 100.0

    def test_extract_dimensions_float_values(self):
        """Test extraction with float dimensions"""
        svg_float = '''<svg width="123.45" height="67.89" xmlns="test"/>'''
        width, height = SVGValidator.extract_dimensions(svg_float)
        assert width == 123.45
        assert height == 67.89

    def test_extract_dimensions_no_svg_tag(self):
        """Test extraction without SVG tag"""
        no_svg = '''<rect width="100" height="100"/>'''
        width, height = SVGValidator.extract_dimensions(no_svg)
        assert width is None
        assert height is None

    def test_extract_dimensions_partial_dimensions(self):
        """Test extraction with only one dimension"""
        only_width = '''<svg width="100" xmlns="test"/>'''
        width, height = SVGValidator.extract_dimensions(only_width)
        assert width == 100.0
        assert height is None

    def test_extract_dimensions_invalid_values(self):
        """Test extraction with invalid dimension values"""
        invalid_dims = '''<svg width="abc" height="def" xmlns="test"/>'''
        width, height = SVGValidator.extract_dimensions(invalid_dims)
        assert width is None
        assert height is None

    def test_extract_dimensions_error_handling(self):
        """Test error handling in dimension extraction"""
        with patch('re.search', side_effect=AttributeError("Regex error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                width, height = SVGValidator.extract_dimensions('<svg/>')
                assert width is None
                assert height is None
                mock_logger.error.assert_called()

    def test_sanitize_svg_content_removes_scripts(self, svg_with_malicious_content):
        """Test removal of script tags"""
        sanitized = SVGValidator.sanitize_svg_content(svg_with_malicious_content)

        assert '<script' not in sanitized
        assert 'alert(' not in sanitized

    def test_sanitize_svg_content_removes_event_handlers(self, svg_with_malicious_content):
        """Test removal of event handlers"""
        sanitized = SVGValidator.sanitize_svg_content(svg_with_malicious_content)

        assert 'onclick=' not in sanitized
        assert 'onmouseover=' not in sanitized

    def test_sanitize_svg_content_removes_external_refs(self, svg_with_malicious_content):
        """Test removal of external references"""
        sanitized = SVGValidator.sanitize_svg_content(svg_with_malicious_content)

        assert 'https://malicious.com' not in sanitized

    def test_sanitize_svg_content_removes_javascript_hrefs(self):
        """Test removal of javascript: hrefs"""
        js_svg = '''<svg><use href="javascript:alert('evil')"/></svg>'''
        sanitized = SVGValidator.sanitize_svg_content(js_svg)

        assert 'javascript:' not in sanitized

    def test_sanitize_svg_content_preserves_safe_content(self, simple_svg):
        """Test that safe content is preserved"""
        sanitized = SVGValidator.sanitize_svg_content(simple_svg)

        assert sanitized == simple_svg
        assert '<rect' in sanitized
        assert 'fill="#ff0000"' in sanitized

    def test_sanitize_svg_content_error_handling(self):
        """Test error handling in sanitization"""
        with patch('re.sub', side_effect=Exception("Regex error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                result = SVGValidator.sanitize_svg_content('<svg>content</svg>')
                assert result == '<svg>content</svg>'
                mock_logger.error.assert_called()

    def test_replace_fill_color_basic(self):
        """Test basic color replacement"""
        svg = '''<svg><rect fill="#000000"/><circle fill="#000000"/></svg>'''
        result = SVGValidator.replace_fill_color(svg, "#000000", "#ff0000")

        assert 'fill="#000000"' not in result
        assert result.count('fill="#ff0000"') == 2

    def test_replace_fill_color_case_insensitive(self):
        """Test case-insensitive replacement"""
        svg = '''<svg><rect fill="#000000"/><circle fill="#000000"/></svg>'''
        result = SVGValidator.replace_fill_color(svg, "#000000", "#ff0000")

        assert 'fill="#000000"' not in result
        assert 'fill="#ff0000"' in result

    def test_replace_fill_color_same_colors(self):
        """Test no change when colors are same"""
        svg = '''<svg><rect fill="#000000"/></svg>'''
        result = SVGValidator.replace_fill_color(svg, "#000000", "#000000")

        assert result == svg

    def test_replace_fill_color_no_matches(self):
        """Test replacement when no matches found"""
        svg = '''<svg><rect fill="#ff0000"/></svg>'''
        result = SVGValidator.replace_fill_color(svg, "#000000", "#00ff00")

        assert result == svg

    def test_replace_fill_color_error_handling(self):
        """Test error handling in color replacement"""
        with patch('backend.utils.svg_validator.logger') as mock_logger:
            # Test with None input that might cause string methods to fail
            result = SVGValidator.replace_fill_color(None, "#000", "#fff")
            assert result is None
            mock_logger.error.assert_called()

    def test_create_svg_header_basic(self):
        """Test basic SVG header creation"""
        header = SVGValidator.create_svg_header(100, 200)

        assert 'width="100"' in header
        assert 'height="200"' in header
        assert 'xmlns="http://www.w3.org/2000/svg"' in header
        assert 'viewBox="0 0 100 200"' in header
        assert '<?xml version="1.0"' in header

    def test_create_svg_header_without_xmlns(self):
        """Test header creation without xmlns"""
        header = SVGValidator.create_svg_header(100, 200, xmlns=False)

        assert 'xmlns=' not in header
        assert 'viewBox="0 0 100 200"' in header

    def test_create_svg_header_float_dimensions(self):
        """Test header with float dimensions"""
        header = SVGValidator.create_svg_header(100.5, 200.75)

        assert 'width="100.5"' in header
        assert 'height="200.75"' in header

    def test_create_svg_header_error_handling(self):
        """Test error handling in header creation"""
        with patch('backend.utils.svg_validator.logger') as mock_logger:
            # Force an exception
            with patch('builtins.str', side_effect=Exception("String error")):
                result = SVGValidator.create_svg_header(100, 200)
                assert 'width="100"' in result  # Should fallback
                mock_logger.error.assert_called()

    def test_get_svg_info_comprehensive(self, complex_svg):
        """Test comprehensive SVG info extraction"""
        info = SVGValidator.get_svg_info(complex_svg)

        assert info['width'] == 300.0
        assert info['height'] == 300.0
        assert info['has_viewbox'] is False
        assert info['has_xmlns'] is True
        assert info['is_valid'] is True
        assert info['path_count'] == 1
        assert info['rect_count'] == 1
        assert info['circle_count'] == 1
        assert info['size_bytes'] > 0
        assert 'total_elements' in info

    def test_get_svg_info_with_viewbox(self, svg_with_viewbox):
        """Test info extraction with viewBox"""
        info = SVGValidator.get_svg_info(svg_with_viewbox)

        assert info['has_viewbox'] is True
        assert info['circle_count'] == 1

    def test_get_svg_info_invalid_svg(self):
        """Test info extraction with invalid SVG"""
        invalid = '''<svg xmlns="test"><rect><invalid'''
        info = SVGValidator.get_svg_info(invalid)

        assert info['is_valid'] is False

    def test_get_svg_info_error_handling(self):
        """Test error handling in SVG info extraction"""
        with patch.object(SVGValidator, 'extract_dimensions', side_effect=Exception("Error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                info = SVGValidator.get_svg_info('<svg/>')
                assert 'error' in info
                mock_logger.error.assert_called()

    def test_optimize_svg_structure_adds_viewbox(self, svg_without_viewbox):
        """Test optimization adds viewBox"""
        optimized = SVGValidator.optimize_svg_structure(svg_without_viewbox)

        assert 'viewBox="0 0 200 150"' in optimized

    def test_optimize_svg_structure_removes_whitespace(self):
        """Test whitespace removal"""
        messy_svg = '''<svg xmlns="test">

        <rect fill="red"/>

        </svg>'''

        optimized = SVGValidator.optimize_svg_structure(messy_svg)

        assert '>\n        \n        <' not in optimized

    def test_optimize_svg_structure_removes_comments(self):
        """Test comment removal"""
        commented_svg = '''<svg xmlns="test">
        <!-- comment -->
        <rect fill="red"/>
        <!-- another comment -->
        </svg>'''

        optimized = SVGValidator.optimize_svg_structure(commented_svg)

        assert '<!--' not in optimized
        assert '-->' not in optimized

    def test_optimize_svg_structure_error_handling(self):
        """Test error handling in optimization"""
        with patch.object(SVGValidator, 'add_viewbox_if_missing', side_effect=Exception("Error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                result = SVGValidator.optimize_svg_structure('<svg/>')
                assert result == '<svg/>'
                mock_logger.error.assert_called()

    def test_validate_svg_file_success(self, temp_svg_file):
        """Test successful file validation"""
        assert validate_svg_file(temp_svg_file) is True

    def test_validate_svg_file_not_found(self):
        """Test validation of non-existent file"""
        assert validate_svg_file("nonexistent.svg") is False

    @patch('builtins.open', mock_open(read_data='<svg xmlns="test"><rect/></svg>'))
    def test_validate_svg_file_read_success(self):
        """Test successful file reading and validation"""
        with patch.object(SVGValidator, 'validate_svg_structure', return_value=True):
            result = validate_svg_file("test.svg")
            assert result is True

    def test_validate_svg_file_read_error(self):
        """Test file reading error"""
        with patch('builtins.open', side_effect=IOError("Read error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                result = validate_svg_file("test.svg")
                assert result is False
                mock_logger.error.assert_called()

    def test_add_viewbox_to_file_success(self, svg_without_viewbox):
        """Test adding viewBox to file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as input_tmp:
            input_tmp.write(svg_without_viewbox)
            input_path = input_tmp.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as output_tmp:
            output_path = output_tmp.name

        try:
            success = add_viewbox_to_file(input_path, output_path)
            assert success is True

            # Verify output
            with open(output_path, 'r') as f:
                content = f.read()
            assert 'viewBox="0 0 200 150"' in content

        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_add_viewbox_to_file_not_found(self):
        """Test adding viewBox to non-existent file"""
        assert add_viewbox_to_file("nonexistent.svg", "output.svg") is False

    def test_add_viewbox_to_file_read_error(self):
        """Test file read error"""
        with patch('builtins.open', side_effect=IOError("Read error")):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                result = add_viewbox_to_file("input.svg", "output.svg")
                assert result is False
                mock_logger.error.assert_called()

    def test_add_viewbox_to_file_write_error(self, temp_svg_file):
        """Test file write error"""
        with patch('builtins.open', side_effect=[mock_open(read_data='<svg/>').return_value, IOError("Write error")]):
            with patch('backend.utils.svg_validator.logger') as mock_logger:
                result = add_viewbox_to_file(temp_svg_file, "output.svg")
                assert result is False
                mock_logger.error.assert_called()

    def test_edge_cases_empty_svg_tag(self):
        """Test empty SVG tag handling"""
        empty_svg = '''<svg></svg>'''

        width, height = SVGValidator.extract_dimensions(empty_svg)
        assert width is None
        assert height is None

        assert SVGValidator.validate_svg_structure(empty_svg) is False

    def test_edge_cases_nested_svg_elements(self):
        """Test handling nested SVG elements"""
        nested_svg = '''<svg width="100" height="100" xmlns="test">
        <g>
            <svg width="50" height="50">
                <rect fill="red"/>
            </svg>
        </g>
        </svg>'''

        # Should extract dimensions from outer SVG only
        width, height = SVGValidator.extract_dimensions(nested_svg)
        assert width == 100.0
        assert height == 100.0

    def test_edge_cases_special_characters_in_svg(self):
        """Test SVG with special characters"""
        special_svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <text>Special chars: áéíóú ñ © ® ™ & < > "</text>
        </svg>'''

        assert SVGValidator.validate_svg_structure(special_svg) is True
        sanitized = SVGValidator.sanitize_svg_content(special_svg)
        assert 'Special chars:' in sanitized

    def test_edge_cases_very_large_svg(self):
        """Test handling of very large SVG content"""
        # Create large SVG content
        large_content = '<svg xmlns="test">' + '<rect/>' * 10000 + '</svg>'

        info = SVGValidator.get_svg_info(large_content)
        assert info['size_bytes'] > 100000
        assert info['rect_count'] == 10000

    def test_edge_cases_unicode_content(self):
        """Test SVG with Unicode content"""
        unicode_svg = '''<svg xmlns="http://www.w3.org/2000/svg">
        <text>Unicode: 🎨 🖼️ 🎭 中文 العربية русский</text>
        </svg>'''

        assert SVGValidator.validate_svg_structure(unicode_svg) is True
        info = SVGValidator.get_svg_info(unicode_svg)
        assert info['size_bytes'] > len(unicode_svg.encode('ascii', errors='ignore'))

    def test_logging_behavior(self, simple_svg):
        """Test that logging calls are made appropriately"""
        with patch('backend.utils.svg_validator.logger') as mock_logger:
            # Test debug logging
            SVGValidator.add_viewbox_if_missing(svg_with_viewbox)
            mock_logger.debug.assert_called()

            # Test warning logging
            SVGValidator.extract_dimensions('<svg/>')
            mock_logger.warning.assert_called()

    def test_memory_efficiency_large_svg(self):
        """Test memory efficiency with large SVG"""
        # Create moderately large SVG
        large_svg = '''<svg width="1000" height="1000" xmlns="test">''' + \
                   '''<rect fill="red"/>''' * 1000 + \
                   '''</svg>'''

        # Should handle without memory issues
        result = SVGValidator.optimize_svg_structure(large_svg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_concurrent_processing_safety(self, simple_svg):
        """Test thread safety of static methods"""
        import threading
        results = []

        def process_svg():
            result = SVGValidator.validate_svg_structure(simple_svg)
            results.append(result)

        # Run multiple threads
        threads = [threading.Thread(target=process_svg) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert all(results)
        assert len(results) == 10