#!/usr/bin/env python3
"""
Unit tests for BaseConverter abstract class.

Tests the base converter functionality that all converters inherit.
"""

import pytest
import sys
from pathlib import Path

from backend.converters.base import BaseConverter


class MockConverter(BaseConverter):
    """Mock converter for testing base functionality."""

    def __init__(self):
        super().__init__(name="Mock Converter")

    def get_name(self) -> str:
        return "Mock Converter"

    def convert(self, image_path: str, **kwargs) -> str:
        # Simple mock implementation
        return f'<svg><rect width="100" height="100" fill="red"/></svg>'


class TestBaseConverter:
    """Test cases for BaseConverter abstract class."""

    def test_base_converter_initialization(self):
        """Test that base converter can be initialized properly."""
        converter = MockConverter()
        assert converter.get_name() == "Mock Converter"

    def test_abstract_methods_enforced(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because BaseConverter is abstract
            BaseConverter()

    def test_convert_method_signature(self):
        """Test that convert method has correct signature."""
        converter = MockConverter()
        result = converter.convert("test_path.png")
        assert isinstance(result, str)
        assert "<svg>" in result

    def test_convert_with_kwargs(self):
        """Test that convert method accepts keyword arguments."""
        converter = MockConverter()
        result = converter.convert("test_path.png", threshold=128, quality="high")
        assert isinstance(result, str)

    def test_get_name_method(self):
        """Test that get_name method returns string."""
        converter = MockConverter()
        name = converter.get_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_converter_inheritance(self):
        """Test that mock converter properly inherits from BaseConverter."""
        converter = MockConverter()
        assert isinstance(converter, BaseConverter)

    def test_converter_name_attribute(self):
        """Test that converter name is properly stored."""
        converter = MockConverter()
        # Access the name through get_name method
        assert converter.get_name() == "Mock Converter"

    def test_convert_returns_svg_string(self):
        """Test that convert method returns SVG string."""
        converter = MockConverter()
        svg_result = converter.convert("dummy.png")

        # Should return SVG content
        assert isinstance(svg_result, str)
        assert svg_result.startswith('<svg')
        assert svg_result.endswith('</svg>')

    def test_convert_parameter_handling(self):
        """Test that convert method handles various parameters."""
        converter = MockConverter()

        # Test with no parameters
        result1 = converter.convert("test.png")
        assert isinstance(result1, str)

        # Test with various parameter types
        result2 = converter.convert(
            "test.png",
            threshold=128,
            quality=0.8,
            option="value",
            flag=True
        )
        assert isinstance(result2, str)

        # Results should be consistent for same input
        result3 = converter.convert("test.png")
        assert result1 == result3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])