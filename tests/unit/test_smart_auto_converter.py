#!/usr/bin/env python3
"""
Unit tests for SmartAutoConverter class.

Tests intelligent routing and parameter optimization functionality.
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.converters.base import BaseConverter
from backend.converters.smart_auto_converter import SmartAutoConverter


class TestSmartAutoConverter:
    """Test cases for SmartAutoConverter class."""

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_smart_auto_initialization(self, mock_potrace, mock_vtracer, mock_detector):
        """Test that Smart Auto converter initializes properly."""
        converter = SmartAutoConverter()
        assert converter.get_name() == "Smart Auto"
        assert hasattr(converter, 'color_detector')
        assert hasattr(converter, 'vtracer_converter')
        assert hasattr(converter, 'smart_potrace_converter')
        assert hasattr(converter, 'routing_stats')

    @patch('converters.smart_auto_converter.ColorDetector')
    def test_smart_auto_initialization_color_detector_fail(self, mock_detector):
        """Test initialization when color detector fails."""
        mock_detector.side_effect = ImportError("Color detector failed")

        with pytest.raises(ImportError):
            SmartAutoConverter()

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_smart_auto_inheritance(self, mock_potrace, mock_vtracer, mock_detector):
        """Test that Smart Auto converter properly inherits from BaseConverter."""
        converter = SmartAutoConverter()
        assert isinstance(converter, BaseConverter)

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_convert_routes_to_vtracer(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test conversion routing to VTracer for colored images."""
        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.analyze_image.return_value = {
            'is_colored': True,
            'recommended_converter': 'vtracer',
            'confidence': 0.95,
            'unique_colors': 50,
            'has_gradients': False
        }
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer.convert.return_value = '<svg>vtracer result</svg>'
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()
        result = converter.convert("test.png")

        # Should route to VTracer
        mock_vtracer.convert.assert_called_once()
        mock_potrace.convert.assert_not_called()
        assert 'vtracer result' in result

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_convert_routes_to_potrace(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test conversion routing to Smart Potrace for B&W images."""
        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.analyze_image.return_value = {
            'is_colored': False,
            'recommended_converter': 'potrace',
            'confidence': 0.88,
            'unique_colors': 2,
            'has_gradients': False
        }
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace.convert.return_value = '<svg>potrace result</svg>'
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()
        result = converter.convert("test.png")

        # Should route to Smart Potrace
        mock_potrace.convert.assert_called_once()
        mock_vtracer.convert.assert_not_called()
        assert 'potrace result' in result

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_convert_with_analysis_error_fallback(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test fallback when color analysis fails."""
        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.analyze_image.side_effect = Exception("Analysis failed")
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer.convert.return_value = '<svg>fallback result</svg>'
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()
        result = converter.convert("test.png")

        # Should fallback to VTracer
        mock_vtracer.convert.assert_called_once()
        assert 'fallback result' in result

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_vtracer_params_few_colors(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test VTracer parameter optimization for few colors."""
        converter = SmartAutoConverter()

        analysis = {
            'unique_colors': 5,
            'has_gradients': False
        }

        params = converter._get_vtracer_params(analysis)

        # Should optimize for few colors
        assert params['color_precision'] == 3
        assert params['layer_difference'] == 32

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_vtracer_params_many_colors(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test VTracer parameter optimization for many colors."""
        converter = SmartAutoConverter()

        analysis = {
            'unique_colors': 150,
            'has_gradients': False
        }

        params = converter._get_vtracer_params(analysis)

        # Should optimize for many colors
        assert params['color_precision'] == 8
        assert params['layer_difference'] == 8

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_vtracer_params_gradients(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test VTracer parameter optimization for gradients."""
        converter = SmartAutoConverter()

        analysis = {
            'unique_colors': 50,
            'has_gradients': True
        }

        params = converter._get_vtracer_params(analysis)

        # Should optimize for gradients
        assert params['color_precision'] == 8
        assert params['layer_difference'] == 8
        assert params['path_precision'] == 6

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_vtracer_params_user_override(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test that user parameters override optimized VTracer parameters."""
        converter = SmartAutoConverter()

        analysis = {'unique_colors': 5}
        user_params = {'color_precision': 10, 'corner_threshold': 30}

        params = converter._get_vtracer_params(analysis, **user_params)

        # User parameters should override
        assert params['color_precision'] == 10  # User override
        assert params['corner_threshold'] == 30  # User override
        assert params['layer_difference'] == 32  # Optimized default

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_potrace_params_pure_bw(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test Smart Potrace parameter optimization for pure B&W."""
        converter = SmartAutoConverter()

        analysis = {'unique_colors': 2}

        params = converter._get_potrace_params(analysis)

        # Should optimize for pure B&W
        assert params['turdsize'] == 1
        assert params['opttolerance'] == 0.1
        assert params['turnpolicy'] == 'black'

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_potrace_params_antialiased_bw(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test Smart Potrace parameter optimization for antialiased B&W."""
        converter = SmartAutoConverter()

        analysis = {'unique_colors': 8}

        params = converter._get_potrace_params(analysis)

        # Should optimize for antialiased B&W
        assert params['turdsize'] == 2
        assert params['opttolerance'] == 0.15

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_potrace_params_complex_grayscale(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test Smart Potrace parameter optimization for complex grayscale."""
        converter = SmartAutoConverter()

        analysis = {'unique_colors': 50}

        params = converter._get_potrace_params(analysis)

        # Should optimize for complex grayscale
        assert params['turdsize'] == 3
        assert params['opttolerance'] == 0.25

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_create_metadata_comment(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test metadata comment creation."""
        converter = SmartAutoConverter()

        analysis = {
            'is_colored': True,
            'confidence': 0.85,
            'unique_colors': 25
        }
        decision_data = {'analysis_time': 0.002}

        metadata = converter._create_metadata_comment(
            analysis, "VTracer", decision_data, 0.15
        )

        assert "Smart Auto Converter Metadata" in metadata
        assert "VTracer" in metadata
        assert "Colored" in metadata
        assert "85.00%" in metadata
        assert "25" in metadata

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_add_metadata_to_svg(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test adding metadata to SVG content."""
        converter = SmartAutoConverter()

        svg_content = '<?xml version="1.0"?>\n<svg>content</svg>'
        metadata = '<!-- Test metadata -->'

        result = converter._add_metadata_to_svg(svg_content, metadata)

        lines = result.split('\n')
        assert lines[0] == '<?xml version="1.0"?>'
        assert metadata in lines[1]
        assert '<svg>content</svg>' in result

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_routing_stats_tracking(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test routing statistics tracking."""
        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.analyze_image.return_value = {
            'is_colored': True,
            'recommended_converter': 'vtracer',
            'confidence': 0.90,
            'unique_colors': 30
        }
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer.convert.return_value = '<svg>test</svg>'
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()

        # Perform conversion
        converter.convert("test.png")

        # Check stats
        stats = converter.get_routing_stats()
        assert stats['total_conversions'] == 1
        assert stats['vtracer_routes'] == 1
        assert stats['potrace_routes'] == 0
        assert stats['vtracer_percentage'] == 100.0
        assert len(stats['decisions']) == 1

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_get_routing_stats_empty(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test routing statistics when no conversions performed."""
        converter = SmartAutoConverter()

        stats = converter.get_routing_stats()
        assert stats['total_conversions'] == 0
        assert stats['vtracer_percentage'] == 0
        assert stats['potrace_percentage'] == 0
        assert stats['average_confidence'] == 0
        assert stats['average_analysis_time'] == 0

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_convert_with_analysis(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test convert_with_analysis method."""
        # Setup mocks
        mock_detector = MagicMock()
        analysis_result = {
            'is_colored': False,
            'recommended_converter': 'potrace',
            'confidence': 0.75,
            'unique_colors': 5
        }
        mock_detector.analyze_image.return_value = analysis_result
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace.convert.return_value = '<svg>analysis test</svg>'
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()
        result = converter.convert_with_analysis("test.png")

        assert result['success'] == True
        assert 'analysis test' in result['svg']
        assert result['routing_analysis'] == analysis_result
        assert result['routed_to'] == 'potrace'
        assert result['size'] > 0
        assert 'total_time' in result

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_convert_primary_fails_fallback_succeeds(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test fallback when primary converter fails."""
        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.analyze_image.return_value = {
            'is_colored': True,
            'recommended_converter': 'vtracer',
            'confidence': 0.95,
            'unique_colors': 50
        }
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer.convert.side_effect = Exception("VTracer failed")
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace.convert.return_value = '<svg>fallback success</svg>'
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()
        result = converter.convert("test.png")

        # Should fallback to Smart Potrace
        mock_vtracer.convert.assert_called_once()
        mock_potrace.convert.assert_called_once()
        assert 'fallback success' in result

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_convert_both_converters_fail(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test when both primary and fallback converters fail."""
        # Setup mocks
        mock_detector = MagicMock()
        mock_detector.analyze_image.return_value = {
            'is_colored': True,
            'recommended_converter': 'vtracer',
            'confidence': 0.95,
            'unique_colors': 50
        }
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer.convert.side_effect = Exception("VTracer failed")
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace.convert.side_effect = Exception("Potrace failed")
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()

        with pytest.raises(Exception) as exc_info:
            converter.convert("test.png")

        assert "Both converters failed" in str(exc_info.value)

    @patch('converters.smart_auto_converter.ColorDetector')
    @patch('converters.smart_auto_converter.VTracerConverter')
    @patch('converters.smart_auto_converter.SmartPotraceConverter')
    def test_routing_stats_multiple_conversions(self, mock_potrace_class, mock_vtracer_class, mock_detector_class):
        """Test routing statistics with multiple conversions."""
        # Setup mocks for mixed routing
        mock_detector = MagicMock()
        analysis_results = [
            {'is_colored': True, 'recommended_converter': 'vtracer', 'confidence': 0.9, 'unique_colors': 50},
            {'is_colored': False, 'recommended_converter': 'potrace', 'confidence': 0.8, 'unique_colors': 3},
            {'is_colored': True, 'recommended_converter': 'vtracer', 'confidence': 0.7, 'unique_colors': 25}
        ]
        mock_detector.analyze_image.side_effect = analysis_results
        mock_detector_class.return_value = mock_detector

        mock_vtracer = MagicMock()
        mock_vtracer.convert.return_value = '<svg>vtracer</svg>'
        mock_vtracer_class.return_value = mock_vtracer

        mock_potrace = MagicMock()
        mock_potrace.convert.return_value = '<svg>potrace</svg>'
        mock_potrace_class.return_value = mock_potrace

        converter = SmartAutoConverter()

        # Perform multiple conversions
        converter.convert("test1.png")
        converter.convert("test2.png")
        converter.convert("test3.png")

        stats = converter.get_routing_stats()
        assert stats['total_conversions'] == 3
        assert stats['vtracer_routes'] == 2
        assert stats['potrace_routes'] == 1
        assert abs(stats['vtracer_percentage'] - 66.7) < 0.1
        assert abs(stats['potrace_percentage'] - 33.3) < 0.1
        assert abs(stats['average_confidence'] - 0.8) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])