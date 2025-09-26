#!/usr/bin/env python3
"""
Tests for converter modules.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converters.vtracer_converter import VTracerConverter
from utils.quality_metrics import QualityMetrics
from utils.image_loader import ImageLoader


class TestVTracerConverter:
    """Test VTracer converter."""

    @pytest.fixture
    def converter(self):
        """Create converter instance."""
        return VTracerConverter()

    def test_converter_initialization(self, converter):
        """Test converter initializes correctly."""
        assert converter is not None
        assert hasattr(converter, 'convert_with_params')


class TestQualityMetrics:
    """Test quality metrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return QualityMetrics()

    def test_metrics_initialization(self, metrics):
        """Test metrics initializes correctly."""
        assert metrics is not None
        assert hasattr(metrics, 'calculate_ssim')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
