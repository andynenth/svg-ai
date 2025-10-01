#!/usr/bin/env python3
"""
Comprehensive tests for AIEnhancedConverter class.
Tests AI-enhanced SVG conversion with parameter optimization, caching, and error handling.
"""

import pytest
import numpy as np
import tempfile
import os
import time
import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from backend.converters.ai_enhanced_converter import AIEnhancedConverter


class TestAIEnhancedConverter:
    """Comprehensive test suite for AIEnhancedConverter class"""

    @pytest.fixture
    def converter(self):
        """Create AI enhanced converter instance for testing"""
        return AIEnhancedConverter()

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing"""
        return {
            "edge_density": 0.4,
            "unique_colors": 0.3,
            "entropy": 0.6,
            "corner_density": 0.2,
            "gradient_strength": 0.3,
            "complexity_score": 0.5
        }

    @pytest.fixture
    def sample_optimization_result(self):
        """Create sample optimization result"""
        return {
            "parameters": {
                "colormode": "color",
                "color_precision": 6,
                "layer_difference": 16,
                "path_precision": 5,
                "corner_threshold": 60,
                "length_threshold": 5.0,
                "max_iterations": 10,
                "splice_threshold": 45
            },
            "confidence": 0.8,
            "method": "method_1_correlation",
            "logo_type": "simple"
        }

    @pytest.fixture
    def temp_image_file(self):
        """Create temporary image file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create minimal PNG data
            tmp.write(b'\x89PNG\r\n\x1a\n' + b'fake_png_data')
            tmp_path = tmp.name
        yield tmp_path
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def test_initialization(self, converter):
        """Test converter initialization"""
        assert isinstance(converter, AIEnhancedConverter)
        assert hasattr(converter, 'optimizer')
        assert hasattr(converter, 'feature_extractor')
        assert hasattr(converter, 'error_handler')
        assert hasattr(converter, 'quality_metrics')
        assert hasattr(converter, 'performance_optimizer')

        # Check caches are initialized
        assert converter.optimization_cache == {}
        assert converter.feature_cache == {}
        assert converter.cache_hits == 0
        assert converter.cache_misses == 0

        # Check configuration
        assert converter.config['enable_ai_optimization'] is True
        assert converter.config['enable_caching'] is True
        assert converter.config['cache_max_size'] == 1000

    def test_get_name(self, converter):
        """Test converter name retrieval"""
        name = converter.get_name()
        assert isinstance(name, str)
        assert "AI-Enhanced Converter" in name

    @patch('backend.converters.ai_enhanced_converter.Path.exists')
    @patch('backend.converters.ai_enhanced_converter.vtracer.convert_image_to_svg_py')
    @patch('builtins.open', mock_open(read_data='<svg>test content</svg>'))
    def test_convert_with_explicit_parameters(self, mock_vtracer, mock_exists, converter, temp_image_file):
        """Test conversion with explicitly provided parameters"""
        mock_exists.return_value = True

        # Mock feature extraction
        with patch.object(converter, '_get_features_with_cache') as mock_features:
            mock_features.return_value = {"edge_density": 0.5}

            # Test with explicit parameters
            explicit_params = {"color_precision": 8, "corner_threshold": 30}
            result = converter.convert(temp_image_file, parameters=explicit_params)

            assert isinstance(result, str)
            assert 'test content' in result
            mock_vtracer.assert_called_once()

    @patch('backend.converters.ai_enhanced_converter.Path.exists')
    def test_convert_file_not_found(self, mock_exists, converter):
        """Test conversion with non-existent file"""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            converter.convert("nonexistent.png")

    @patch('backend.converters.ai_enhanced_converter.Path.exists')
    def test_convert_with_ai_optimization(self, mock_exists, converter, temp_image_file, sample_features, sample_optimization_result):
        """Test conversion with AI optimization"""
        mock_exists.return_value = True

        # Mock dependencies
        with patch.object(converter, '_get_features_with_cache') as mock_features, \
             patch.object(converter, '_get_optimization_with_cache') as mock_optimization, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert, \
             patch.object(converter, '_track_conversion_result') as mock_track:

            mock_features.return_value = sample_features
            mock_optimization.return_value = sample_optimization_result
            mock_convert.return_value = '<svg>optimized content</svg>'

            result = converter.convert(temp_image_file)

            assert result == '<svg>optimized content</svg>'
            mock_features.assert_called_once()
            mock_optimization.assert_called_once()
            mock_convert.assert_called_once()
            mock_track.assert_called_once()

    def test_convert_with_ai_tier_success(self, converter, temp_image_file, sample_features, sample_optimization_result):
        """Test AI tier conversion success"""
        with patch('backend.converters.ai_enhanced_converter.Path.exists', return_value=True), \
             patch.object(converter, '_get_features_with_cache') as mock_features, \
             patch.object(converter, '_get_tier_optimization') as mock_tier_opt, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert, \
             patch.object(converter, '_track_conversion_result') as mock_track:

            mock_features.return_value = sample_features
            mock_tier_opt.return_value = sample_optimization_result
            mock_convert.return_value = '<svg>tier content</svg>'

            result = converter.convert_with_ai_tier(temp_image_file, tier=2)

            assert result['success'] is True
            assert result['svg'] == '<svg>tier content</svg>'
            assert result['tier_used'] == 2
            assert 'processing_time' in result
            assert result['predicted_quality'] is None  # No quality predictor mocked

    def test_convert_with_ai_tier_with_metadata(self, converter, temp_image_file, sample_features, sample_optimization_result):
        """Test AI tier conversion with metadata"""
        with patch('backend.converters.ai_enhanced_converter.Path.exists', return_value=True), \
             patch.object(converter, '_get_features_with_cache') as mock_features, \
             patch.object(converter, '_get_tier_optimization') as mock_tier_opt, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert, \
             patch.object(converter.quality_metrics, 'calculate_comprehensive_metrics') as mock_quality:

            mock_features.return_value = sample_features
            mock_tier_opt.return_value = sample_optimization_result
            mock_convert.return_value = '<svg>content</svg>'
            mock_quality.return_value = {'ssim': 0.85}

            result = converter.convert_with_ai_tier(temp_image_file, tier=1, include_metadata=True)

            assert result['success'] is True
            assert result['actual_quality'] == 0.85
            assert 'metadata' in result
            assert result['metadata']['features'] == sample_features
            assert result['metadata']['tier'] == 1

    def test_convert_with_ai_tier_failure(self, converter):
        """Test AI tier conversion failure"""
        with patch('backend.converters.ai_enhanced_converter.Path.exists', return_value=False):
            result = converter.convert_with_ai_tier("nonexistent.png", tier=1)

            assert result['success'] is False
            assert 'error' in result
            assert result['tier_used'] == 1
            assert result['svg'] is None

    def test_get_tier_optimization_tier1(self, converter, sample_features, sample_optimization_result):
        """Test tier 1 optimization"""
        with patch.object(converter, '_get_optimization_with_cache') as mock_opt:
            mock_opt.return_value = sample_optimization_result

            result = converter._get_tier_optimization(sample_features, "test.png", 1)

            assert result == sample_optimization_result
            mock_opt.assert_called_once()

    def test_get_tier_optimization_tier2(self, converter, sample_features, sample_optimization_result):
        """Test tier 2 optimization with quality prediction"""
        enhanced_result = sample_optimization_result.copy()
        enhanced_result['method'] = 'Method 1 + Quality Prediction'

        with patch.object(converter, '_get_optimization_with_cache') as mock_opt, \
             patch.object(converter, '_enhance_with_quality_prediction') as mock_enhance:

            mock_opt.return_value = sample_optimization_result
            mock_enhance.return_value = enhanced_result

            result = converter._get_tier_optimization(sample_features, "test.png", 2)

            assert result['method'] == 'Method 1 + Quality Prediction'
            mock_enhance.assert_called_once()

    def test_get_tier_optimization_tier3(self, converter, sample_features, sample_optimization_result):
        """Test tier 3 full optimization"""
        with patch.object(converter, '_get_full_optimization') as mock_full:
            mock_full.return_value = sample_optimization_result

            result = converter._get_tier_optimization(sample_features, "test.png", 3)

            mock_full.assert_called_once()

    def test_get_tier_optimization_invalid_tier(self, converter, sample_features):
        """Test invalid tier fallback"""
        with patch.object(converter, '_get_optimization_with_cache') as mock_opt:
            mock_opt.return_value = {"parameters": {}, "confidence": 0.0}

            result = converter._get_tier_optimization(sample_features, "test.png", 99)

            mock_opt.assert_called_once()

    def test_enhance_with_quality_prediction(self, converter, sample_features, sample_optimization_result):
        """Test quality prediction enhancement"""
        # High complexity features
        high_complexity_features = sample_features.copy()
        high_complexity_features['complexity_score'] = 0.8

        result = converter._enhance_with_quality_prediction(
            sample_optimization_result, high_complexity_features, "test.png"
        )

        assert result['method'] == 'Method 1 + Quality Prediction'
        assert result['confidence'] >= sample_optimization_result['confidence']

        # Check parameter adjustments for high complexity
        assert result['parameters']['max_iterations'] > sample_optimization_result['parameters']['max_iterations']

    def test_enhance_with_quality_prediction_low_complexity(self, converter, sample_features, sample_optimization_result):
        """Test quality prediction with low complexity"""
        # Low complexity features
        low_complexity_features = sample_features.copy()
        low_complexity_features['complexity_score'] = 0.2

        result = converter._enhance_with_quality_prediction(
            sample_optimization_result, low_complexity_features, "test.png"
        )

        # Should optimize for speed with lower iterations
        assert result['parameters']['max_iterations'] < sample_optimization_result['parameters']['max_iterations']

    def test_get_full_optimization(self, converter, sample_features, sample_optimization_result):
        """Test full optimization method"""
        with patch.object(converter, '_get_optimization_with_cache') as mock_opt, \
             patch.object(converter, '_enhance_with_quality_prediction') as mock_enhance:

            mock_opt.return_value = sample_optimization_result
            mock_enhance.return_value = sample_optimization_result

            result = converter._get_full_optimization(sample_features, "test.png")

            assert result['method'] == 'Full Optimization (Method 1+2+3)'
            assert result['confidence'] >= sample_optimization_result['confidence']

    def test_get_features_with_cache_miss(self, converter, sample_features):
        """Test feature extraction with cache miss"""
        with patch.object(converter.feature_extractor, 'extract_features') as mock_extract, \
             patch('backend.converters.ai_enhanced_converter.Path') as mock_path:

            mock_extract.return_value = sample_features
            mock_path.return_value.stat.return_value.st_mtime = 123456789

            result = converter._get_features_with_cache("test.png")

            assert result == sample_features
            assert converter.cache_misses == 1
            mock_extract.assert_called_once()

    def test_get_features_with_cache_hit(self, converter, sample_features):
        """Test feature extraction with cache hit"""
        # Pre-populate cache
        cache_key = converter._generate_cache_key("test.png", "123456789")
        converter.feature_cache[cache_key] = sample_features

        with patch('backend.converters.ai_enhanced_converter.Path') as mock_path:
            mock_path.return_value.stat.return_value.st_mtime = 123456789

            result = converter._get_features_with_cache("test.png")

            assert result == sample_features
            assert converter.cache_hits == 1

    def test_get_features_with_cache_extraction_error(self, converter):
        """Test feature extraction error handling"""
        with patch.object(converter.feature_extractor, 'extract_features', side_effect=Exception("Extraction failed")), \
             patch.object(converter.error_handler, 'detect_error') as mock_detect, \
             patch.object(converter.error_handler, 'attempt_recovery') as mock_recovery, \
             patch.object(converter, '_get_default_features') as mock_default, \
             patch('backend.converters.ai_enhanced_converter.Path') as mock_path:

            mock_path.return_value.stat.return_value.st_mtime = 123456789
            mock_detect.return_value = {"type": "feature_extraction_error"}
            mock_recovery.return_value = {"success": True}
            mock_default.return_value = {"edge_density": 0.3}

            result = converter._get_features_with_cache("test.png")

            assert result == {"edge_density": 0.3}
            mock_recovery.assert_called_once()

    def test_get_optimization_with_cache_ai_disabled(self, converter, sample_features):
        """Test optimization when AI is disabled"""
        converter.config['enable_ai_optimization'] = False

        result = converter._get_optimization_with_cache(sample_features, "test.png")

        assert result['method'] == 'default'
        assert result['confidence'] == 0.0

    def test_get_optimization_with_cache_miss(self, converter, sample_features, sample_optimization_result):
        """Test optimization with cache miss"""
        with patch.object(converter.optimizer, 'optimize') as mock_optimize, \
             patch.object(converter, '_infer_logo_type') as mock_infer:

            mock_optimize.return_value = {"parameters": sample_optimization_result["parameters"]}
            mock_infer.return_value = "simple"

            result = converter._get_optimization_with_cache(sample_features, "test.png")

            assert "parameters" in result
            assert result["logo_type"] == "simple"
            assert converter.cache_misses == 1

    def test_get_optimization_with_cache_hit(self, converter, sample_features, sample_optimization_result):
        """Test optimization with cache hit"""
        # Pre-populate cache with similar features
        feature_key = tuple(sorted(sample_features.items()))
        converter.optimization_cache[feature_key] = sample_optimization_result
        converter.config['similarity_threshold'] = 0.5  # Lower threshold for testing

        with patch.object(converter, '_find_similar_optimization') as mock_find:
            mock_find.return_value = sample_optimization_result

            result = converter._get_optimization_with_cache(sample_features, "test.png")

            assert result == sample_optimization_result
            assert converter.cache_hits == 1

    def test_get_optimization_with_cache_error(self, converter, sample_features):
        """Test optimization error handling"""
        with patch.object(converter.optimizer, 'optimize', side_effect=Exception("Optimization failed")), \
             patch.object(converter.error_handler, 'detect_error') as mock_detect, \
             patch.object(converter.error_handler, 'attempt_recovery') as mock_recovery, \
             patch.object(converter, '_infer_logo_type'):

            mock_detect.return_value = {"type": "optimization_error"}
            mock_recovery.return_value = {"success": False}

            result = converter._get_optimization_with_cache(sample_features, "test.png")

            assert result['method'] == 'final_fallback'
            assert result['confidence'] == 0.0

    @patch('backend.converters.ai_enhanced_converter.vtracer.convert_image_to_svg_py')
    @patch('builtins.open', mock_open(read_data='<svg>converted content</svg>'))
    @patch('backend.converters.ai_enhanced_converter.tempfile.NamedTemporaryFile')
    def test_convert_with_optimized_params_success(self, mock_tempfile, mock_vtracer, converter):
        """Test successful VTracer conversion"""
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        parameters = {"color_precision": 6, "corner_threshold": 60}

        result = converter._convert_with_optimized_params("test.png", parameters)

        assert result == '<svg>converted content</svg>'
        mock_vtracer.assert_called_once()

    @patch('backend.converters.ai_enhanced_converter.vtracer.convert_image_to_svg_py')
    @patch('builtins.open', mock_open(read_data=''))
    @patch('backend.converters.ai_enhanced_converter.tempfile.NamedTemporaryFile')
    def test_convert_with_optimized_params_empty_svg(self, mock_tempfile, mock_vtracer, converter):
        """Test handling of empty SVG content"""
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        parameters = {"color_precision": 6}

        with pytest.raises(ValueError, match="VTracer produced empty SVG content"):
            converter._convert_with_optimized_params("test.png", parameters)

    def test_convert_with_optimized_params_error_recovery(self, converter):
        """Test VTracer conversion error recovery"""
        with patch('backend.converters.ai_enhanced_converter.vtracer.convert_image_to_svg_py', side_effect=Exception("VTracer failed")), \
             patch.object(converter.error_handler, 'detect_error') as mock_detect, \
             patch.object(converter.error_handler, 'attempt_recovery') as mock_recovery:

            mock_detect.return_value = {"type": "vtracer_error"}
            mock_recovery.return_value = {"success": False}

            parameters = {"color_precision": 6}

            with pytest.raises(RuntimeError, match="VTracer conversion failed"):
                converter._convert_with_optimized_params("test.png", parameters)

    def test_handle_conversion_error_with_recovery(self, converter):
        """Test conversion error handling with successful recovery"""
        exception = Exception("Test error")

        with patch.object(converter.error_handler, 'detect_error') as mock_detect, \
             patch.object(converter.error_handler, 'attempt_recovery') as mock_recovery, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert:

            mock_detect.return_value = {"type": "conversion_error"}
            mock_recovery.return_value = {
                "success": True,
                "fallback_parameters": {"color_precision": 4}
            }
            mock_convert.return_value = "<svg>recovered</svg>"

            result = converter._handle_conversion_error(exception, "test.png")

            assert result == "<svg>recovered</svg>"
            mock_convert.assert_called()

    def test_handle_conversion_error_final_fallback(self, converter):
        """Test conversion error final fallback"""
        exception = Exception("Test error")

        with patch.object(converter.error_handler, 'detect_error') as mock_detect, \
             patch.object(converter.error_handler, 'attempt_recovery') as mock_recovery, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert:

            mock_detect.return_value = {"type": "conversion_error"}
            mock_recovery.return_value = {"success": False}
            mock_convert.return_value = "<svg>fallback</svg>"

            result = converter._handle_conversion_error(exception, "test.png")

            assert result == "<svg>fallback</svg>"

    def test_track_conversion_result(self, converter, sample_features, sample_optimization_result):
        """Test conversion result tracking"""
        initial_count = len(converter.conversion_metadata)

        converter._track_conversion_result(
            "test.png", sample_features, sample_optimization_result,
            1.5, "<svg>content</svg>"
        )

        assert len(converter.conversion_metadata) == initial_count + 1

        metadata = converter.conversion_metadata[-1]
        assert metadata['image_path'] == "test.png"
        assert metadata['features'] == sample_features
        assert metadata['conversion_time'] == 1.5
        assert metadata['svg_size'] == len("<svg>content</svg>")

    def test_track_conversion_result_with_quality(self, converter, sample_features, sample_optimization_result):
        """Test conversion tracking with quality calculation"""
        with patch.object(converter.quality_metrics, 'compare_images') as mock_quality, \
             patch('backend.converters.ai_enhanced_converter.tempfile.NamedTemporaryFile') as mock_temp:

            mock_quality.return_value = {"ssim": 0.85}
            mock_temp_file = MagicMock()
            mock_temp_file.name = '/tmp/test.svg'
            mock_temp.return_value.__enter__.return_value = mock_temp_file

            converter._track_conversion_result(
                "test.png", sample_features, sample_optimization_result,
                1.5, "<svg>content</svg>", calculate_quality=True
            )

            metadata = converter.conversion_metadata[-1]
            assert metadata['quality_metrics'] == {"ssim": 0.85}

    def test_find_similar_optimization(self, converter, sample_features, sample_optimization_result):
        """Test finding similar optimization in cache"""
        # Add optimization to cache
        feature_key = tuple(sorted(sample_features.items()))
        converter.optimization_cache[feature_key] = sample_optimization_result

        # Test with identical features
        result = converter._find_similar_optimization(sample_features)
        assert result == sample_optimization_result

    def test_find_similar_optimization_no_match(self, converter, sample_features):
        """Test finding similar optimization when no match exists"""
        # Different features
        different_features = {
            "edge_density": 0.9,
            "unique_colors": 0.8,
            "entropy": 0.1
        }

        result = converter._find_similar_optimization(different_features)
        assert result is None

    def test_calculate_feature_similarity(self, converter):
        """Test feature similarity calculation"""
        features1 = {"edge_density": 0.5, "unique_colors": 0.3}
        features2 = {"edge_density": 0.6, "unique_colors": 0.4}

        similarity = converter._calculate_feature_similarity(features1, features2)

        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.7  # Should be fairly similar

    def test_calculate_feature_similarity_no_common_keys(self, converter):
        """Test feature similarity with no common keys"""
        features1 = {"edge_density": 0.5}
        features2 = {"unique_colors": 0.3}

        similarity = converter._calculate_feature_similarity(features1, features2)
        assert similarity == 0.0

    def test_cache_optimization_result(self, converter, sample_features, sample_optimization_result):
        """Test optimization result caching"""
        converter._cache_optimization_result(sample_features, sample_optimization_result)

        feature_key = tuple(sorted(sample_features.items()))
        assert feature_key in converter.optimization_cache
        assert converter.optimization_cache[feature_key] == sample_optimization_result

    def test_cache_optimization_result_size_limit(self, converter, sample_optimization_result):
        """Test optimization cache size management"""
        # Fill cache beyond limit
        converter.config['cache_max_size'] = 2

        for i in range(5):
            features = {"test_feature": float(i)}
            converter._cache_optimization_result(features, sample_optimization_result)

        # Cache should be limited to max size
        assert len(converter.optimization_cache) <= converter.config['cache_max_size'] + 100  # FIFO removes 100

    def test_manage_feature_cache(self, converter, sample_features):
        """Test feature cache management"""
        cache_key = "test_key"
        converter._manage_feature_cache(cache_key, sample_features)

        assert cache_key in converter.feature_cache
        assert converter.feature_cache[cache_key] == sample_features

    def test_generate_cache_key(self, converter):
        """Test cache key generation"""
        key = converter._generate_cache_key("test", "path", 123)

        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

        # Same inputs should generate same key
        key2 = converter._generate_cache_key("test", "path", 123)
        assert key == key2

    def test_infer_logo_type_simple(self, converter):
        """Test logo type inference for simple logos"""
        simple_features = {
            "complexity_score": 0.2,
            "edge_density": 0.1,
            "entropy": 0.4,
            "unique_colors": 0.3
        }

        logo_type = converter._infer_logo_type(simple_features)
        assert logo_type == "simple"

    def test_infer_logo_type_text(self, converter):
        """Test logo type inference for text logos"""
        text_features = {
            "complexity_score": 0.5,
            "edge_density": 0.4,
            "entropy": 0.9,
            "unique_colors": 0.1
        }

        logo_type = converter._infer_logo_type(text_features)
        assert logo_type == "text"

    def test_infer_logo_type_gradient(self, converter):
        """Test logo type inference for gradient logos"""
        gradient_features = {
            "complexity_score": 0.5,
            "edge_density": 0.4,
            "entropy": 0.6,
            "unique_colors": 0.8,
            "gradient_strength": 0.7
        }

        logo_type = converter._infer_logo_type(gradient_features)
        assert logo_type == "gradient"

    def test_infer_logo_type_complex(self, converter):
        """Test logo type inference for complex logos"""
        complex_features = {
            "complexity_score": 0.8,
            "edge_density": 0.7,
            "entropy": 0.6,
            "unique_colors": 0.5
        }

        logo_type = converter._infer_logo_type(complex_features)
        assert logo_type == "complex"

    def test_validate_vtracer_parameters(self, converter):
        """Test VTracer parameter validation"""
        input_params = {
            "color_precision": 15,  # Too high, should be clamped
            "corner_threshold": 5,  # Too low, should be clamped
            "invalid_param": "test",  # Invalid, should be ignored
            "colormode": "binary"  # Valid
        }

        validated = converter._validate_vtracer_parameters(input_params)

        assert validated["color_precision"] == 10  # Clamped to max
        assert validated["corner_threshold"] == 10  # Clamped to min
        assert validated["colormode"] == "binary"  # Preserved
        assert "invalid_param" not in validated

    def test_validate_vtracer_parameters_invalid_types(self, converter):
        """Test parameter validation with invalid types"""
        input_params = {
            "color_precision": "invalid",  # Wrong type
            "corner_threshold": None  # Wrong type
        }

        validated = converter._validate_vtracer_parameters(input_params)

        # Should use defaults for invalid types
        assert isinstance(validated["color_precision"], int)
        assert isinstance(validated["corner_threshold"], int)

    def test_get_default_vtracer_params(self, converter):
        """Test default VTracer parameters"""
        defaults = converter._get_default_vtracer_params()

        assert defaults["colormode"] == "color"
        assert isinstance(defaults["color_precision"], int)
        assert isinstance(defaults["corner_threshold"], int)
        assert 1 <= defaults["color_precision"] <= 10

    def test_get_default_features(self, converter):
        """Test default feature values"""
        defaults = converter._get_default_features()

        assert "edge_density" in defaults
        assert "unique_colors" in defaults
        assert "entropy" in defaults
        assert "complexity_score" in defaults

        # All values should be normalized
        for value in defaults.values():
            assert 0.0 <= value <= 1.0

    def test_get_optimization_stats(self, converter):
        """Test optimization statistics retrieval"""
        # Add some fake cache data
        converter.cache_hits = 10
        converter.cache_misses = 5
        converter.optimization_cache["test"] = {}
        converter.feature_cache["test"] = {}

        stats = converter.get_optimization_stats()

        assert stats["total_conversions"] == 15
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["cache_hit_rate"] == 10/15
        assert stats["optimization_cache_size"] == 1
        assert stats["feature_cache_size"] == 1

    def test_configure(self, converter):
        """Test configuration updates"""
        initial_value = converter.config["cache_max_size"]

        converter.configure(cache_max_size=2000)

        assert converter.config["cache_max_size"] == 2000
        assert converter.config["cache_max_size"] != initial_value

    def test_configure_invalid_key(self, converter):
        """Test configuration with invalid key"""
        # Should not raise exception, just log warning
        converter.configure(invalid_key="test")

        # Original config should be unchanged
        assert "invalid_key" not in converter.config

    def test_clear_cache(self, converter):
        """Test cache clearing"""
        # Add some cache data
        converter.optimization_cache["test"] = {}
        converter.feature_cache["test"] = {}
        converter.cache_hits = 10
        converter.cache_misses = 5

        converter.clear_cache()

        assert len(converter.optimization_cache) == 0
        assert len(converter.feature_cache) == 0
        assert converter.cache_hits == 0
        assert converter.cache_misses == 0

    def test_export_conversion_history(self, converter, sample_features, sample_optimization_result):
        """Test conversion history export"""
        # Add some history
        converter._track_conversion_result(
            "test.png", sample_features, sample_optimization_result,
            1.0, "<svg>test</svg>"
        )

        history = converter.export_conversion_history()

        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["image_path"] == "test.png"

    def test_batch_convert_success(self, converter, sample_features, sample_optimization_result):
        """Test successful batch conversion"""
        image_paths = ["test1.png", "test2.png"]

        with patch.object(converter, '_get_features_with_cache') as mock_features, \
             patch.object(converter, '_get_optimization_with_cache') as mock_opt, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert:

            mock_features.return_value = sample_features
            mock_opt.return_value = sample_optimization_result
            mock_convert.return_value = "<svg>batch content</svg>"

            results = converter.batch_convert(image_paths)

            assert len(results) == 2
            for result in results:
                assert result["success"] is True
                assert result["svg_content"] == "<svg>batch content</svg>"

    def test_batch_convert_with_errors(self, converter, sample_features):
        """Test batch conversion with some failures"""
        image_paths = ["test1.png", "test2.png"]

        with patch.object(converter, '_get_features_with_cache') as mock_features, \
             patch.object(converter, '_get_optimization_with_cache') as mock_opt, \
             patch.object(converter, '_convert_with_optimized_params') as mock_convert:

            mock_features.side_effect = [sample_features, Exception("Feature extraction failed")]
            mock_opt.return_value = {"parameters": {}, "confidence": 0.0}
            mock_convert.side_effect = ["<svg>success</svg>", Exception("Conversion failed")]

            results = converter.batch_convert(image_paths)

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is False

    def test_convert_with_quality_validation_success(self, converter, temp_image_file):
        """Test conversion with quality validation"""
        with patch.object(converter, 'convert') as mock_convert, \
             patch.object(converter.quality_metrics, 'compare_images') as mock_quality, \
             patch('backend.converters.ai_enhanced_converter.tempfile.NamedTemporaryFile') as mock_temp:

            mock_convert.return_value = "<svg>quality test</svg>"
            mock_quality.return_value = {"ssim": 0.9}
            mock_temp_file = MagicMock()
            mock_temp_file.name = '/tmp/test.svg'
            mock_temp.return_value.__enter__.return_value = mock_temp_file

            # Add some conversion metadata
            converter.conversion_metadata.append({"test": "metadata"})

            result = converter.convert_with_quality_validation(temp_image_file)

            assert result["svg_content"] == "<svg>quality test</svg>"
            assert result["quality_metrics"]["ssim"] == 0.9
            assert result["meets_quality_target"] is True  # 0.9 >= 0.85

    def test_convert_with_quality_validation_failure(self, converter, temp_image_file):
        """Test conversion with quality validation failure"""
        with patch.object(converter, 'convert') as mock_convert, \
             patch.object(converter.quality_metrics, 'compare_images', side_effect=Exception("Quality calc failed")):

            mock_convert.return_value = "<svg>quality test</svg>"

            result = converter.convert_with_quality_validation(temp_image_file)

            assert result["svg_content"] == "<svg>quality test</svg>"
            assert "error" in result["quality_metrics"]

    def test_metadata_history_limit(self, converter, sample_features, sample_optimization_result):
        """Test metadata history size limiting"""
        # Add more than limit
        for i in range(1200):  # More than 1000 limit
            converter._track_conversion_result(
                f"test{i}.png", sample_features, sample_optimization_result,
                1.0, f"<svg>test{i}</svg>"
            )

        # Should be limited to 500 (after cleanup)
        assert len(converter.conversion_metadata) == 500

    def test_error_handling_in_various_methods(self, converter):
        """Test error handling in various utility methods"""
        # Test feature similarity with error
        with patch('builtins.sum', side_effect=Exception("Math error")):
            similarity = converter._calculate_feature_similarity({}, {})
            assert similarity == 0.0

        # Test optimization enhancement with error
        with patch('backend.converters.ai_enhanced_converter.logger') as mock_logger:
            with patch('builtins.min', side_effect=Exception("Enhancement error")):
                result = converter._enhance_with_quality_prediction({}, {}, "test.png")
                mock_logger.warning.assert_called()

    def test_concurrent_cache_access(self, converter, sample_features):
        """Test thread safety of cache operations"""
        import threading

        def cache_operation():
            for i in range(10):
                key = f"test_{threading.current_thread().ident}_{i}"
                converter._manage_feature_cache(key, sample_features)

        threads = [threading.Thread(target=cache_operation) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have cached items from all threads
        assert len(converter.feature_cache) > 0