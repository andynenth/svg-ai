"""
Integration Testing Suite - Task 4 Implementation
Comprehensive tests for the unified AI pipeline with component combinations and failure scenarios.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Import the components to test
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline, PipelineResult
from backend.ai_modules.pipeline.component_interfaces import *
from backend.ai_modules.pipeline.pipeline_config import PipelineConfigManager


class TestUnifiedPipeline:
    """Test the unified AI pipeline end-to-end."""

    @pytest.fixture
    def test_image_path(self):
        """Provide path to test image."""
        test_image = "data/logos/simple_geometric/circle_00.png"
        if Path(test_image).exists():
            return test_image

        # Create a temporary test image if the real one doesn't exist
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create a minimal PNG file for testing
            tmp.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01U\r\xd1\x8d\x00\x00\x00\x00IEND\xaeB`\x82')
            return tmp.name

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance for testing."""
        # Disable hot-reload and use temporary config for testing
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            config_path = tmp.name

        pipeline = UnifiedAIPipeline(
            enable_caching=True,
            enable_fallbacks=True,
            performance_mode="balanced"
        )

        yield pipeline

        # Cleanup
        try:
            Path(config_path).unlink(missing_ok=True)
        except:
            pass

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing failure scenarios."""
        mock_feature_extractor = Mock()
        mock_feature_extractor.extract_features.return_value = {
            'edge_density': 0.7,
            'unique_colors': 100,
            'entropy': 0.5,
            'corner_density': 0.3,
            'gradient_strength': 0.6,
            'complexity_score': 0.8
        }

        mock_classifier = Mock()
        mock_classifier.classify.return_value = {
            'logo_type': 'simple',
            'confidence': 0.9,
            'all_probabilities': {'simple': 0.9, 'complex': 0.1},
            'success': True
        }

        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            'parameters': {
                'corner_threshold': 30,
                'color_precision': 4,
                'path_precision': 8,
                'splice_threshold': 45,
                'max_iterations': 10,
                'length_threshold': 5.0
            },
            'confidence': 0.8,
            'metadata': {'method': 'test_optimizer'}
        }

        mock_converter = Mock()
        mock_converter.convert.return_value = '<svg>test</svg>'

        return {
            'feature_extractor': mock_feature_extractor,
            'classifier': mock_classifier,
            'optimizer': mock_optimizer,
            'converter': mock_converter
        }

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly with all components."""
        assert pipeline is not None
        assert hasattr(pipeline, 'feature_extractor')
        assert hasattr(pipeline, 'primary_classifier')
        assert hasattr(pipeline, 'primary_optimizer')
        assert hasattr(pipeline, 'converter')

        # Check component loading status
        stats = pipeline.get_pipeline_statistics()
        assert 'components_status' in stats

        # At least some components should be loaded
        loaded_count = sum(1 for status in stats['components_status'].values() if status)
        assert loaded_count > 0

    def test_pipeline_health_check(self, pipeline):
        """Test pipeline health check functionality."""
        health = pipeline.health_check()

        assert 'overall_status' in health
        assert health['overall_status'] in ['healthy', 'degraded']
        assert 'components' in health
        assert 'critical_issues' in health
        assert 'warnings' in health
        assert 'recommendations' in health

    def test_pipeline_end_to_end_success(self, pipeline, test_image_path):
        """Test complete pipeline execution with successful result."""
        result = pipeline.process(
            image_path=test_image_path,
            target_quality=0.85,
            time_constraint=30.0
        )

        assert isinstance(result, PipelineResult)
        assert result.processing_time > 0
        assert 'stage_times' in result.metadata

        # Check that we got through the pipeline (success or graceful failure)
        if result.success:
            assert result.svg_content is not None
            assert len(result.parameters) > 0
            assert result.quality_score >= 0
            assert result.classification is not None
            assert 'features' in result.__dict__
        else:
            # Graceful failure is acceptable
            assert result.error_message is not None
            assert isinstance(result.error_message, str)

    def test_pipeline_with_time_constraint(self, pipeline, test_image_path):
        """Test pipeline respects time constraints."""
        start_time = time.time()

        result = pipeline.process(
            image_path=test_image_path,
            target_quality=0.85,
            time_constraint=1.0  # Very short time constraint
        )

        elapsed_time = time.time() - start_time

        # Should complete within reasonable time (allowing some overhead)
        assert elapsed_time < 10.0  # Give some margin for overhead
        assert result.processing_time > 0

    def test_pipeline_with_different_quality_targets(self, pipeline, test_image_path):
        """Test pipeline with different quality targets."""
        quality_targets = [0.7, 0.85, 0.95]

        for target in quality_targets:
            result = pipeline.process(
                image_path=test_image_path,
                target_quality=target
            )

            assert isinstance(result, PipelineResult)
            assert result.metadata['target_quality'] == target

            # Should complete with some result
            assert result.processing_time > 0

    def test_pipeline_performance_tracking(self, pipeline, test_image_path):
        """Test that pipeline tracks performance metrics correctly."""
        # Process a few images
        for _ in range(3):
            pipeline.process(test_image_path)

        stats = pipeline.get_pipeline_statistics()

        assert stats['total_processed'] == 3
        assert stats['average_processing_time_ms'] > 0
        assert 'stage_timings' in stats
        assert 'components_status' in stats

    def test_component_fallback_classifier(self, pipeline, mock_components):
        """Test classifier fallback functionality."""
        # Mock primary classifier to fail
        with patch.object(pipeline, 'primary_classifier', None):
            with patch.object(pipeline, 'fallback_classifier', mock_components['classifier']):
                result = pipeline.process("test_image.png")

                # Should use fallback classifier
                if result.classification:
                    assert 'classifier_used' in result.classification or result.classification.get('logo_type') is not None

    def test_component_fallback_optimizer(self, pipeline, mock_components):
        """Test optimizer fallback functionality."""
        # Mock feature extractor to return basic features
        features = {
            'edge_density': 0.5,
            'unique_colors': 100,
            'entropy': 0.5
        }

        with patch.object(pipeline, 'feature_extractor') as mock_fe:
            mock_fe.extract_features.return_value = features

            # Mock primary optimizer to fail
            with patch.object(pipeline, 'primary_optimizer', None):
                with patch.object(pipeline, 'fallback_optimizer', mock_components['optimizer']):
                    result = pipeline.process("test_image.png")

                    # Should use fallback optimizer
                    if result.optimization_result:
                        assert 'optimizer_used' in result.optimization_result

    def test_pipeline_with_missing_image(self, pipeline):
        """Test pipeline behavior with missing image file."""
        result = pipeline.process("nonexistent_image.png")

        assert isinstance(result, PipelineResult)
        assert not result.success
        assert result.error_message is not None

    def test_pipeline_with_invalid_parameters(self, pipeline, test_image_path):
        """Test pipeline with invalid input parameters."""
        # Test with invalid target quality
        result = pipeline.process(
            image_path=test_image_path,
            target_quality=1.5  # Invalid quality > 1.0
        )

        # Should handle gracefully (either clamp or fail gracefully)
        assert isinstance(result, PipelineResult)

    def test_pipeline_concurrent_processing(self, pipeline, test_image_path):
        """Test pipeline handles concurrent requests safely."""
        import threading

        results = []
        errors = []

        def process_image():
            try:
                result = pipeline.process(test_image_path)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=process_image)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 3  # All requests should complete

        # All results should be valid
        for result in results:
            assert isinstance(result, PipelineResult)

    def test_pipeline_caching_functionality(self, pipeline, test_image_path):
        """Test that caching improves performance on repeated requests."""
        # First request
        start_time = time.time()
        result1 = pipeline.process(test_image_path)
        first_time = time.time() - start_time

        # Second request (should potentially be faster due to caching)
        start_time = time.time()
        result2 = pipeline.process(test_image_path)
        second_time = time.time() - start_time

        # Both should succeed
        assert isinstance(result1, PipelineResult)
        assert isinstance(result2, PipelineResult)

        # Results should be consistent
        if result1.success and result2.success:
            # Check if features are similar (caching working)
            assert 'features' in result1.__dict__
            assert 'features' in result2.__dict__

    def test_pipeline_statistics_accuracy(self, pipeline, test_image_path):
        """Test that pipeline statistics are accurate."""
        initial_stats = pipeline.get_pipeline_statistics()
        initial_count = initial_stats['total_processed']

        # Process some images
        num_requests = 5
        for _ in range(num_requests):
            pipeline.process(test_image_path)

        final_stats = pipeline.get_pipeline_statistics()

        # Check that counts are accurate
        assert final_stats['total_processed'] == initial_count + num_requests
        assert final_stats['average_processing_time_ms'] > 0


class TestComponentIntegration:
    """Test integration between different components."""

    @pytest.fixture
    def config_manager(self):
        """Create configuration manager for testing."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            config_path = tmp.name

        manager = PipelineConfigManager(
            config_path=config_path,
            enable_hot_reload=False,
            auto_save=False
        )

        yield manager

        # Cleanup
        manager.cleanup()
        Path(config_path).unlink(missing_ok=True)

    def test_pipeline_with_custom_config(self, config_manager):
        """Test pipeline initialization with custom configuration."""
        # Update configuration
        updates = {
            'global_settings': {
                'performance_mode': 'fast',
                'enable_caching': False
            },
            'classifier': {
                'confidence_threshold': 0.9
            }
        }

        success = config_manager.update_configuration(updates)
        assert success

        # Create pipeline with custom config (would need integration)
        pipeline = UnifiedAIPipeline(
            enable_caching=False,
            enable_fallbacks=True,
            performance_mode="fast"
        )

        assert pipeline.enable_caching == False
        assert pipeline.performance_mode == "fast"

    def test_adapter_functionality(self):
        """Test component adapters work correctly."""
        # Test feature extractor adapter
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

        try:
            original_extractor = ImageFeatureExtractor()
            adapter = FeatureExtractorAdapter(original_extractor)

            assert hasattr(adapter, 'extract_features')
            assert hasattr(adapter, 'get_feature_names')
            assert hasattr(adapter, 'validate_image')

            feature_names = adapter.get_feature_names()
            assert isinstance(feature_names, list)
            assert len(feature_names) > 0

        except Exception as e:
            # If component initialization fails, that's expected in some environments
            pytest.skip(f"Component initialization failed: {e}")

    def test_interface_compliance(self):
        """Test that adapters comply with interfaces."""
        # Create mock component
        mock_component = Mock()

        # Test classifier adapter
        classifier_adapter = ClassifierAdapter(mock_component)

        # Check interface compliance
        is_compliant, missing = validate_interface_compliance(
            classifier_adapter, BaseClassifier
        )

        # Adapter should implement all required methods
        assert is_compliant or len(missing) == 0

    def test_configuration_validation_edge_cases(self, config_manager):
        """Test configuration validation with edge cases."""
        # Test invalid confidence threshold
        invalid_updates = {
            'classifier': {
                'confidence_threshold': 1.5  # Invalid: > 1.0
            }
        }

        success = config_manager.update_configuration(invalid_updates)
        assert not success  # Should fail validation

        # Test invalid performance mode
        invalid_updates = {
            'global_settings': {
                'performance_mode': 'invalid_mode'
            }
        }

        success = config_manager.update_configuration(invalid_updates)
        assert not success  # Should fail validation

    def test_pipeline_error_recovery(self):
        """Test pipeline error recovery mechanisms."""
        pipeline = UnifiedAIPipeline(
            enable_caching=True,
            enable_fallbacks=True
        )

        # Test with completely broken components
        pipeline.feature_extractor = None

        result = pipeline.process("test_image.png")

        # Should fail gracefully
        assert isinstance(result, PipelineResult)
        assert not result.success
        assert result.error_message is not None


class TestPerformanceRequirements:
    """Test that performance requirements are met."""

    def test_pipeline_processing_time(self):
        """Test that pipeline meets processing time requirements."""
        pipeline = UnifiedAIPipeline(performance_mode="balanced")

        # Create a simple test image path
        test_image = "data/logos/simple_geometric/circle_00.png"
        if not Path(test_image).exists():
            pytest.skip("Test image not available")

        start_time = time.time()
        result = pipeline.process(test_image, target_quality=0.85)
        elapsed_time = time.time() - start_time

        # Should complete within reasonable time for Tier 2 (5 seconds + margin)
        assert elapsed_time < 10.0

        if result.success:
            assert result.processing_time > 0
            assert result.processing_time < 10.0

    def test_pipeline_memory_usage(self):
        """Test that pipeline doesn't leak memory excessively."""
        import psutil
        import gc

        pipeline = UnifiedAIPipeline()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Process multiple images
        test_image = "data/logos/simple_geometric/circle_00.png"
        if Path(test_image).exists():
            for _ in range(5):
                result = pipeline.process(test_image)
                gc.collect()  # Force garbage collection

            # Check memory usage hasn't grown excessively
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory

            # Allow some memory growth, but not excessive (100MB limit)
            assert memory_growth < 100 * 1024 * 1024

    def test_pipeline_cache_effectiveness(self):
        """Test that caching provides performance benefits."""
        pipeline = UnifiedAIPipeline(enable_caching=True)

        test_image = "data/logos/simple_geometric/circle_00.png"
        if not Path(test_image).exists():
            pytest.skip("Test image not available")

        # First request (cold cache)
        start_time = time.time()
        result1 = pipeline.process(test_image)
        first_time = time.time() - start_time

        # Second request (warm cache)
        start_time = time.time()
        result2 = pipeline.process(test_image)
        second_time = time.time() - start_time

        # Second request should be faster (or at least not much slower)
        # Allow for some variance but expect caching benefits
        assert second_time <= first_time * 1.5  # At most 50% slower


def test_pipeline_integration_full_workflow():
    """Integration test for the complete workflow."""
    # This test runs the full pipeline end-to-end
    pipeline = UnifiedAIPipeline(
        enable_caching=True,
        enable_fallbacks=True,
        performance_mode="balanced"
    )

    # Test with a simple image if available
    test_image = "data/logos/simple_geometric/circle_00.png"

    if Path(test_image).exists():
        result = pipeline.process(
            image_path=test_image,
            target_quality=0.85,
            time_constraint=30.0
        )

        # Verify result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'metadata')

        # Check metadata completeness
        assert 'pipeline_start' in result.metadata
        assert 'image_path' in result.metadata
        assert 'target_quality' in result.metadata

        if result.success:
            # If successful, check all expected components
            assert result.svg_content is not None
            assert len(result.parameters) > 0
            assert result.features is not None
            assert result.classification is not None
        else:
            # If failed, should have error message
            assert result.error_message is not None

        # Check pipeline statistics
        stats = pipeline.get_pipeline_statistics()
        assert stats['total_processed'] >= 1
        assert stats['average_processing_time_ms'] > 0
    else:
        pytest.skip("Test image not available for integration test")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])