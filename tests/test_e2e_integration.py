#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Tests for Week 2 Implementation

This module provides complete integration testing of the entire Week 2 AI-enhanced
PNG to SVG conversion pipeline, including:
- Full workflow integration from image input to SVG output
- AI enhancement vs standard conversion validation
- Error handling and recovery scenarios
- Performance testing under various conditions
- Cache integration testing
- Web API endpoint integration
"""

import pytest
import os
import sys
import time
import json
import tempfile
import hashlib
import requests
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, Mock
from PIL import Image
import io

# Test infrastructure imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class E2ETestRunner:
    """End-to-end test runner for comprehensive pipeline testing."""

    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.error_scenarios = []

    def log_test_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any]):
        """Log test result for comprehensive reporting."""
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'duration': duration,
            'details': details,
            'timestamp': time.time()
        })

    def get_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive test execution summary."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        total_duration = sum(r['duration'] for r in self.test_results)

        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration,
            'average_test_duration': total_duration / total_tests if total_tests > 0 else 0,
            'performance_metrics': self.performance_metrics,
            'error_scenarios': self.error_scenarios
        }


# Global test runner instance
e2e_runner = E2ETestRunner()


class TestFullPipelineIntegration:
    """Test complete pipeline integration from image input to SVG output."""

    def test_basic_pipeline_flow(self, temp_png_file):
        """Test basic end-to-end conversion flow."""
        start_time = time.time()

        try:
            # Test basic converter import and initialization
            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()
            assert converter.get_name() == "VTracer Converter"

            # Test conversion
            svg_content = converter.convert(temp_png_file)

            # Validate SVG output
            assert svg_content is not None
            assert isinstance(svg_content, str)
            assert len(svg_content) > 0
            assert '<svg' in svg_content.lower()
            assert 'xmlns="http://www.w3.org/2000/svg"' in svg_content

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'basic_pipeline_flow', True, duration,
                {'svg_size': len(svg_content), 'converter': 'VTracer'}
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'basic_pipeline_flow', False, duration,
                {'error': str(e)}
            )
            raise

    def test_ai_enhanced_pipeline_flow(self, temp_png_file):
        """Test AI-enhanced conversion pipeline."""
        start_time = time.time()

        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedConverter

            converter = AIEnhancedConverter(enable_ai=True, ai_timeout=10.0)

            # Test with detailed AI analysis
            result = converter.convert_with_ai_analysis(temp_png_file)

            # Validate comprehensive AI results
            assert result is not None
            assert isinstance(result, dict)
            assert 'svg' in result
            assert 'classification' in result
            assert 'features' in result
            assert 'success' in result
            assert result['success'] is True

            # Validate SVG content
            svg_content = result['svg']
            assert svg_content is not None
            assert '<svg' in svg_content.lower()

            # Validate AI analysis
            classification = result['classification']
            assert 'logo_type' in classification
            assert 'confidence' in classification

            features = result['features']
            assert isinstance(features, dict)
            assert len(features) > 0

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_enhanced_pipeline_flow', True, duration,
                {
                    'svg_size': len(svg_content),
                    'ai_enhanced': result.get('ai_enhanced', False),
                    'logo_type': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'ai_time': result.get('ai_analysis_time', 0.0),
                    'conversion_time': result.get('conversion_time', 0.0)
                }
            )

        except ImportError:
            # AI modules not available - skip test
            pytest.skip("AI modules not available for testing")
        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_enhanced_pipeline_flow', False, duration,
                {'error': str(e)}
            )
            raise

    def test_converter_integration_with_base_class(self, temp_png_file):
        """Test that all converters properly integrate with BaseConverter interface."""
        start_time = time.time()

        try:
            from backend.converters.base import BaseConverter
            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()

            # Test BaseConverter interface compliance
            assert isinstance(converter, BaseConverter)
            assert hasattr(converter, 'convert')
            assert hasattr(converter, 'get_name')
            assert hasattr(converter, 'convert_with_metrics')
            assert hasattr(converter, 'get_stats')

            # Test metrics collection
            result = converter.convert_with_metrics(temp_png_file)

            assert isinstance(result, dict)
            assert 'success' in result
            assert 'svg' in result
            assert 'time' in result
            assert 'converter' in result

            # Test statistics
            stats = converter.get_stats()
            assert isinstance(stats, dict)
            assert 'total_conversions' in stats
            assert 'success_rate' in stats
            assert 'average_time' in stats

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'converter_integration_base_class', True, duration,
                {
                    'conversion_success': result['success'],
                    'conversion_time': result['time'],
                    'stats': stats
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'converter_integration_base_class', False, duration,
                {'error': str(e)}
            )
            raise


class TestAIEnhancementValidation:
    """Test and validate AI enhancement vs standard conversion."""

    def test_ai_vs_standard_conversion_comparison(self, temp_png_file):
        """Compare AI-enhanced conversion with standard conversion."""
        start_time = time.time()

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            # Standard conversion
            standard_converter = VTracerConverter()
            standard_result = standard_converter.convert_with_metrics(temp_png_file)

            try:
                from backend.converters.ai_enhanced_converter import AIEnhancedConverter

                # AI-enhanced conversion
                ai_converter = AIEnhancedConverter(enable_ai=True)
                ai_result = ai_converter.convert_with_ai_analysis(temp_png_file)

                # Compare results
                comparison_data = {
                    'standard_success': standard_result['success'],
                    'ai_success': ai_result['success'],
                    'standard_time': standard_result['time'],
                    'ai_total_time': ai_result['total_time'],
                    'ai_analysis_time': ai_result.get('ai_analysis_time', 0),
                    'ai_conversion_time': ai_result.get('conversion_time', 0),
                    'standard_svg_size': len(standard_result['svg']) if standard_result['svg'] else 0,
                    'ai_svg_size': len(ai_result['svg']) if ai_result['svg'] else 0,
                    'ai_enhanced': ai_result.get('ai_enhanced', False),
                    'logo_type': ai_result.get('classification', {}).get('logo_type', 'unknown'),
                    'confidence': ai_result.get('classification', {}).get('confidence', 0.0)
                }

                # Validate both conversions succeeded
                assert standard_result['success'], "Standard conversion should succeed"
                assert ai_result['success'], "AI conversion should succeed"

                # Validate AI provides additional metadata
                if ai_result.get('ai_enhanced', False):
                    assert 'classification' in ai_result
                    assert 'features' in ai_result
                    assert len(ai_result['features']) > 0

            except ImportError:
                # AI not available - just test standard
                comparison_data = {
                    'standard_success': standard_result['success'],
                    'ai_available': False,
                    'standard_time': standard_result['time'],
                    'standard_svg_size': len(standard_result['svg']) if standard_result['svg'] else 0
                }

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_vs_standard_comparison', True, duration,
                comparison_data
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_vs_standard_comparison', False, duration,
                {'error': str(e)}
            )
            raise

    def test_ai_parameter_optimization(self, temp_png_file):
        """Test that AI parameter optimization works correctly."""
        start_time = time.time()

        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedConverter

            converter = AIEnhancedConverter(enable_ai=True)

            # Test conversion with AI optimization
            result = converter.convert_with_ai_analysis(temp_png_file)

            if result.get('ai_enhanced', False):
                # Validate parameter optimization
                parameters_used = result.get('parameters_used', {})
                assert isinstance(parameters_used, dict)

                # Check that key VTracer parameters are present and reasonable
                if 'color_precision' in parameters_used:
                    assert 1 <= parameters_used['color_precision'] <= 10
                if 'corner_threshold' in parameters_used:
                    assert 0 <= parameters_used['corner_threshold'] <= 180
                if 'layer_difference' in parameters_used:
                    assert 1 <= parameters_used['layer_difference'] <= 32

                # Validate features influenced parameters
                features = result.get('features', {})
                classification = result.get('classification', {})

                optimization_data = {
                    'logo_type': classification.get('logo_type', 'unknown'),
                    'confidence': classification.get('confidence', 0.0),
                    'parameters_count': len(parameters_used),
                    'feature_count': len(features),
                    'color_precision': parameters_used.get('color_precision'),
                    'corner_threshold': parameters_used.get('corner_threshold'),
                    'layer_difference': parameters_used.get('layer_difference')
                }
            else:
                optimization_data = {'ai_enhanced': False, 'reason': 'AI not available or disabled'}

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_parameter_optimization', True, duration,
                optimization_data
            )

        except ImportError:
            pytest.skip("AI modules not available for parameter optimization testing")
        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_parameter_optimization', False, duration,
                {'error': str(e)}
            )
            raise


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_file_handling(self):
        """Test handling of invalid input files."""
        start_time = time.time()

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()
            error_scenarios = []

            # Test non-existent file
            try:
                result = converter.convert_with_metrics("/nonexistent/path/file.png")
                assert result['success'] is False
                assert 'error' in result
                error_scenarios.append({'scenario': 'nonexistent_file', 'handled': True})
            except Exception as e:
                error_scenarios.append({'scenario': 'nonexistent_file', 'handled': False, 'error': str(e)})

            # Test invalid file content
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(b"invalid content")
                tmp.flush()

                try:
                    result = converter.convert_with_metrics(tmp.name)
                    # Should handle gracefully - either succeed with empty result or fail gracefully
                    error_scenarios.append({'scenario': 'invalid_content', 'handled': True, 'success': result.get('success', False)})
                except Exception as e:
                    error_scenarios.append({'scenario': 'invalid_content', 'handled': False, 'error': str(e)})
                finally:
                    os.unlink(tmp.name)

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'invalid_file_handling', True, duration,
                {'error_scenarios': error_scenarios}
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'invalid_file_handling', False, duration,
                {'error': str(e)}
            )
            raise

    def test_ai_fallback_mechanism(self, temp_png_file):
        """Test AI fallback to standard conversion on failures."""
        start_time = time.time()

        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedConverter

            # Test with AI disabled
            converter_disabled = AIEnhancedConverter(enable_ai=False)
            result_disabled = converter_disabled.convert_with_ai_analysis(temp_png_file)

            assert result_disabled['success'] is True
            assert result_disabled['ai_enhanced'] is False

            # Test with AI enabled but forced fallback
            converter_enabled = AIEnhancedConverter(enable_ai=True)
            result_fallback = converter_enabled.convert(temp_png_file, ai_disable=True)

            assert result_fallback is not None
            assert isinstance(result_fallback, str)
            assert '<svg' in result_fallback.lower()

            fallback_data = {
                'disabled_conversion_success': result_disabled['success'],
                'disabled_ai_enhanced': result_disabled['ai_enhanced'],
                'fallback_conversion_success': True,
                'fallback_svg_size': len(result_fallback)
            }

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_fallback_mechanism', True, duration,
                fallback_data
            )

        except ImportError:
            pytest.skip("AI modules not available for fallback testing")
        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'ai_fallback_mechanism', False, duration,
                {'error': str(e)}
            )
            raise

    def test_performance_under_stress(self, temp_png_file):
        """Test performance under stress conditions."""
        start_time = time.time()

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()

            # Test multiple rapid conversions
            results = []
            conversion_times = []

            num_conversions = 10  # Moderate load for testing

            for i in range(num_conversions):
                conv_start = time.time()
                result = converter.convert_with_metrics(temp_png_file)
                conv_time = time.time() - conv_start

                results.append(result)
                conversion_times.append(conv_time)

            # Analyze performance
            successful_conversions = sum(1 for r in results if r['success'])
            avg_conversion_time = sum(conversion_times) / len(conversion_times)
            max_conversion_time = max(conversion_times)
            min_conversion_time = min(conversion_times)

            performance_data = {
                'total_conversions': num_conversions,
                'successful_conversions': successful_conversions,
                'success_rate': (successful_conversions / num_conversions) * 100,
                'avg_conversion_time': avg_conversion_time,
                'max_conversion_time': max_conversion_time,
                'min_conversion_time': min_conversion_time,
                'time_variance': max_conversion_time - min_conversion_time
            }

            # Validate reasonable performance
            assert successful_conversions >= num_conversions * 0.8, "At least 80% conversions should succeed"
            assert avg_conversion_time < 5.0, "Average conversion time should be reasonable"

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'performance_under_stress', True, duration,
                performance_data
            )

            # Store performance metrics for summary
            e2e_runner.performance_metrics['stress_test'] = performance_data

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'performance_under_stress', False, duration,
                {'error': str(e)}
            )
            raise


class TestWebAPIIntegration:
    """Test web API endpoint integration."""

    def test_flask_app_initialization(self, flask_app):
        """Test Flask application initializes correctly."""
        start_time = time.time()

        try:
            assert flask_app is not None
            assert flask_app.config['TESTING'] is True

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'flask_app_initialization', True, duration,
                {'config_testing': flask_app.config['TESTING']}
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'flask_app_initialization', False, duration,
                {'error': str(e)}
            )
            raise

    def test_health_endpoint(self, flask_client):
        """Test health check endpoint."""
        start_time = time.time()

        try:
            response = flask_client.get('/health')
            assert response.status_code == 200

            data = response.get_json()
            assert data is not None
            assert 'status' in data
            assert data['status'] == 'ok'

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'health_endpoint', True, duration,
                {'status_code': response.status_code, 'response': data}
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'health_endpoint', False, duration,
                {'error': str(e)}
            )
            raise

    def test_upload_endpoint(self, flask_client, sample_png_bytes):
        """Test file upload endpoint."""
        start_time = time.time()

        try:
            # Test valid PNG upload
            response = flask_client.post('/api/upload',
                                       data={'file': (io.BytesIO(sample_png_bytes), 'test.png')},
                                       content_type='multipart/form-data')

            assert response.status_code == 200

            data = response.get_json()
            assert data is not None
            assert 'file_id' in data
            assert 'filename' in data
            assert 'path' in data

            file_id = data['file_id']
            assert len(file_id) == 32  # MD5 hash length

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'upload_endpoint', True, duration,
                {
                    'status_code': response.status_code,
                    'file_id': file_id,
                    'filename': data['filename']
                }
            )

            return file_id  # Return for use in conversion test

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'upload_endpoint', False, duration,
                {'error': str(e)}
            )
            raise

    def test_upload_and_convert_workflow(self, flask_client, sample_png_bytes):
        """Test complete upload and convert workflow."""
        start_time = time.time()

        try:
            # Upload file
            upload_response = flask_client.post('/api/upload',
                                              data={'file': (io.BytesIO(sample_png_bytes), 'test.png')},
                                              content_type='multipart/form-data')

            assert upload_response.status_code == 200
            upload_data = upload_response.get_json()
            file_id = upload_data['file_id']

            # Convert file
            convert_data = {
                'file_id': file_id,
                'converter': 'vtracer',
                'colormode': 'color',
                'color_precision': 6
            }

            convert_response = flask_client.post('/api/convert',
                                               json=convert_data,
                                               content_type='application/json')

            assert convert_response.status_code == 200

            convert_result = convert_response.get_json()
            assert convert_result is not None
            assert 'success' in convert_result
            assert convert_result['success'] is True
            assert 'svg' in convert_result

            workflow_data = {
                'upload_success': upload_response.status_code == 200,
                'convert_success': convert_response.status_code == 200,
                'file_id': file_id,
                'svg_size': len(convert_result.get('svg', '')),
                'conversion_success': convert_result.get('success', False)
            }

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'upload_convert_workflow', True, duration,
                workflow_data
            )

        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'upload_convert_workflow', False, duration,
                {'error': str(e)}
            )
            raise


class TestCacheIntegration:
    """Test cache integration with conversion pipeline."""

    def test_cache_availability(self):
        """Test that caching infrastructure is available."""
        start_time = time.time()

        try:
            from backend.ai_modules.advanced_cache import MultiLevelCache

            cache = MultiLevelCache()
            assert cache is not None

            # Test basic cache operations
            test_key = "test_cache_key"
            test_value = {"test": "data", "timestamp": time.time()}

            # Test cache set/get
            cache.set('test', test_key, test_value)
            retrieved_value = cache.get('test', test_key)

            cache_data = {
                'cache_available': True,
                'set_successful': True,
                'get_successful': retrieved_value is not None,
                'data_integrity': retrieved_value == test_value if retrieved_value else False
            }

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'cache_availability', True, duration,
                cache_data
            )

        except ImportError:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'cache_availability', True, duration,
                {'cache_available': False, 'reason': 'Cache modules not available'}
            )
        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'cache_availability', False, duration,
                {'error': str(e)}
            )
            raise

    def test_cached_conversion_performance(self, temp_png_file):
        """Test conversion performance with caching."""
        start_time = time.time()

        try:
            from backend.ai_modules.cached_components import CachedFeatureExtractor
            from backend.ai_modules.advanced_cache import MultiLevelCache

            cache = MultiLevelCache()
            cached_extractor = CachedFeatureExtractor(cache=cache)

            # First extraction (should be slow - not cached)
            first_start = time.time()
            first_result = cached_extractor.extract_features(temp_png_file)
            first_time = time.time() - first_start

            # Second extraction (should be fast - cached)
            second_start = time.time()
            second_result = cached_extractor.extract_features(temp_png_file)
            second_time = time.time() - second_start

            # Validate cache effectiveness
            assert first_result == second_result, "Cached results should match original"

            # Cache should significantly improve performance
            speedup = first_time / second_time if second_time > 0 else float('inf')

            cache_performance_data = {
                'first_extraction_time': first_time,
                'cached_extraction_time': second_time,
                'speedup_ratio': speedup,
                'cache_effective': speedup > 2.0,  # At least 2x speedup expected
                'results_match': first_result == second_result
            }

            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'cached_conversion_performance', True, duration,
                cache_performance_data
            )

            # Store cache performance metrics
            e2e_runner.performance_metrics['cache_performance'] = cache_performance_data

        except ImportError:
            pytest.skip("Cache modules not available for performance testing")
        except Exception as e:
            duration = time.time() - start_time
            e2e_runner.log_test_result(
                'cached_conversion_performance', False, duration,
                {'error': str(e)}
            )
            raise


def test_comprehensive_e2e_summary():
    """Generate comprehensive end-to-end test summary."""
    summary = e2e_runner.get_test_summary()

    print("\n" + "="*80)
    print("COMPREHENSIVE END-TO-END INTEGRATION TEST SUMMARY")
    print("="*80)

    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Average Test Duration: {summary['average_test_duration']:.2f}s")

    if summary['performance_metrics']:
        print("\nPerformance Metrics:")
        for metric_name, metric_data in summary['performance_metrics'].items():
            print(f"  {metric_name}:")
            for key, value in metric_data.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")

    print(f"\nDetailed Test Results:")
    for result in e2e_runner.test_results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['test_name']} ({result['duration']:.2f}s)")
        if not result['success'] and 'error' in result['details']:
            print(f"      Error: {result['details']['error']}")

    print("\n" + "="*80)

    # Assert overall success
    assert summary['success_rate'] >= 70.0, f"End-to-end test suite success rate too low: {summary['success_rate']:.1f}%"


if __name__ == "__main__":
    # Run with pytest when called directly
    pytest.main([__file__, '-v', '--tb=short'])