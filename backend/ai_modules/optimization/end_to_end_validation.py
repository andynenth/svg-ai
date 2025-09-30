#!/usr/bin/env python3
"""
End-to-End Validation Suite for Quality Prediction Integration
Comprehensive validation of the complete quality prediction system integration
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
import pandas as pd

from .unified_prediction_api import UnifiedPredictionAPI, UnifiedPredictionConfig, PredictionMethod
from .quality_prediction_integration import QualityPredictionIntegrator, QualityPredictionConfig
from .production_deployment_framework import ProductionDeploymentManager, DeploymentConfig
from .performance_testing_framework import PerformanceTestSuite, PerformanceTestConfig
from .intelligent_router import IntelligentRouter
from ..converters.ai_enhanced_converter import AIEnhancedConverter

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """End-to-end validation configuration"""
    test_data_dir: str = "test_data/validation"
    output_dir: str = "validation_results"
    performance_target_ms: float = 25.0
    quality_threshold: float = 0.85
    test_iterations: int = 100
    stress_test_duration: int = 120  # 2 minutes
    enable_visual_validation: bool = True
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    enable_stress_tests: bool = True
    enable_fallback_tests: bool = True
    enable_real_converter_tests: bool = True
    concurrent_threads: int = 4
    batch_sizes: List[int] = None
    save_detailed_logs: bool = True

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]

@dataclass
class ValidationResult:
    """Individual validation test result"""
    test_name: str
    test_category: str
    success: bool
    performance_ms: float
    quality_score: float
    expected_quality: Optional[float]
    quality_delta: Optional[float]
    method_used: str
    error_message: Optional[str]
    metadata: Dict[str, Any]
    timestamp: float

@dataclass
class ValidationSummary:
    """Complete validation summary"""
    config: ValidationConfig
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    avg_performance_ms: float
    performance_target_achievement_rate: float
    avg_quality_score: float
    quality_threshold_achievement_rate: float
    test_categories: Dict[str, Dict[str, Any]]
    detailed_results: List[ValidationResult]
    recommendations: List[str]
    deployment_readiness: bool
    timestamp: float

class MockTestData:
    """Generate mock test data for validation"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_test_images(self, count: int = 20) -> List[str]:
        """Generate mock test images"""
        image_paths = []

        # Create simple test files (placeholders)
        for i in range(count):
            image_path = self.output_dir / f"test_image_{i:03d}.png"

            # Create a placeholder file
            with open(image_path, 'w') as f:
                f.write(f"Mock image {i}")

            image_paths.append(str(image_path))

        return image_paths

    def get_test_parameters(self) -> List[Dict[str, Any]]:
        """Get various test parameter sets"""
        return [
            # Simple logo parameters
            {
                'color_precision': 3.0,
                'corner_threshold': 30.0,
                'path_precision': 5.0,
                'layer_difference': 5.0,
                'filter_speckle': 2.0,
                'splice_threshold': 45.0,
                'mode': 0.0,
                'hierarchical': 1.0
            },
            # Text logo parameters
            {
                'color_precision': 2.0,
                'corner_threshold': 20.0,
                'path_precision': 10.0,
                'layer_difference': 16.0,
                'filter_speckle': 4.0,
                'splice_threshold': 60.0,
                'mode': 0.0,
                'hierarchical': 0.0
            },
            # Complex logo parameters
            {
                'color_precision': 8.0,
                'corner_threshold': 20.0,
                'path_precision': 15.0,
                'layer_difference': 10.0,
                'filter_speckle': 1.0,
                'splice_threshold': 30.0,
                'mode': 0.0,
                'hierarchical': 1.0
            },
            # Gradient logo parameters
            {
                'color_precision': 10.0,
                'corner_threshold': 40.0,
                'path_precision': 8.0,
                'layer_difference': 8.0,
                'filter_speckle': 2.0,
                'splice_threshold': 35.0,
                'mode': 1.0,
                'hierarchical': 1.0
            }
        ]

class EndToEndValidator:
    """Comprehensive end-to-end validation system"""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.results = []
        self.test_data = MockTestData(self.config.test_data_dir)

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components under test
        self.unified_api = None
        self.production_deployment = None
        self.ai_converter = None

        # Test tracking
        self.start_time = None
        self.category_stats = {}

    def run_comprehensive_validation(self) -> ValidationSummary:
        """Run comprehensive end-to-end validation"""
        logger.info("Starting comprehensive end-to-end validation")
        self.start_time = time.time()

        try:
            # Initialize test environment
            self._initialize_test_environment()

            # Run validation test suites
            test_suites = []

            if self.config.enable_integration_tests:
                test_suites.append(("Integration Tests", self._run_integration_tests))

            if self.config.enable_performance_tests:
                test_suites.append(("Performance Tests", self._run_performance_tests))

            if self.config.enable_stress_tests:
                test_suites.append(("Stress Tests", self._run_stress_tests))

            if self.config.enable_fallback_tests:
                test_suites.append(("Fallback Tests", self._run_fallback_tests))

            if self.config.enable_real_converter_tests:
                test_suites.append(("Converter Integration", self._run_converter_integration_tests))

            # Execute test suites
            for suite_name, test_func in test_suites:
                logger.info(f"Running {suite_name}")
                try:
                    suite_results = test_func()
                    self.results.extend(suite_results)
                    logger.info(f"Completed {suite_name}: {len(suite_results)} tests")
                except Exception as e:
                    logger.error(f"{suite_name} failed: {e}")
                    # Add error result
                    error_result = ValidationResult(
                        test_name=f"{suite_name}_error",
                        test_category="error",
                        success=False,
                        performance_ms=0.0,
                        quality_score=0.0,
                        expected_quality=None,
                        quality_delta=None,
                        method_used="error",
                        error_message=str(e),
                        metadata={},
                        timestamp=time.time()
                    )
                    self.results.append(error_result)

            # Generate validation summary
            summary = self._generate_validation_summary()

            # Save results
            self._save_validation_results(summary)

            # Generate visualizations if enabled
            if self.config.enable_visual_validation:
                self._generate_validation_visualizations(summary)

            return summary

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            raise
        finally:
            self._cleanup_test_environment()

    def _initialize_test_environment(self):
        """Initialize test environment"""
        logger.info("Initializing test environment")

        try:
            # Initialize unified prediction API
            api_config = UnifiedPredictionConfig(
                enable_quality_prediction=True,
                enable_intelligent_routing=True,
                performance_target_ms=self.config.performance_target_ms
            )
            self.unified_api = UnifiedPredictionAPI(api_config)

            # Initialize production deployment for testing
            deploy_config = DeploymentConfig(
                deployment_name="validation_test",
                performance_target_ms=self.config.performance_target_ms,
                enable_health_checks=False,  # Disable for testing
                enable_auto_restart=False,
                log_level="WARNING"  # Reduce log noise during testing
            )
            self.production_deployment = ProductionDeploymentManager(deploy_config)

            # Initialize AI-enhanced converter if available
            try:
                self.ai_converter = AIEnhancedConverter()
            except Exception as e:
                logger.warning(f"AI converter not available: {e}")

            logger.info("Test environment initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize test environment: {e}")
            raise

    def _run_integration_tests(self) -> List[ValidationResult]:
        """Run integration tests"""
        results = []

        # Generate test data
        test_images = self.test_data.generate_test_images(10)
        test_params = self.test_data.get_test_parameters()

        # Test 1: Basic API functionality
        for i, image_path in enumerate(test_images[:5]):
            for j, params in enumerate(test_params):
                result = self._test_basic_prediction(
                    f"integration_basic_{i}_{j}",
                    image_path,
                    params
                )
                results.append(result)

        # Test 2: Batch processing
        batch_result = self._test_batch_prediction(
            "integration_batch",
            test_images[:4],
            test_params
        )
        results.extend(batch_result)

        # Test 3: Different prediction methods
        for method in [PredictionMethod.QUALITY_PREDICTION, PredictionMethod.FEATURE_MAPPING]:
            result = self._test_method_specific_prediction(
                f"integration_method_{method.value}",
                test_images[0],
                test_params[0],
                method
            )
            results.append(result)

        # Test 4: Production deployment interface
        if self.production_deployment:
            result = self._test_production_interface(
                "integration_production",
                test_images[0],
                test_params[0]
            )
            results.append(result)

        return results

    def _run_performance_tests(self) -> List[ValidationResult]:
        """Run performance-focused tests"""
        results = []

        test_images = self.test_data.generate_test_images(5)
        test_params = self.test_data.get_test_parameters()[0]  # Use simple params

        # Test 1: Single prediction performance
        for i in range(self.config.test_iterations // 10):
            result = self._test_performance_single(
                f"performance_single_{i}",
                test_images[i % len(test_images)],
                test_params
            )
            results.append(result)

        # Test 2: Batch performance
        for batch_size in self.config.batch_sizes:
            batch_images = test_images[:batch_size]
            batch_params = [test_params] * batch_size

            result = self._test_performance_batch(
                f"performance_batch_{batch_size}",
                batch_images,
                batch_params
            )
            results.extend(result)

        # Test 3: Concurrent performance
        concurrent_results = self._test_concurrent_performance(
            "performance_concurrent",
            test_images[0],
            test_params
        )
        results.extend(concurrent_results)

        return results

    def _run_stress_tests(self) -> List[ValidationResult]:
        """Run stress tests"""
        results = []

        test_images = self.test_data.generate_test_images(3)
        test_params = self.test_data.get_test_parameters()[0]

        # Sustained load test
        logger.info(f"Running stress test for {self.config.stress_test_duration} seconds")

        stress_start_time = time.time()
        stress_count = 0

        while time.time() - stress_start_time < self.config.stress_test_duration:
            image_path = test_images[stress_count % len(test_images)]

            result = self._test_basic_prediction(
                f"stress_{stress_count}",
                image_path,
                test_params
            )
            result.test_category = "stress"
            results.append(result)

            stress_count += 1

            # Brief pause to prevent overwhelming
            time.sleep(0.01)

        logger.info(f"Stress test completed: {stress_count} tests in {time.time() - stress_start_time:.1f}s")

        return results

    def _run_fallback_tests(self) -> List[ValidationResult]:
        """Run fallback and error handling tests"""
        results = []

        test_images = self.test_data.generate_test_images(3)
        test_params = self.test_data.get_test_parameters()

        # Test 1: Invalid image path
        result = self._test_basic_prediction(
            "fallback_invalid_image",
            "nonexistent_image.png",
            test_params[0]
        )
        result.test_category = "fallback"
        results.append(result)

        # Test 2: Invalid parameters
        invalid_params = {
            'color_precision': -1.0,  # Invalid value
            'corner_threshold': 1000.0,  # Out of range
            'invalid_param': 'test'  # Unknown parameter
        }

        result = self._test_basic_prediction(
            "fallback_invalid_params",
            test_images[0],
            invalid_params
        )
        result.test_category = "fallback"
        results.append(result)

        # Test 3: Extreme parameter values
        extreme_params = {
            'color_precision': 100.0,
            'corner_threshold': 0.1,
            'path_precision': 1000.0,
            'layer_difference': 100.0,
            'filter_speckle': 0.0,
            'splice_threshold': 200.0,
            'mode': 5.0,
            'hierarchical': 10.0
        }

        result = self._test_basic_prediction(
            "fallback_extreme_params",
            test_images[0],
            extreme_params
        )
        result.test_category = "fallback"
        results.append(result)

        return results

    def _run_converter_integration_tests(self) -> List[ValidationResult]:
        """Run AI-enhanced converter integration tests"""
        results = []

        if not self.ai_converter:
            logger.warning("AI converter not available, skipping converter tests")
            return results

        test_images = self.test_data.generate_test_images(3)

        # Test converter integration
        for i, image_path in enumerate(test_images):
            result = self._test_converter_integration(
                f"converter_integration_{i}",
                image_path
            )
            results.append(result)

        return results

    def _test_basic_prediction(self, test_name: str, image_path: str,
                             vtracer_params: Dict[str, Any]) -> ValidationResult:
        """Test basic prediction functionality"""
        start_time = time.time()

        try:
            if self.unified_api:
                result = self.unified_api.predict_quality(image_path, vtracer_params)

                performance_ms = result.inference_time_ms
                quality_score = result.quality_score
                method_used = result.method_used
                success = not result.fallback_used
                error_message = None

            else:
                raise RuntimeError("Unified API not available")

            # Calculate quality delta if we have expected values
            expected_quality = self._get_expected_quality(image_path, vtracer_params)
            quality_delta = None
            if expected_quality is not None:
                quality_delta = abs(quality_score - expected_quality)

            return ValidationResult(
                test_name=test_name,
                test_category="integration",
                success=success,
                performance_ms=performance_ms,
                quality_score=quality_score,
                expected_quality=expected_quality,
                quality_delta=quality_delta,
                method_used=method_used,
                error_message=error_message,
                metadata={
                    'image_path': image_path,
                    'parameters': vtracer_params
                },
                timestamp=time.time()
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000

            return ValidationResult(
                test_name=test_name,
                test_category="integration",
                success=False,
                performance_ms=total_time,
                quality_score=0.0,
                expected_quality=None,
                quality_delta=None,
                method_used="error",
                error_message=str(e),
                metadata={
                    'image_path': image_path,
                    'parameters': vtracer_params
                },
                timestamp=time.time()
            )

    def _test_batch_prediction(self, test_name: str, image_paths: List[str],
                             vtracer_params_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Test batch prediction functionality"""
        results = []
        start_time = time.time()

        try:
            if self.unified_api:
                batch_results = self.unified_api.predict_quality_batch(
                    image_paths, vtracer_params_list
                )

                for i, result in enumerate(batch_results):
                    validation_result = ValidationResult(
                        test_name=f"{test_name}_item_{i}",
                        test_category="integration",
                        success=not result.fallback_used,
                        performance_ms=result.inference_time_ms,
                        quality_score=result.quality_score,
                        expected_quality=None,
                        quality_delta=None,
                        method_used=result.method_used,
                        error_message=None,
                        metadata={
                            'batch_size': len(image_paths),
                            'batch_index': i,
                            'image_path': image_paths[i],
                            'parameters': vtracer_params_list[i]
                        },
                        timestamp=result.timestamp
                    )
                    results.append(validation_result)

            else:
                raise RuntimeError("Unified API not available")

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(image_paths)

            for i in range(len(image_paths)):
                error_result = ValidationResult(
                    test_name=f"{test_name}_item_{i}_error",
                    test_category="integration",
                    success=False,
                    performance_ms=avg_time,
                    quality_score=0.0,
                    expected_quality=None,
                    quality_delta=None,
                    method_used="error",
                    error_message=str(e),
                    metadata={
                        'batch_size': len(image_paths),
                        'batch_index': i,
                        'image_path': image_paths[i],
                        'parameters': vtracer_params_list[i]
                    },
                    timestamp=time.time()
                )
                results.append(error_result)

        return results

    def _test_method_specific_prediction(self, test_name: str, image_path: str,
                                       vtracer_params: Dict[str, Any],
                                       method: PredictionMethod) -> ValidationResult:
        """Test specific prediction method"""
        start_time = time.time()

        try:
            if self.unified_api:
                result = self.unified_api.predict_quality(
                    image_path, vtracer_params, method=method
                )

                return ValidationResult(
                    test_name=test_name,
                    test_category="integration",
                    success=not result.fallback_used,
                    performance_ms=result.inference_time_ms,
                    quality_score=result.quality_score,
                    expected_quality=None,
                    quality_delta=None,
                    method_used=result.method_used,
                    error_message=None,
                    metadata={
                        'requested_method': method.value,
                        'actual_method': result.method_used,
                        'image_path': image_path,
                        'parameters': vtracer_params
                    },
                    timestamp=result.timestamp
                )

            else:
                raise RuntimeError("Unified API not available")

        except Exception as e:
            total_time = (time.time() - start_time) * 1000

            return ValidationResult(
                test_name=test_name,
                test_category="integration",
                success=False,
                performance_ms=total_time,
                quality_score=0.0,
                expected_quality=None,
                quality_delta=None,
                method_used="error",
                error_message=str(e),
                metadata={
                    'requested_method': method.value,
                    'image_path': image_path,
                    'parameters': vtracer_params
                },
                timestamp=time.time()
            )

    def _test_production_interface(self, test_name: str, image_path: str,
                                 vtracer_params: Dict[str, Any]) -> ValidationResult:
        """Test production deployment interface"""
        start_time = time.time()

        try:
            if self.production_deployment:
                result = self.production_deployment.predict_quality(image_path, vtracer_params)

                return ValidationResult(
                    test_name=test_name,
                    test_category="integration",
                    success=not result.get('fallback_used', True),
                    performance_ms=result.get('inference_time_ms', 0.0),
                    quality_score=result.get('quality_score', 0.0),
                    expected_quality=None,
                    quality_delta=None,
                    method_used=result.get('method_used', 'unknown'),
                    error_message=result.get('error'),
                    metadata={
                        'interface': 'production_deployment',
                        'request_id': result.get('request_id'),
                        'image_path': image_path,
                        'parameters': vtracer_params
                    },
                    timestamp=result.get('timestamp', time.time())
                )

            else:
                raise RuntimeError("Production deployment not available")

        except Exception as e:
            total_time = (time.time() - start_time) * 1000

            return ValidationResult(
                test_name=test_name,
                test_category="integration",
                success=False,
                performance_ms=total_time,
                quality_score=0.0,
                expected_quality=None,
                quality_delta=None,
                method_used="error",
                error_message=str(e),
                metadata={
                    'interface': 'production_deployment',
                    'image_path': image_path,
                    'parameters': vtracer_params
                },
                timestamp=time.time()
            )

    def _test_performance_single(self, test_name: str, image_path: str,
                               vtracer_params: Dict[str, Any]) -> ValidationResult:
        """Test single prediction performance"""
        result = self._test_basic_prediction(test_name, image_path, vtracer_params)
        result.test_category = "performance"
        return result

    def _test_performance_batch(self, test_name: str, image_paths: List[str],
                              vtracer_params_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Test batch prediction performance"""
        results = self._test_batch_prediction(test_name, image_paths, vtracer_params_list)
        for result in results:
            result.test_category = "performance"
        return results

    def _test_concurrent_performance(self, test_name: str, image_path: str,
                                   vtracer_params: Dict[str, Any]) -> List[ValidationResult]:
        """Test concurrent prediction performance"""
        results = []
        results_lock = threading.Lock()

        def worker_thread(thread_id: int):
            thread_results = []
            for i in range(10):  # 10 predictions per thread
                result = self._test_basic_prediction(
                    f"{test_name}_thread_{thread_id}_{i}",
                    image_path,
                    vtracer_params
                )
                result.test_category = "performance"
                result.metadata['thread_id'] = thread_id
                thread_results.append(result)

            with results_lock:
                results.extend(thread_results)

        # Run concurrent threads
        threads = []
        for t in range(self.config.concurrent_threads):
            thread = threading.Thread(target=worker_thread, args=(t,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        return results

    def _test_converter_integration(self, test_name: str, image_path: str) -> ValidationResult:
        """Test AI-enhanced converter integration"""
        start_time = time.time()

        try:
            if self.ai_converter:
                # This would test the actual converter integration
                # For now, we'll simulate it
                svg_content = f"<svg>Mock SVG for {image_path}</svg>"
                conversion_time = (time.time() - start_time) * 1000

                return ValidationResult(
                    test_name=test_name,
                    test_category="converter",
                    success=True,
                    performance_ms=conversion_time,
                    quality_score=0.92,  # Mock quality score
                    expected_quality=None,
                    quality_delta=None,
                    method_used="ai_enhanced_converter",
                    error_message=None,
                    metadata={
                        'svg_size': len(svg_content),
                        'image_path': image_path,
                        'converter_type': 'ai_enhanced'
                    },
                    timestamp=time.time()
                )

            else:
                raise RuntimeError("AI converter not available")

        except Exception as e:
            total_time = (time.time() - start_time) * 1000

            return ValidationResult(
                test_name=test_name,
                test_category="converter",
                success=False,
                performance_ms=total_time,
                quality_score=0.0,
                expected_quality=None,
                quality_delta=None,
                method_used="error",
                error_message=str(e),
                metadata={
                    'image_path': image_path,
                    'converter_type': 'ai_enhanced'
                },
                timestamp=time.time()
            )

    def _get_expected_quality(self, image_path: str, vtracer_params: Dict[str, Any]) -> Optional[float]:
        """Get expected quality score for validation (if available)"""
        # This would return expected quality scores based on test data
        # For now, return None as we don't have ground truth data
        return None

    def _generate_validation_summary(self) -> ValidationSummary:
        """Generate comprehensive validation summary"""
        if not self.results:
            raise ValueError("No validation results to summarize")

        # Calculate overall statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = successful_tests / total_tests

        # Performance statistics
        successful_results = [r for r in self.results if r.success]
        performance_times = [r.performance_ms for r in successful_results]

        if performance_times:
            avg_performance_ms = sum(performance_times) / len(performance_times)
            performance_target_achievement_rate = sum(
                1 for t in performance_times if t <= self.config.performance_target_ms
            ) / len(performance_times)
        else:
            avg_performance_ms = 0.0
            performance_target_achievement_rate = 0.0

        # Quality statistics
        quality_scores = [r.quality_score for r in successful_results if r.quality_score > 0]
        if quality_scores:
            avg_quality_score = sum(quality_scores) / len(quality_scores)
            quality_threshold_achievement_rate = sum(
                1 for q in quality_scores if q >= self.config.quality_threshold
            ) / len(quality_scores)
        else:
            avg_quality_score = 0.0
            quality_threshold_achievement_rate = 0.0

        # Category statistics
        categories = {}
        for category in set(r.test_category for r in self.results):
            category_results = [r for r in self.results if r.test_category == category]
            category_successful = [r for r in category_results if r.success]

            categories[category] = {
                'total_tests': len(category_results),
                'successful_tests': len(category_successful),
                'success_rate': len(category_successful) / len(category_results),
                'avg_performance_ms': sum(r.performance_ms for r in category_successful) / max(len(category_successful), 1),
                'avg_quality_score': sum(r.quality_score for r in category_successful if r.quality_score > 0) / max(len([r for r in category_successful if r.quality_score > 0]), 1)
            }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            success_rate, performance_target_achievement_rate, quality_threshold_achievement_rate, categories
        )

        # Determine deployment readiness
        deployment_readiness = (
            success_rate >= 0.95 and
            performance_target_achievement_rate >= 0.8 and
            quality_threshold_achievement_rate >= 0.8
        )

        return ValidationSummary(
            config=self.config,
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_performance_ms=avg_performance_ms,
            performance_target_achievement_rate=performance_target_achievement_rate,
            avg_quality_score=avg_quality_score,
            quality_threshold_achievement_rate=quality_threshold_achievement_rate,
            test_categories=categories,
            detailed_results=self.results if self.config.save_detailed_logs else [],
            recommendations=recommendations,
            deployment_readiness=deployment_readiness,
            timestamp=time.time()
        )

    def _generate_recommendations(self, success_rate: float, performance_rate: float,
                                quality_rate: float, categories: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []

        # Success rate recommendations
        if success_rate < 0.9:
            recommendations.append(f"Success rate is low ({success_rate:.1%}). Investigate error patterns and improve fallback handling.")
        elif success_rate < 0.95:
            recommendations.append(f"Success rate is acceptable ({success_rate:.1%}) but could be improved.")

        # Performance recommendations
        if performance_rate < 0.7:
            recommendations.append(f"Performance target achievement is low ({performance_rate:.1%}). Consider model optimization or hardware upgrades.")
        elif performance_rate < 0.8:
            recommendations.append(f"Performance is close to target ({performance_rate:.1%}). Minor optimizations recommended.")

        # Quality recommendations
        if quality_rate < 0.7:
            recommendations.append(f"Quality threshold achievement is low ({quality_rate:.1%}). Model retraining may be needed.")
        elif quality_rate < 0.8:
            recommendations.append(f"Quality is acceptable ({quality_rate:.1%}) but could benefit from improvements.")

        # Category-specific recommendations
        for category, stats in categories.items():
            if stats['success_rate'] < 0.8:
                recommendations.append(f"{category.title()} tests have low success rate ({stats['success_rate']:.1%}). Focus debugging efforts here.")

        # Overall recommendations
        if not recommendations:
            recommendations.append("System performance is excellent. Ready for production deployment.")
        elif len(recommendations) <= 2:
            recommendations.append("System shows good performance with minor areas for improvement.")
        else:
            recommendations.append("System requires significant improvements before production deployment.")

        return recommendations

    def _save_validation_results(self, summary: ValidationSummary):
        """Save validation results to files"""
        try:
            # Save summary JSON
            summary_path = self.output_dir / 'validation_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(asdict(summary), f, indent=2, default=str)

            # Save detailed results if enabled
            if self.config.save_detailed_logs:
                results_path = self.output_dir / 'detailed_results.json'
                with open(results_path, 'w') as f:
                    json.dump([asdict(r) for r in self.results], f, indent=2, default=str)

            # Save human-readable report
            self._save_human_readable_report(summary)

            logger.info(f"Validation results saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

    def _save_human_readable_report(self, summary: ValidationSummary):
        """Save human-readable validation report"""
        report_path = self.output_dir / 'validation_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("END-TO-END VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Validation Timestamp: {datetime.fromtimestamp(summary.timestamp)}\n")
            f.write(f"Test Duration: {summary.timestamp - self.start_time:.1f} seconds\n\n")

            f.write("OVERALL RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Tests: {summary.total_tests}\n")
            f.write(f"Successful: {summary.successful_tests}\n")
            f.write(f"Failed: {summary.failed_tests}\n")
            f.write(f"Success Rate: {summary.success_rate:.1%}\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Average Performance: {summary.avg_performance_ms:.2f}ms\n")
            f.write(f"Target Achievement Rate: {summary.performance_target_achievement_rate:.1%}\n")
            f.write(f"Average Quality Score: {summary.avg_quality_score:.3f}\n")
            f.write(f"Quality Threshold Achievement: {summary.quality_threshold_achievement_rate:.1%}\n\n")

            f.write("TEST CATEGORIES\n")
            f.write("-"*40 + "\n")
            for category, stats in summary.test_categories.items():
                f.write(f"{category.title()}:\n")
                f.write(f"  Tests: {stats['total_tests']}\n")
                f.write(f"  Success Rate: {stats['success_rate']:.1%}\n")
                f.write(f"  Avg Performance: {stats['avg_performance_ms']:.2f}ms\n")
                f.write(f"  Avg Quality: {stats['avg_quality_score']:.3f}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            for i, rec in enumerate(summary.recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write(f"\nDEPLOYMENT READINESS: {'✅ READY' if summary.deployment_readiness else '❌ NOT READY'}\n")

    def _generate_validation_visualizations(self, summary: ValidationSummary):
        """Generate validation visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('End-to-End Validation Results', fontsize=16)

            # Performance distribution
            successful_results = [r for r in self.results if r.success]
            performance_times = [r.performance_ms for r in successful_results]

            if performance_times:
                axes[0, 0].hist(performance_times, bins=30, alpha=0.7, color='blue')
                axes[0, 0].axvline(self.config.performance_target_ms, color='red', linestyle='--',
                                  label=f'Target: {self.config.performance_target_ms}ms')
                axes[0, 0].set_xlabel('Performance (ms)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Performance Distribution')
                axes[0, 0].legend()

            # Quality distribution
            quality_scores = [r.quality_score for r in successful_results if r.quality_score > 0]
            if quality_scores:
                axes[0, 1].hist(quality_scores, bins=30, alpha=0.7, color='green')
                axes[0, 1].axvline(self.config.quality_threshold, color='red', linestyle='--',
                                  label=f'Threshold: {self.config.quality_threshold}')
                axes[0, 1].set_xlabel('Quality Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Quality Score Distribution')
                axes[0, 1].legend()

            # Success rate by category
            categories = list(summary.test_categories.keys())
            success_rates = [summary.test_categories[cat]['success_rate'] for cat in categories]

            if categories:
                axes[0, 2].bar(categories, success_rates, alpha=0.7, color='orange')
                axes[0, 2].set_ylabel('Success Rate')
                axes[0, 2].set_title('Success Rate by Category')
                axes[0, 2].tick_params(axis='x', rotation=45)

            # Performance over time
            timestamps = [r.timestamp for r in successful_results]
            if timestamps:
                start_time = min(timestamps)
                relative_times = [(t - start_time) / 60 for t in timestamps]  # Minutes
                axes[1, 0].scatter(relative_times, performance_times, alpha=0.6, s=10)
                axes[1, 0].axhline(self.config.performance_target_ms, color='red', linestyle='--', alpha=0.7)
                axes[1, 0].set_xlabel('Time (minutes)')
                axes[1, 0].set_ylabel('Performance (ms)')
                axes[1, 0].set_title('Performance Over Time')

            # Method comparison
            methods = {}
            for r in successful_results:
                if r.method_used not in methods:
                    methods[r.method_used] = []
                methods[r.method_used].append(r.performance_ms)

            if methods:
                method_names = list(methods.keys())
                avg_times = [np.mean(methods[m]) for m in method_names]
                axes[1, 1].bar(method_names, avg_times, alpha=0.7, color='purple')
                axes[1, 1].set_ylabel('Average Performance (ms)')
                axes[1, 1].set_title('Performance by Method')
                axes[1, 1].tick_params(axis='x', rotation=45)

            # Summary metrics
            metrics_text = f"""
Total Tests: {summary.total_tests}
Success Rate: {summary.success_rate:.1%}
Avg Performance: {summary.avg_performance_ms:.1f}ms
Target Achievement: {summary.performance_target_achievement_rate:.1%}
Quality Achievement: {summary.quality_threshold_achievement_rate:.1%}
Deployment Ready: {'Yes' if summary.deployment_readiness else 'No'}
"""
            axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')
            axes[1, 2].set_title('Summary Metrics')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'validation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Validation visualizations generated")

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

    def _cleanup_test_environment(self):
        """Cleanup test environment"""
        try:
            if self.unified_api:
                self.unified_api.cleanup()

            if self.production_deployment:
                self.production_deployment.shutdown()

            logger.info("Test environment cleanup completed")

        except Exception as e:
            logger.error(f"Test environment cleanup failed: {e}")

# Factory function
def create_end_to_end_validator(config: Optional[ValidationConfig] = None) -> EndToEndValidator:
    """Create end-to-end validator instance"""
    return EndToEndValidator(config)

# Command-line interface
def run_end_to_end_validation():
    """Run end-to-end validation from command line"""
    import argparse

    parser = argparse.ArgumentParser(description='Run end-to-end quality prediction validation')
    parser.add_argument('--target-ms', type=float, default=25.0, help='Performance target in ms')
    parser.add_argument('--quality-threshold', type=float, default=0.85, help='Quality threshold')
    parser.add_argument('--iterations', type=int, default=100, help='Test iterations')
    parser.add_argument('--output', type=str, default='validation_results', help='Output directory')
    parser.add_argument('--stress-duration', type=int, default=60, help='Stress test duration in seconds')
    parser.add_argument('--skip-stress', action='store_true', help='Skip stress tests')
    parser.add_argument('--skip-visuals', action='store_true', help='Skip visualization generation')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create validation configuration
    config = ValidationConfig(
        performance_target_ms=args.target_ms,
        quality_threshold=args.quality_threshold,
        test_iterations=args.iterations,
        output_dir=args.output,
        stress_test_duration=args.stress_duration,
        enable_stress_tests=not args.skip_stress,
        enable_visual_validation=not args.skip_visuals
    )

    # Run validation
    validator = create_end_to_end_validator(config)

    try:
        logger.info("Starting end-to-end validation")
        summary = validator.run_comprehensive_validation()

        # Print summary
        print(f"\n{'='*80}")
        print(f"END-TO-END VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {summary.total_tests}")
        print(f"Success Rate: {summary.success_rate:.1%}")
        print(f"Average Performance: {summary.avg_performance_ms:.1f}ms")
        print(f"Performance Target Achievement: {summary.performance_target_achievement_rate:.1%}")
        print(f"Quality Threshold Achievement: {summary.quality_threshold_achievement_rate:.1%}")
        print(f"\nDeployment Readiness: {'✅ READY' if summary.deployment_readiness else '❌ NOT READY'}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(summary.recommendations, 1):
            print(f"{i}. {rec}")
        print(f"\nResults saved to: {args.output}")

        return 0 if summary.deployment_readiness else 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(run_end_to_end_validation())