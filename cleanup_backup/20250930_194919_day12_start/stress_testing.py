#!/usr/bin/env python3
"""
Stress Testing & Reliability - Task 4 Implementation
Comprehensive stress testing to validate system reliability and stability.
"""

import sys
import time
import json
import logging
import argparse
import threading
import tempfile
import resource
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import psutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
try:
    from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
    from backend.converters.ai_enhanced_converter import AIEnhancedConverter
    from backend.converters.vtracer_converter import VTracerConverter
except ImportError as e:
    print(f"Warning: Failed to import required modules: {e}")
    print("Some stress tests may not be available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Container for stress test results."""
    test_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    requests_sent: int
    requests_successful: int
    requests_failed: int
    success_rate: float
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    cpu_usage_avg: float
    errors: List[str]
    meets_reliability_target: bool

    def __post_init__(self):
        self.success_rate = self.requests_successful / max(1, self.requests_sent)
        self.meets_reliability_target = self.success_rate >= 0.95  # 95% reliability target


class StressTester:
    """
    Comprehensive stress testing system for AI pipeline reliability validation.
    """

    def __init__(self):
        """Initialize stress tester."""
        self.results: List[StressTestResult] = []
        self.pipeline = None
        self.test_images = []
        self._initialize_components()
        self._load_test_images()

        # Stress testing configuration
        self.reliability_target = 0.95  # 95% success rate
        self.concurrent_target = 10     # Handle 10 concurrent requests
        self.memory_limit_mb = 1000     # Memory limit for stress testing

        logger.info("Stress tester initialized")

    def _initialize_components(self):
        """Initialize pipeline for stress testing."""
        try:
            self.pipeline = UnifiedAIPipeline(
                enable_caching=True,
                enable_fallbacks=True,
                performance_mode="balanced"
            )
            logger.info("✓ Pipeline initialized for stress testing")
        except Exception as e:
            logger.error(f"✗ Failed to initialize pipeline: {e}")
            self.pipeline = None

    def _load_test_images(self):
        """Load test images for stress testing."""
        base_path = Path("data/logos")
        if not base_path.exists():
            logger.warning(f"Test data path {base_path} not found")
            return

        categories = ["simple_geometric", "text_based", "gradients", "complex", "abstract"]
        for category in categories:
            category_path = base_path / category
            if category_path.exists():
                # Get 2 images per category for stress testing
                category_images = list(category_path.glob("*.png"))
                category_images = [
                    str(img) for img in category_images
                    if "optimized" not in str(img) and ".cache" not in str(img)
                ][:2]
                self.test_images.extend(category_images)

        logger.info(f"Loaded {len(self.test_images)} images for stress testing")

    def test_high_load(self, concurrent_requests: int = 10, total_requests: int = 100) -> StressTestResult:
        """Test system under high load."""
        logger.info(f"Starting high load test: {concurrent_requests} concurrent, {total_requests} total requests")

        if not self.pipeline or not self.test_images:
            logger.error("Pipeline or test images not available")
            return self._create_failed_result("high_load", "Pipeline or test images not available")

        start_time = datetime.now()
        memory_start = self._get_memory_usage()

        # Track results
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        cpu_samples = []

        try:
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                # Submit requests
                futures = []
                for i in range(total_requests):
                    # Cycle through test images
                    image_path = self.test_images[i % len(self.test_images)]
                    future = executor.submit(self._process_request_with_timing, image_path, f"request_{i}")
                    futures.append(future)

                # Collect results with progress monitoring
                for i, future in enumerate(as_completed(futures)):
                    try:
                        success, duration, error_msg = future.result(timeout=60)

                        if success:
                            successful_requests += 1
                            response_times.append(duration)
                        else:
                            failed_requests += 1
                            if error_msg:
                                errors.append(error_msg)

                        # Sample CPU usage
                        cpu_samples.append(psutil.cpu_percent())

                        # Progress logging
                        if (i + 1) % 10 == 0:
                            logger.info(f"  Completed {i + 1}/{total_requests} requests")

                    except Exception as e:
                        failed_requests += 1
                        errors.append(f"Request processing exception: {str(e)}")

        except Exception as e:
            logger.error(f"High load test failed: {e}")
            return self._create_failed_result("high_load", str(e))

        # Calculate metrics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        memory_end = self._get_memory_usage()
        memory_peak = max(memory_start, memory_end)  # Simple peak estimation

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

        result = StressTestResult(
            test_name="high_load",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            requests_sent=total_requests,
            requests_successful=successful_requests,
            requests_failed=failed_requests,
            success_rate=successful_requests / total_requests,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
            memory_peak_mb=memory_peak,
            cpu_usage_avg=avg_cpu,
            errors=errors[:10],  # Keep first 10 errors
            meets_reliability_target=successful_requests / total_requests >= self.reliability_target
        )

        self.results.append(result)

        logger.info(f"High load test completed:")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Success rate: {result.success_rate:.1%}")
        logger.info(f"  Avg response time: {avg_response_time:.3f}s")
        logger.info(f"  Memory usage: {memory_start:.1f} → {memory_end:.1f} MB")
        logger.info(f"  Meets reliability target: {'✓' if result.meets_reliability_target else '✗'}")

        return result

    def test_resource_limits(self, memory_limit_mb: int = 500) -> StressTestResult:
        """Test with limited resources."""
        logger.info(f"Starting resource limits test with {memory_limit_mb}MB memory limit")

        if not self.pipeline or not self.test_images:
            logger.error("Pipeline or test images not available")
            return self._create_failed_result("resource_limits", "Pipeline or test images not available")

        start_time = datetime.now()
        memory_start = self._get_memory_usage()

        # Test configuration
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        memory_violations = 0

        try:
            # Process images while monitoring memory
            for i, image_path in enumerate(self.test_images * 5):  # Process each image 5 times
                try:
                    # Check memory before processing
                    current_memory = self._get_memory_usage()
                    if current_memory > memory_limit_mb:
                        memory_violations += 1
                        logger.warning(f"Memory limit exceeded: {current_memory:.1f}MB > {memory_limit_mb}MB")

                        # Force garbage collection
                        gc.collect()
                        current_memory = self._get_memory_usage()

                        if current_memory > memory_limit_mb:
                            failed_requests += 1
                            errors.append(f"Memory limit violation: {current_memory:.1f}MB")
                            continue

                    # Process request
                    success, duration, error_msg = self._process_request_with_timing(image_path, f"resource_test_{i}")

                    if success:
                        successful_requests += 1
                        response_times.append(duration)
                    else:
                        failed_requests += 1
                        if error_msg:
                            errors.append(error_msg)

                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Processed {i + 1} requests, memory: {current_memory:.1f}MB")

                except Exception as e:
                    failed_requests += 1
                    errors.append(f"Resource test exception: {str(e)}")

        except Exception as e:
            logger.error(f"Resource limits test failed: {e}")
            return self._create_failed_result("resource_limits", str(e))

        # Calculate metrics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        memory_end = self._get_memory_usage()
        total_requests = successful_requests + failed_requests

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0

        result = StressTestResult(
            test_name="resource_limits",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            requests_sent=total_requests,
            requests_successful=successful_requests,
            requests_failed=failed_requests,
            success_rate=successful_requests / max(1, total_requests),
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
            memory_peak_mb=max(memory_start, memory_end),
            cpu_usage_avg=psutil.cpu_percent(),
            errors=errors[:10],
            meets_reliability_target=memory_violations == 0 and successful_requests > 0
        )

        self.results.append(result)

        logger.info(f"Resource limits test completed:")
        logger.info(f"  Memory violations: {memory_violations}")
        logger.info(f"  Success rate: {result.success_rate:.1%}")
        logger.info(f"  Memory stable: {'✓' if memory_violations == 0 else '✗'}")

        return result

    def test_long_running(self, duration_minutes: int = 60, requests_per_minute: int = 10) -> StressTestResult:
        """Test system stability over time."""
        logger.info(f"Starting long running test: {duration_minutes} minutes, {requests_per_minute} req/min")

        if not self.pipeline or not self.test_images:
            logger.error("Pipeline or test images not available")
            return self._create_failed_result("long_running", "Pipeline or test images not available")

        start_time = datetime.now()
        end_target = start_time + timedelta(minutes=duration_minutes)
        memory_start = self._get_memory_usage()

        # Track metrics over time
        successful_requests = 0
        failed_requests = 0
        response_times = []
        errors = []
        memory_samples = []
        cpu_samples = []

        request_count = 0

        try:
            while datetime.now() < end_target:
                # Process requests for this minute
                minute_start = time.time()

                for _ in range(requests_per_minute):
                    if datetime.now() >= end_target:
                        break

                    # Select image cyclically
                    image_path = self.test_images[request_count % len(self.test_images)]

                    try:
                        success, duration, error_msg = self._process_request_with_timing(
                            image_path, f"long_running_{request_count}"
                        )

                        if success:
                            successful_requests += 1
                            response_times.append(duration)
                        else:
                            failed_requests += 1
                            if error_msg:
                                errors.append(error_msg)

                        request_count += 1

                        # Sample system metrics
                        memory_samples.append(self._get_memory_usage())
                        cpu_samples.append(psutil.cpu_percent())

                    except Exception as e:
                        failed_requests += 1
                        errors.append(f"Long running exception: {str(e)}")

                # Wait for rest of minute
                minute_elapsed = time.time() - minute_start
                if minute_elapsed < 60:
                    time.sleep(60 - minute_elapsed)

                # Log progress
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                current_memory = self._get_memory_usage()
                logger.info(f"  {elapsed_minutes:.1f}min: {successful_requests} successful, "
                          f"memory: {current_memory:.1f}MB")

        except Exception as e:
            logger.error(f"Long running test failed: {e}")
            return self._create_failed_result("long_running", str(e))

        # Calculate final metrics
        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()
        memory_end = self._get_memory_usage()
        memory_peak = max(memory_samples) if memory_samples else memory_end

        # Check for memory leaks (significant growth over time)
        memory_growth = memory_end - memory_start
        memory_stable = memory_growth < 100  # Less than 100MB growth considered stable

        # Check for performance degradation
        if len(response_times) >= 20:
            early_times = response_times[:10]
            late_times = response_times[-10:]
            early_avg = sum(early_times) / len(early_times)
            late_avg = sum(late_times) / len(late_times)
            performance_degradation = (late_avg - early_avg) / early_avg > 0.5  # 50% degradation threshold
        else:
            performance_degradation = False

        total_requests = successful_requests + failed_requests
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

        result = StressTestResult(
            test_name="long_running",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=actual_duration,
            requests_sent=total_requests,
            requests_successful=successful_requests,
            requests_failed=failed_requests,
            success_rate=successful_requests / max(1, total_requests),
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
            memory_peak_mb=memory_peak,
            cpu_usage_avg=avg_cpu,
            errors=errors[:10],
            meets_reliability_target=(
                memory_stable and
                not performance_degradation and
                successful_requests / max(1, total_requests) >= self.reliability_target
            )
        )

        self.results.append(result)

        logger.info(f"Long running test completed:")
        logger.info(f"  Actual duration: {actual_duration/60:.1f} minutes")
        logger.info(f"  Total requests: {total_requests}")
        logger.info(f"  Success rate: {result.success_rate:.1%}")
        logger.info(f"  Memory growth: {memory_growth:+.1f}MB")
        logger.info(f"  Memory stable: {'✓' if memory_stable else '✗'}")
        logger.info(f"  Performance stable: {'✓' if not performance_degradation else '✗'}")

        return result

    def test_error_recovery(self) -> StressTestResult:
        """Test recovery from failures."""
        logger.info("Starting error recovery test")

        start_time = datetime.now()
        memory_start = self._get_memory_usage()

        recovery_tests = []

        # Test 1: Invalid image path
        logger.info("  Testing recovery from invalid image path...")
        recovery_tests.append(self._test_invalid_image_recovery())

        # Test 2: Corrupted image data (simulate)
        logger.info("  Testing recovery from processing errors...")
        recovery_tests.append(self._test_processing_error_recovery())

        # Test 3: Memory pressure simulation
        logger.info("  Testing recovery from resource constraints...")
        recovery_tests.append(self._test_resource_pressure_recovery())

        # Test 4: Concurrent error handling
        logger.info("  Testing concurrent error handling...")
        recovery_tests.append(self._test_concurrent_error_recovery())

        # Calculate overall recovery metrics
        total_recovery_tests = len(recovery_tests)
        successful_recoveries = sum(1 for test in recovery_tests if test.get('recovered', False))

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        memory_end = self._get_memory_usage()

        all_errors = []
        for test in recovery_tests:
            all_errors.extend(test.get('errors', []))

        result = StressTestResult(
            test_name="error_recovery",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            requests_sent=total_recovery_tests,
            requests_successful=successful_recoveries,
            requests_failed=total_recovery_tests - successful_recoveries,
            success_rate=successful_recoveries / total_recovery_tests,
            avg_response_time=duration / total_recovery_tests,
            max_response_time=duration,
            min_response_time=0,
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
            memory_peak_mb=max(memory_start, memory_end),
            cpu_usage_avg=psutil.cpu_percent(),
            errors=all_errors[:10],
            meets_reliability_target=successful_recoveries >= total_recovery_tests * 0.75  # 75% recovery rate
        )

        self.results.append(result)

        logger.info(f"Error recovery test completed:")
        logger.info(f"  Recovery tests: {successful_recoveries}/{total_recovery_tests}")
        logger.info(f"  Recovery rate: {result.success_rate:.1%}")
        logger.info(f"  System recovers from failures: {'✓' if result.meets_reliability_target else '✗'}")

        return result

    def _test_invalid_image_recovery(self) -> Dict[str, Any]:
        """Test recovery from invalid image paths."""
        try:
            if not self.pipeline:
                return {'recovered': False, 'errors': ['Pipeline not available']}

            # Try to process non-existent image
            result = self.pipeline.process("/nonexistent/image.png")

            # Check if system handled error gracefully
            if result is not None and hasattr(result, 'error_message'):
                return {'recovered': True, 'errors': []}
            else:
                return {'recovered': False, 'errors': ['No error handling for invalid image']}

        except Exception as e:
            # System should not crash
            return {'recovered': False, 'errors': [f'System crashed on invalid image: {str(e)}']}

    def _test_processing_error_recovery(self) -> Dict[str, Any]:
        """Test recovery from processing errors."""
        try:
            if not self.pipeline or not self.test_images:
                return {'recovered': False, 'errors': ['Pipeline or test images not available']}

            # Test with valid image but force error conditions
            test_image = self.test_images[0]

            # Try processing with extreme parameters that might cause issues
            result = self.pipeline.process(
                image_path=test_image,
                target_quality=-1.0,  # Invalid quality
                time_constraint=0.001  # Impossible time constraint
            )

            # Check if system handled errors gracefully
            if result is not None:
                return {'recovered': True, 'errors': []}
            else:
                return {'recovered': False, 'errors': ['System failed to handle processing errors']}

        except Exception as e:
            return {'recovered': False, 'errors': [f'Processing error recovery failed: {str(e)}']}

    def _test_resource_pressure_recovery(self) -> Dict[str, Any]:
        """Test recovery from resource pressure."""
        try:
            # Simulate memory pressure by creating large objects
            large_objects = []
            try:
                # Allocate memory to create pressure (careful not to crash system)
                for _ in range(10):
                    large_objects.append(bytearray(10 * 1024 * 1024))  # 10MB objects

                # Try to process image under memory pressure
                if self.pipeline and self.test_images:
                    result = self.pipeline.process(self.test_images[0])
                    recovered = result is not None
                else:
                    recovered = False

            finally:
                # Clean up memory
                del large_objects
                gc.collect()

            return {'recovered': recovered, 'errors': [] if recovered else ['Failed under memory pressure']}

        except Exception as e:
            return {'recovered': False, 'errors': [f'Resource pressure test failed: {str(e)}']}

    def _test_concurrent_error_recovery(self) -> Dict[str, Any]:
        """Test concurrent error handling."""
        try:
            if not self.pipeline:
                return {'recovered': False, 'errors': ['Pipeline not available']}

            successful_recoveries = 0
            total_tests = 5

            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit mix of valid and invalid requests
                futures = []

                # Invalid requests
                futures.append(executor.submit(self._process_request_with_timing, "/invalid1.png", "concurrent_error_1"))
                futures.append(executor.submit(self._process_request_with_timing, "/invalid2.png", "concurrent_error_2"))

                # Valid requests (if available)
                if self.test_images:
                    for i in range(3):
                        futures.append(executor.submit(self._process_request_with_timing, self.test_images[0], f"concurrent_valid_{i}"))

                # Collect results
                for future in as_completed(futures):
                    try:
                        success, duration, error_msg = future.result(timeout=30)
                        # Any completion (even with errors) counts as recovery
                        successful_recoveries += 1
                    except Exception:
                        # Timeout or other issues
                        pass

            recovery_rate = successful_recoveries / total_tests
            return {
                'recovered': recovery_rate >= 0.6,  # 60% of concurrent operations should complete
                'errors': [] if recovery_rate >= 0.6 else [f'Low concurrent recovery rate: {recovery_rate:.1%}']
            }

        except Exception as e:
            return {'recovered': False, 'errors': [f'Concurrent error recovery failed: {str(e)}']}

    def _process_request_with_timing(self, image_path: str, request_id: str) -> Tuple[bool, float, Optional[str]]:
        """Process a request and return success, duration, and error message."""
        start_time = time.time()
        try:
            if not self.pipeline:
                return False, 0.0, "Pipeline not available"

            result = self.pipeline.process(image_path)
            duration = time.time() - start_time

            if result and result.success:
                return True, duration, None
            else:
                error_msg = result.error_message if result else "Unknown processing error"
                return False, duration, error_msg

        except Exception as e:
            duration = time.time() - start_time
            return False, duration, str(e)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _create_failed_result(self, test_name: str, error_message: str) -> StressTestResult:
        """Create a failed test result."""
        return StressTestResult(
            test_name=test_name,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_seconds=0.0,
            requests_sent=0,
            requests_successful=0,
            requests_failed=1,
            success_rate=0.0,
            avg_response_time=0.0,
            max_response_time=0.0,
            min_response_time=0.0,
            memory_start_mb=self._get_memory_usage(),
            memory_end_mb=self._get_memory_usage(),
            memory_peak_mb=self._get_memory_usage(),
            cpu_usage_avg=0.0,
            errors=[error_message],
            meets_reliability_target=False
        )

    def generate_stress_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress test report."""
        logger.info("Generating stress test report...")

        # Overall statistics
        total_tests = len(self.results)
        tests_meeting_target = sum(1 for r in self.results if r.meets_reliability_target)
        overall_success_rate = tests_meeting_target / total_tests if total_tests > 0 else 0

        # Collect all errors
        all_errors = []
        for result in self.results:
            all_errors.extend(result.errors)

        # Performance analysis
        all_response_times = []
        for result in self.results:
            if result.requests_successful > 0:
                all_response_times.append(result.avg_response_time)

        avg_system_response = sum(all_response_times) / len(all_response_times) if all_response_times else 0

        # Memory analysis
        memory_stable = all(
            result.memory_end_mb - result.memory_start_mb < 100  # Less than 100MB growth
            for result in self.results
        )

        report = {
            'summary': {
                'total_stress_tests': total_tests,
                'tests_meeting_reliability_target': tests_meeting_target,
                'overall_success_rate': overall_success_rate,
                'system_stable_under_stress': overall_success_rate >= 0.75,
                'memory_stable': memory_stable,
                'avg_system_response_time': avg_system_response
            },
            'test_results': [asdict(r) for r in self.results],
            'reliability_assessment': {
                'handles_concurrent_load': any(r.test_name == 'high_load' and r.meets_reliability_target for r in self.results),
                'handles_resource_constraints': any(r.test_name == 'resource_limits' and r.meets_reliability_target for r in self.results),
                'stable_over_time': any(r.test_name == 'long_running' and r.meets_reliability_target for r in self.results),
                'recovers_from_failures': any(r.test_name == 'error_recovery' and r.meets_reliability_target for r in self.results)
            },
            'error_analysis': {
                'total_errors': len(all_errors),
                'unique_errors': len(set(all_errors)),
                'common_errors': self._analyze_common_errors(all_errors)
            },
            'recommendations': self._generate_stress_recommendations()
        }

        return report

    def _analyze_common_errors(self, errors: List[str]) -> List[Dict[str, Any]]:
        """Analyze and categorize common errors."""
        error_counts = {}
        for error in errors:
            # Simplify error messages for grouping
            simplified = error.split(':')[0] if ':' in error else error
            error_counts[simplified] = error_counts.get(simplified, 0) + 1

        # Sort by frequency
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {'error_type': error, 'count': count, 'percentage': count / len(errors) * 100}
            for error, count in common_errors[:5]  # Top 5 errors
        ]

    def _generate_stress_recommendations(self) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []

        # Check high load performance
        high_load_results = [r for r in self.results if r.test_name == 'high_load']
        if high_load_results and not high_load_results[0].meets_reliability_target:
            recommendations.append(
                "⚠ High load performance below target. Consider implementing request queuing, "
                "connection pooling, or load balancing."
            )

        # Check memory stability
        memory_issues = [r for r in self.results if r.memory_end_mb - r.memory_start_mb > 100]
        if memory_issues:
            recommendations.append(
                "⚠ Memory growth detected during stress testing. Check for memory leaks "
                "and implement more aggressive garbage collection."
            )

        # Check error recovery
        recovery_results = [r for r in self.results if r.test_name == 'error_recovery']
        if recovery_results and not recovery_results[0].meets_reliability_target:
            recommendations.append(
                "⚠ Error recovery needs improvement. Implement better error handling, "
                "circuit breakers, and graceful degradation."
            )

        # Check long running stability
        long_running_results = [r for r in self.results if r.test_name == 'long_running']
        if long_running_results and not long_running_results[0].meets_reliability_target:
            recommendations.append(
                "⚠ Long running stability issues detected. Monitor for memory leaks "
                "and performance degradation over time."
            )

        # Positive recommendations
        if all(r.meets_reliability_target for r in self.results):
            recommendations.append("✅ All stress tests passed - system demonstrates excellent reliability")

        if not recommendations:
            recommendations.append("✅ System shows good stress test performance")

        return recommendations

    def save_results(self, filename: str = "stress_test_results.json"):
        """Save stress test results to file."""
        report = self.generate_stress_report()

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Stress test results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to {filename}: {e}")

    def print_summary(self):
        """Print human-readable summary of stress test results."""
        report = self.generate_stress_report()

        print("\n" + "="*80)
        print("STRESS TEST RESULTS")
        print("="*80)

        # Summary
        summary = report['summary']
        print(f"\n🔥 STRESS TEST SUMMARY:")
        print(f"   • Total tests: {summary['total_stress_tests']}")
        print(f"   • Tests meeting target: {summary['tests_meeting_reliability_target']}")
        print(f"   • Overall success rate: {summary['overall_success_rate']:.1%}")
        print(f"   • System stable: {'✅' if summary['system_stable_under_stress'] else '❌'}")
        print(f"   • Memory stable: {'✅' if summary['memory_stable'] else '❌'}")

        # Reliability assessment
        reliability = report['reliability_assessment']
        print(f"\n🛡️  RELIABILITY ASSESSMENT:")
        for test_type, passed in reliability.items():
            status = "✅" if passed else "❌"
            readable_name = test_type.replace('_', ' ').title()
            print(f"   • {readable_name}: {status}")

        # Error analysis
        error_analysis = report['error_analysis']
        print(f"\n🚨 ERROR ANALYSIS:")
        print(f"   • Total errors: {error_analysis['total_errors']}")
        print(f"   • Unique error types: {error_analysis['unique_errors']}")

        if error_analysis['common_errors']:
            print(f"   • Most common errors:")
            for error_info in error_analysis['common_errors'][:3]:
                print(f"     - {error_info['error_type']}: {error_info['count']} ({error_info['percentage']:.1f}%)")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")

        print("\n" + "="*80)


def main():
    """Main stress testing execution function."""
    parser = argparse.ArgumentParser(description="Stress Testing Suite")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent requests for high load test")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds for long running test")
    parser.add_argument("--memory-limit", type=int, default=500, help="Memory limit in MB for resource test")
    parser.add_argument("--output", default="stress_test_results.json", help="Output file for results")
    parser.add_argument("--high-load-only", action="store_true", help="Run only high load test")
    parser.add_argument("--long-running-only", action="store_true", help="Run only long running test")
    parser.add_argument("--recovery-only", action="store_true", help="Run only error recovery test")

    args = parser.parse_args()

    try:
        tester = StressTester()

        if args.high_load_only:
            tester.test_high_load(concurrent_requests=args.concurrent, total_requests=args.concurrent * 10)
        elif args.long_running_only:
            tester.test_long_running(duration_minutes=args.duration // 60)
        elif args.recovery_only:
            tester.test_error_recovery()
        else:
            # Run all stress tests
            tester.test_high_load(concurrent_requests=args.concurrent, total_requests=args.concurrent * 5)
            tester.test_resource_limits(memory_limit_mb=args.memory_limit)
            tester.test_error_recovery()

            # Optional long running test (shorter for full suite)
            if args.duration > 300:  # Only if duration > 5 minutes
                tester.test_long_running(duration_minutes=min(args.duration // 60, 10))

        # Generate and save results
        tester.save_results(args.output)
        tester.print_summary()

        # Exit with appropriate code
        report = tester.generate_stress_report()
        if report['summary']['system_stable_under_stress']:
            logger.info("🎉 System passes stress testing!")
            return 0
        else:
            logger.warning("⚠️ System shows stress testing issues")
            return 1

    except Exception as e:
        logger.error(f"Stress testing failed: {e}")
        return 2


if __name__ == "__main__":
    exit(main())