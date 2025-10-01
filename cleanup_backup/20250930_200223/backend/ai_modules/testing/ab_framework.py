"""
A/B Test Framework Core - Task 1 Implementation
Comprehensive A/B testing framework to validate AI enhancements against baseline system.
"""

import time
import json
import logging
import hashlib
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

# Import quality measurement from existing system
try:
    from backend.converters.base import BaseConverter
    from backend.quality.quality_analyzer import QualityAnalyzer
    from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
except ImportError:
    # Fallback imports for testing
    BaseConverter = None
    QualityAnalyzer = None
    UnifiedAIPipeline = None

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for A/B test result."""
    image: str
    group: str
    quality: Dict[str, float]
    duration: float
    parameters: Dict[str, Any]
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    svg_path: Optional[str] = None
    file_size: Optional[int] = None


@dataclass
class TestConfig:
    """Configuration for A/B test."""
    name: str
    assignment_method: str = 'random'  # 'random', 'hash', 'sequential'
    split_ratio: float = 0.5  # Percentage to treatment group
    sample_size_target: Optional[int] = None
    min_sample_size: int = 30
    significance_threshold: float = 0.05
    quality_metrics: List[str] = None
    output_directory: str = 'ab_test_results'
    continuous_mode: bool = False
    early_stopping: bool = True

    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = ['ssim', 'mse', 'psnr']


class ABTestFramework:
    """
    Core A/B testing framework for comparing baseline and AI-enhanced converters.
    """

    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize A/B test framework.

        Args:
            config: Test configuration (uses default if None)
        """
        self.config = config or TestConfig(name='default_test')

        # Initialize converters
        self.test_groups = {
            'control': self._baseline_converter,
            'treatment': self._ai_enhanced_converter
        }

        # Results storage
        self.results = []
        self.assignment_cache = {}

        # Assignment tracking
        self.assignment_counts = {'control': 0, 'treatment': 0}
        self.sequential_counter = 0

        # Thread safety
        self._lock = threading.RLock()

        # Initialize quality analyzer
        self.quality_analyzer = self._initialize_quality_analyzer()

        # Initialize converters
        self.baseline_converter = self._initialize_baseline_converter()
        self.ai_converter = self._initialize_ai_converter()

        logger.info(f"ABTestFramework initialized with config: {self.config.name}")

    def _initialize_quality_analyzer(self):
        """Initialize quality measurement system."""
        try:
            if QualityAnalyzer:
                return QualityAnalyzer()
            else:
                # Fallback quality analyzer
                return MockQualityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize quality analyzer: {e}")
            return MockQualityAnalyzer()

    def _initialize_baseline_converter(self):
        """Initialize baseline converter."""
        try:
            if BaseConverter:
                # Use simple VTracer converter as baseline
                from backend.converters.vtracer_converter import VTracerConverter
                return VTracerConverter()
            else:
                return MockConverter('baseline')
        except Exception as e:
            logger.warning(f"Failed to initialize baseline converter: {e}")
            return MockConverter('baseline')

    def _initialize_ai_converter(self):
        """Initialize AI-enhanced converter."""
        try:
            if UnifiedAIPipeline:
                return UnifiedAIPipeline()
            else:
                return MockConverter('ai_enhanced')
        except Exception as e:
            logger.warning(f"Failed to initialize AI converter: {e}")
            return MockConverter('ai_enhanced')

    def run_test(self, image_path: str, test_config: Optional[Dict] = None) -> TestResult:
        """
        Run A/B test on a single image.

        Args:
            image_path: Path to image file
            test_config: Optional test-specific configuration

        Returns:
            TestResult object with all metrics
        """
        with self._lock:
            try:
                # Merge test config
                merged_config = {**asdict(self.config)}
                if test_config:
                    merged_config.update(test_config)

                # Assign to group
                group = self.assign_group(image_path, merged_config)

                # Run conversion
                start_time = time.time()
                conversion_result = self.test_groups[group](image_path)
                duration = time.time() - start_time

                # Measure quality
                quality = self.measure_quality(image_path, conversion_result)

                # Create test result
                test_result = TestResult(
                    image=image_path,
                    group=group,
                    quality=quality,
                    duration=duration,
                    parameters=conversion_result.get('parameters', {}),
                    timestamp=datetime.now(),
                    success=conversion_result.get('success', True),
                    error=conversion_result.get('error'),
                    svg_path=conversion_result.get('svg_path'),
                    file_size=conversion_result.get('file_size')
                )

                # Record results
                self.results.append(test_result)
                self.assignment_counts[group] += 1

                logger.debug(f"A/B test completed: {image_path} -> {group} "
                           f"(quality: {quality.get('ssim', 0):.3f}, duration: {duration:.2f}s)")

                return test_result

            except Exception as e:
                logger.error(f"A/B test failed for {image_path}: {e}")
                error_result = TestResult(
                    image=image_path,
                    group='error',
                    quality={},
                    duration=0,
                    parameters={},
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e)
                )
                self.results.append(error_result)
                return error_result

    def assign_group(self, image_path: str, test_config: Dict) -> str:
        """
        Assign image to control or treatment group.

        Args:
            image_path: Path to image
            test_config: Test configuration

        Returns:
            Group assignment ('control' or 'treatment')
        """
        assignment_method = test_config.get('assignment_method', 'random')
        split_ratio = test_config.get('split_ratio', 0.5)

        # Check cache for consistency
        cache_key = f"{image_path}_{assignment_method}_{split_ratio}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]

        # Perform assignment
        if assignment_method == 'random':
            group = 'treatment' if random.random() < split_ratio else 'control'

        elif assignment_method == 'hash':
            # Deterministic hash-based assignment
            hash_value = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
            group = 'treatment' if (hash_value % 100) < (split_ratio * 100) else 'control'

        elif assignment_method == 'sequential':
            # Alternating assignment
            group = 'treatment' if self.sequential_counter % 2 == 0 else 'control'
            self.sequential_counter += 1

        else:
            raise ValueError(f"Unknown assignment method: {assignment_method}")

        # Cache assignment
        self.assignment_cache[cache_key] = group
        return group

    def _baseline_converter(self, image_path: str) -> Dict[str, Any]:
        """
        Run baseline converter.

        Args:
            image_path: Path to image

        Returns:
            Conversion result dictionary
        """
        try:
            result = self.baseline_converter.convert(image_path)
            return {
                'svg_path': result.get('output_path'),
                'parameters': result.get('parameters', {}),
                'success': True,
                'method': 'baseline',
                'file_size': self._get_file_size(result.get('output_path'))
            }
        except Exception as e:
            logger.error(f"Baseline conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'baseline'
            }

    def _ai_enhanced_converter(self, image_path: str) -> Dict[str, Any]:
        """
        Run AI-enhanced converter.

        Args:
            image_path: Path to image

        Returns:
            Conversion result dictionary
        """
        try:
            result = self.ai_converter.process(image_path)
            return {
                'svg_path': result.get('output_path'),
                'parameters': result.get('parameters', {}),
                'success': True,
                'method': 'ai_enhanced',
                'file_size': self._get_file_size(result.get('output_path'))
            }
        except Exception as e:
            logger.error(f"AI conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'ai_enhanced'
            }

    def measure_quality(self, original_path: str, conversion_result: Dict) -> Dict[str, float]:
        """
        Measure quality of conversion result.

        Args:
            original_path: Path to original image
            conversion_result: Conversion result

        Returns:
            Quality metrics dictionary
        """
        try:
            if not conversion_result.get('success', False):
                return {'error': True}

            svg_path = conversion_result.get('svg_path')
            if not svg_path or not Path(svg_path).exists():
                return {'error': True, 'reason': 'missing_svg'}

            # Use quality analyzer
            quality_result = self.quality_analyzer.analyze(original_path, svg_path)

            return {
                'ssim': quality_result.get('ssim', 0.0),
                'mse': quality_result.get('mse', float('inf')),
                'psnr': quality_result.get('psnr', 0.0),
                'error': False
            }

        except Exception as e:
            logger.error(f"Quality measurement failed: {e}")
            return {'error': True, 'reason': str(e)}

    def _get_file_size(self, file_path: Optional[str]) -> Optional[int]:
        """Get file size in bytes."""
        if file_path and Path(file_path).exists():
            return Path(file_path).stat().st_size
        return None

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of test results.

        Returns:
            Summary statistics
        """
        with self._lock:
            if not self.results:
                return {'error': 'No results available'}

            # Separate groups
            control_results = [r for r in self.results if r.group == 'control' and r.success]
            treatment_results = [r for r in self.results if r.group == 'treatment' and r.success]

            summary = {
                'total_tests': len(self.results),
                'control_count': len(control_results),
                'treatment_count': len(treatment_results),
                'success_rate': {
                    'control': len(control_results) / max(1, self.assignment_counts['control']),
                    'treatment': len(treatment_results) / max(1, self.assignment_counts['treatment'])
                }
            }

            # Calculate averages for each group
            if control_results:
                summary['control_avg'] = {
                    'ssim': sum(r.quality.get('ssim', 0) for r in control_results) / len(control_results),
                    'duration': sum(r.duration for r in control_results) / len(control_results)
                }

            if treatment_results:
                summary['treatment_avg'] = {
                    'ssim': sum(r.quality.get('ssim', 0) for r in treatment_results) / len(treatment_results),
                    'duration': sum(r.duration for r in treatment_results) / len(treatment_results)
                }

            # Calculate improvement
            if control_results and treatment_results:
                control_ssim = summary['control_avg']['ssim']
                treatment_ssim = summary['treatment_avg']['ssim']
                if control_ssim > 0:
                    summary['improvement'] = {
                        'ssim_pct': ((treatment_ssim - control_ssim) / control_ssim) * 100,
                        'ssim_absolute': treatment_ssim - control_ssim
                    }

            return summary

    def save_results(self, filepath: str):
        """
        Save test results to file.

        Args:
            filepath: Output file path
        """
        with self._lock:
            # Convert results to serializable format
            serializable_results = []
            for result in self.results:
                result_dict = asdict(result)
                result_dict['timestamp'] = result.timestamp.isoformat()
                serializable_results.append(result_dict)

            output_data = {
                'config': asdict(self.config),
                'results': serializable_results,
                'summary': self.get_results_summary(),
                'generated_at': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"A/B test results saved to {filepath}")

    def load_results(self, filepath: str):
        """
        Load test results from file.

        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore results
        self.results = []
        for result_dict in data['results']:
            result_dict['timestamp'] = datetime.fromisoformat(result_dict['timestamp'])
            result = TestResult(**result_dict)
            self.results.append(result)

        # Update assignment counts
        self.assignment_counts = {'control': 0, 'treatment': 0}
        for result in self.results:
            if result.group in self.assignment_counts:
                self.assignment_counts[result.group] += 1

        logger.info(f"Loaded {len(self.results)} A/B test results from {filepath}")

    def clear_results(self):
        """Clear all test results."""
        with self._lock:
            self.results.clear()
            self.assignment_cache.clear()
            self.assignment_counts = {'control': 0, 'treatment': 0}
            self.sequential_counter = 0


class MockQualityAnalyzer:
    """Mock quality analyzer for testing when real one isn't available."""

    def analyze(self, original_path: str, svg_path: str) -> Dict[str, float]:
        """Mock quality analysis."""
        # Return realistic but fake metrics
        return {
            'ssim': random.uniform(0.75, 0.95),
            'mse': random.uniform(0.001, 0.1),
            'psnr': random.uniform(25, 40)
        }


class MockConverter:
    """Mock converter for testing when real converters aren't available."""

    def __init__(self, converter_type: str):
        self.converter_type = converter_type

    def convert(self, image_path: str) -> Dict[str, Any]:
        """Mock conversion."""
        return self._mock_convert(image_path)

    def process(self, image_path: str) -> Dict[str, Any]:
        """Mock processing (AI converter interface)."""
        return self._mock_convert(image_path)

    def _mock_convert(self, image_path: str) -> Dict[str, Any]:
        """Simulate conversion with slight quality differences."""
        # Simulate AI being slightly better
        if self.converter_type == 'ai_enhanced':
            quality_boost = random.uniform(0.02, 0.08)
        else:
            quality_boost = 0

        return {
            'output_path': f"/tmp/mock_{self.converter_type}.svg",
            'parameters': {'mock': True, 'type': self.converter_type},
            'success': True,
            'quality_boost': quality_boost
        }


def test_ab_framework():
    """Test the A/B framework."""
    print("Testing A/B Test Framework...")

    # Create test configuration
    config = TestConfig(
        name='test_framework',
        assignment_method='hash',
        sample_size_target=10
    )

    # Initialize framework
    framework = ABTestFramework(config)

    # Test 1: Single test
    print("\n✓ Testing single A/B test:")
    result = framework.run_test('test_image.png')
    print(f"  Group: {result.group}")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.3f}s")

    # Test 2: Multiple tests
    print("\n✓ Testing multiple A/B tests:")
    test_images = [f'test_{i}.png' for i in range(10)]
    for image in test_images:
        framework.run_test(image)

    # Test 3: Assignment consistency
    print("\n✓ Testing assignment consistency:")
    group1 = framework.assign_group('consistent.png', {'assignment_method': 'hash'})
    group2 = framework.assign_group('consistent.png', {'assignment_method': 'hash'})
    assert group1 == group2, "Hash assignment should be consistent"
    print(f"  Consistent assignment: {group1}")

    # Test 4: Results summary
    print("\n✓ Testing results summary:")
    summary = framework.get_results_summary()
    print(f"  Total tests: {summary['total_tests']}")
    print(f"  Control count: {summary['control_count']}")
    print(f"  Treatment count: {summary['treatment_count']}")

    if 'improvement' in summary:
        print(f"  SSIM improvement: {summary['improvement']['ssim_pct']:.1f}%")

    # Test 5: Save/load results
    print("\n✓ Testing save/load results:")
    test_file = '/tmp/ab_test_results.json'
    framework.save_results(test_file)

    # Create new framework and load
    framework2 = ABTestFramework()
    framework2.load_results(test_file)
    assert len(framework2.results) == len(framework.results)
    print(f"  Successfully saved and loaded {len(framework2.results)} results")

    print("\n✅ All A/B framework tests passed!")
    return framework


if __name__ == "__main__":
    test_ab_framework()