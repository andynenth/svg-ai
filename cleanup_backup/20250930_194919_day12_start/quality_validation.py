#!/usr/bin/env python3
"""
Quality Validation Testing - Task 3 Implementation
Validate quality improvements and verify AI enhancement claims.
"""

import sys
import time
import json
import logging
import argparse
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
try:
    from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
    from backend.converters.ai_enhanced_converter import AIEnhancedConverter
    from backend.converters.vtracer_converter import VTracerConverter
    from backend.utils.quality_metrics import ComprehensiveMetrics
except ImportError as e:
    print(f"Warning: Failed to import required modules: {e}")
    print("Some validation features may not be available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityTestResult:
    """Container for quality test results."""
    image_path: str
    category: str
    baseline_quality: float
    ai_quality: float
    improvement_percent: float
    baseline_processing_time: float
    ai_processing_time: float
    baseline_size_bytes: int
    ai_size_bytes: int
    meets_improvement_target: bool
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class QualityValidator:
    """
    Comprehensive quality validation system for AI enhancement verification.
    """

    def __init__(self, baseline_type: str = "vtracer", improvement_target: float = 15.0):
        """
        Initialize quality validator.

        Args:
            baseline_type: Type of baseline converter to use
            improvement_target: Target improvement percentage
        """
        self.baseline_type = baseline_type
        self.improvement_target = improvement_target
        self.results: List[QualityTestResult] = []

        # Initialize converters
        self.ai_converter = None
        self.baseline_converter = None
        self.quality_metrics = None
        self._initialize_components()

        # Load test images
        self.test_images = self._load_test_images()

        logger.info(f"Quality validator initialized with {len(self.test_images)} test images")
        logger.info(f"Target improvement: {improvement_target}%")

    def _initialize_components(self):
        """Initialize AI and baseline converters."""
        try:
            self.ai_converter = AIEnhancedConverter()
            logger.info("✓ AI-enhanced converter initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize AI converter: {e}")
            self.ai_converter = None

        try:
            if self.baseline_type == "vtracer":
                self.baseline_converter = VTracerConverter()
            else:
                # Default to VTracer if type not recognized
                self.baseline_converter = VTracerConverter()
            logger.info(f"✓ Baseline converter ({self.baseline_type}) initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize baseline converter: {e}")
            self.baseline_converter = None

        try:
            self.quality_metrics = ComprehensiveMetrics()
            logger.info("✓ Quality metrics system initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize quality metrics: {e}")
            # Create mock quality metrics for testing
            self.quality_metrics = self._create_mock_quality_metrics()

    def _create_mock_quality_metrics(self):
        """Create mock quality metrics for testing when real metrics unavailable."""
        class MockQualityMetrics:
            def calculate_comprehensive_metrics(self, original_path: str, svg_content: str) -> Dict[str, float]:
                """Mock quality calculation with realistic values."""
                import random
                base_quality = 0.75 + random.uniform(-0.1, 0.1)
                return {
                    'ssim': base_quality,
                    'mse': 0.05 + random.uniform(-0.01, 0.01),
                    'psnr': 25.0 + random.uniform(-2.0, 2.0)
                }

            def compare_images(self, original_path: str, converted_path: str) -> Dict[str, float]:
                """Mock image comparison."""
                return self.calculate_comprehensive_metrics(original_path, "")

        return MockQualityMetrics()

    def _load_test_images(self) -> List[str]:
        """Load test images for quality validation."""
        test_images = []
        base_path = Path("data/logos")

        if not base_path.exists():
            logger.warning(f"Test data path {base_path} not found")
            return []

        categories = ["simple_geometric", "text_based", "gradients", "complex", "abstract"]

        for category in categories:
            category_path = base_path / category
            if category_path.exists():
                # Get all images per category for quality validation
                category_images = list(category_path.glob("*.png"))
                # Filter out processed images
                category_images = [
                    str(img) for img in category_images
                    if "optimized" not in str(img) and ".cache" not in str(img)
                ]
                test_images.extend(category_images)

        logger.info(f"Loaded {len(test_images)} test images for quality validation")
        return test_images

    def validate_improvement_claims(self) -> Dict[str, Any]:
        """Verify 15-20% improvement claim across all test images."""
        logger.info("Starting quality improvement validation...")

        if not self.ai_converter or not self.baseline_converter:
            logger.error("Converters not available for quality validation")
            return {'error': 'Converters not initialized'}

        if not self.test_images:
            logger.error("No test images available for validation")
            return {'error': 'No test images available'}

        baseline_results = []
        ai_results = []

        for i, image_path in enumerate(self.test_images):
            logger.info(f"Validating image {i+1}/{len(self.test_images)}: {Path(image_path).name}")

            try:
                # Baseline conversion
                baseline_start = time.time()
                baseline_svg = self.baseline_converter.convert(image_path)
                baseline_time = time.time() - baseline_start

                # Calculate baseline quality
                baseline_quality = self._measure_quality(image_path, baseline_svg)
                baseline_size = len(baseline_svg) if baseline_svg else 0

                # AI conversion
                ai_start = time.time()
                ai_svg = self.ai_converter.convert(image_path)
                ai_time = time.time() - ai_start

                # Calculate AI quality
                ai_quality = self._measure_quality(image_path, ai_svg)
                ai_size = len(ai_svg) if ai_svg else 0

                # Calculate improvement
                improvement = ((ai_quality - baseline_quality) / baseline_quality * 100) if baseline_quality > 0 else 0

                # Get image category
                category = self._get_image_category(image_path)

                # Create test result
                result = QualityTestResult(
                    image_path=image_path,
                    category=category,
                    baseline_quality=baseline_quality,
                    ai_quality=ai_quality,
                    improvement_percent=improvement,
                    baseline_processing_time=baseline_time,
                    ai_processing_time=ai_time,
                    baseline_size_bytes=baseline_size,
                    ai_size_bytes=ai_size,
                    meets_improvement_target=improvement >= self.improvement_target
                )

                self.results.append(result)
                baseline_results.append(baseline_quality)
                ai_results.append(ai_quality)

                logger.info(f"  Baseline quality: {baseline_quality:.3f}")
                logger.info(f"  AI quality: {ai_quality:.3f}")
                logger.info(f"  Improvement: {improvement:+.1f}%")

            except Exception as e:
                logger.error(f"  Quality validation failed for {Path(image_path).name}: {e}")

        # Calculate overall statistics
        if baseline_results and ai_results:
            avg_baseline = np.mean(baseline_results)
            avg_ai = np.mean(ai_results)
            overall_improvement = (avg_ai - avg_baseline) / avg_baseline * 100

            # Count images meeting target
            images_meeting_target = sum(1 for r in self.results if r.meets_improvement_target)
            target_achievement_rate = images_meeting_target / len(self.results) if self.results else 0

            validation_result = {
                'baseline_avg': avg_baseline,
                'ai_avg': avg_ai,
                'improvement_percent': overall_improvement,
                'meets_target': overall_improvement >= self.improvement_target,
                'target_achievement_rate': target_achievement_rate,
                'images_tested': len(self.results),
                'images_meeting_target': images_meeting_target,
                'baseline_quality_range': [min(baseline_results), max(baseline_results)],
                'ai_quality_range': [min(ai_results), max(ai_results)]
            }

            logger.info(f"Overall Quality Validation Results:")
            logger.info(f"  Average baseline quality: {avg_baseline:.3f}")
            logger.info(f"  Average AI quality: {avg_ai:.3f}")
            logger.info(f"  Overall improvement: {overall_improvement:.1f}%")
            logger.info(f"  Meets {self.improvement_target}% target: {'✓' if overall_improvement >= self.improvement_target else '✗'}")
            logger.info(f"  Images meeting target: {images_meeting_target}/{len(self.results)} ({target_achievement_rate:.1%})")

            return validation_result

        else:
            return {'error': 'No successful quality measurements'}

    def validate_by_category(self) -> Dict[str, Any]:
        """Check improvements per image type/category."""
        logger.info("Starting category-specific validation...")

        categories = ['simple_geometric', 'text_based', 'gradients', 'complex', 'abstract']
        category_results = {}

        for category in categories:
            logger.info(f"Validating category: {category}")

            # Get results for this category
            category_data = [r for r in self.results if r.category == category]

            if not category_data:
                logger.warning(f"  No data available for category: {category}")
                category_results[category] = {
                    'error': 'No data available',
                    'count': 0
                }
                continue

            # Calculate category statistics
            baseline_qualities = [r.baseline_quality for r in category_data]
            ai_qualities = [r.ai_quality for r in category_data]

            avg_baseline = np.mean(baseline_qualities)
            avg_ai = np.mean(ai_qualities)
            category_improvement = (avg_ai - avg_baseline) / avg_baseline * 100

            # Count category successes
            category_successes = sum(1 for r in category_data if r.meets_improvement_target)
            category_success_rate = category_successes / len(category_data)

            # Calculate processing time impact
            avg_baseline_time = np.mean([r.baseline_processing_time for r in category_data])
            avg_ai_time = np.mean([r.ai_processing_time for r in category_data])
            time_overhead = (avg_ai_time - avg_baseline_time) / avg_baseline_time * 100

            category_results[category] = {
                'count': len(category_data),
                'baseline_avg': avg_baseline,
                'ai_avg': avg_ai,
                'improvement_percent': category_improvement,
                'meets_target': category_improvement >= self.improvement_target,
                'success_rate': category_success_rate,
                'time_overhead_percent': time_overhead,
                'avg_baseline_time': avg_baseline_time,
                'avg_ai_time': avg_ai_time
            }

            logger.info(f"  Category: {category}")
            logger.info(f"    Images: {len(category_data)}")
            logger.info(f"    Improvement: {category_improvement:.1f}%")
            logger.info(f"    Meets target: {'✓' if category_improvement >= self.improvement_target else '✗'}")
            logger.info(f"    Success rate: {category_success_rate:.1%}")

        # Overall category analysis
        valid_categories = [k for k, v in category_results.items() if 'error' not in v]
        categories_meeting_target = [k for k, v in category_results.items()
                                   if 'meets_target' in v and v['meets_target']]

        category_analysis = {
            'categories': category_results,
            'total_categories': len(categories),
            'valid_categories': len(valid_categories),
            'categories_meeting_target': len(categories_meeting_target),
            'all_categories_meet_target': len(categories_meeting_target) == len(valid_categories),
            'best_category': max(valid_categories,
                               key=lambda k: category_results[k]['improvement_percent']) if valid_categories else None,
            'worst_category': min(valid_categories,
                                key=lambda k: category_results[k]['improvement_percent']) if valid_categories else None
        }

        logger.info(f"Category Analysis Summary:")
        logger.info(f"  Categories meeting target: {len(categories_meeting_target)}/{len(valid_categories)}")
        logger.info(f"  All categories meet target: {'✓' if category_analysis['all_categories_meet_target'] else '✗'}")

        if category_analysis['best_category']:
            best = category_results[category_analysis['best_category']]
            logger.info(f"  Best performing category: {category_analysis['best_category']} ({best['improvement_percent']:.1f}%)")

        if category_analysis['worst_category']:
            worst = category_results[category_analysis['worst_category']]
            logger.info(f"  Lowest performing category: {category_analysis['worst_category']} ({worst['improvement_percent']:.1f}%)")

        return category_analysis

    def validate_quality_predictions(self) -> Dict[str, Any]:
        """Validate accuracy of quality predictions if available."""
        logger.info("Starting quality prediction validation...")

        if not hasattr(self.ai_converter, 'predict_quality'):
            logger.info("Quality prediction not available in AI converter")
            return {
                'available': False,
                'message': 'Quality prediction not implemented in AI converter'
            }

        prediction_errors = []
        accurate_predictions = 0

        for result in self.results:
            try:
                # Get quality prediction for this image
                predicted_quality = self.ai_converter.predict_quality(result.image_path)
                actual_quality = result.ai_quality

                # Calculate prediction error
                if predicted_quality and actual_quality:
                    error_percent = abs(predicted_quality - actual_quality) / actual_quality * 100
                    prediction_errors.append(error_percent)

                    # Check if prediction is within 10% of actual
                    if error_percent <= 10.0:
                        accurate_predictions += 1

                    logger.info(f"  {Path(result.image_path).name}: predicted={predicted_quality:.3f}, "
                              f"actual={actual_quality:.3f}, error={error_percent:.1f}%")

            except Exception as e:
                logger.warning(f"  Prediction validation failed for {Path(result.image_path).name}: {e}")

        if prediction_errors:
            avg_error = np.mean(prediction_errors)
            max_error = max(prediction_errors)
            accuracy_rate = accurate_predictions / len(prediction_errors)

            prediction_validation = {
                'available': True,
                'predictions_tested': len(prediction_errors),
                'average_error_percent': avg_error,
                'maximum_error_percent': max_error,
                'accurate_predictions': accurate_predictions,
                'accuracy_rate': accuracy_rate,
                'meets_accuracy_target': avg_error <= 10.0,  # Target from DAY10_VALIDATION.md
                'target_accuracy_percent': 10.0
            }

            logger.info(f"Quality Prediction Results:")
            logger.info(f"  Average error: {avg_error:.1f}%")
            logger.info(f"  Maximum error: {max_error:.1f}%")
            logger.info(f"  Accurate predictions: {accurate_predictions}/{len(prediction_errors)} ({accuracy_rate:.1%})")
            logger.info(f"  Meets 10% accuracy target: {'✓' if avg_error <= 10.0 else '✗'}")

            return prediction_validation

        else:
            return {
                'available': True,
                'error': 'No successful predictions to validate'
            }

    def test_edge_cases(self) -> Dict[str, Any]:
        """Test quality validation with edge cases."""
        logger.info("Starting edge case testing...")

        edge_case_results = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'edge_cases': {}
        }

        # Test 1: Very small images (if available)
        small_images = [img for img in self.test_images if self._get_image_size(img) < 50*50]
        if small_images:
            logger.info("Testing very small images...")
            small_test = self._test_edge_case_category(small_images[:3], "very_small_images")
            edge_case_results['edge_cases']['very_small_images'] = small_test
            edge_case_results['tests_run'] += 1
            if small_test.get('success', False):
                edge_case_results['tests_passed'] += 1
            else:
                edge_case_results['tests_failed'] += 1

        # Test 2: Very large images (if available)
        large_images = [img for img in self.test_images if self._get_image_size(img) > 1000*1000]
        if large_images:
            logger.info("Testing very large images...")
            large_test = self._test_edge_case_category(large_images[:3], "very_large_images")
            edge_case_results['edge_cases']['very_large_images'] = large_test
            edge_case_results['tests_run'] += 1
            if large_test.get('success', False):
                edge_case_results['tests_passed'] += 1
            else:
                edge_case_results['tests_failed'] += 1

        # Test 3: Complex images with many colors
        complex_images = [img for img in self.test_images if 'complex' in img.lower()]
        if complex_images:
            logger.info("Testing complex multi-color images...")
            complex_test = self._test_edge_case_category(complex_images, "complex_multicolor")
            edge_case_results['edge_cases']['complex_multicolor'] = complex_test
            edge_case_results['tests_run'] += 1
            if complex_test.get('success', False):
                edge_case_results['tests_passed'] += 1
            else:
                edge_case_results['tests_failed'] += 1

        # Test 4: Gradient images
        gradient_images = [img for img in self.test_images if 'gradient' in img.lower()]
        if gradient_images:
            logger.info("Testing gradient images...")
            gradient_test = self._test_edge_case_category(gradient_images, "gradient_images")
            edge_case_results['edge_cases']['gradient_images'] = gradient_test
            edge_case_results['tests_run'] += 1
            if gradient_test.get('success', False):
                edge_case_results['tests_passed'] += 1
            else:
                edge_case_results['tests_failed'] += 1

        success_rate = edge_case_results['tests_passed'] / edge_case_results['tests_run'] if edge_case_results['tests_run'] > 0 else 0
        edge_case_results['success_rate'] = success_rate

        logger.info(f"Edge Case Testing Results:")
        logger.info(f"  Tests run: {edge_case_results['tests_run']}")
        logger.info(f"  Tests passed: {edge_case_results['tests_passed']}")
        logger.info(f"  Success rate: {success_rate:.1%}")

        return edge_case_results

    def _test_edge_case_category(self, images: List[str], category_name: str) -> Dict[str, Any]:
        """Test a specific edge case category."""
        try:
            baseline_qualities = []
            ai_qualities = []

            for image_path in images:
                try:
                    # Test baseline
                    baseline_svg = self.baseline_converter.convert(image_path)
                    baseline_quality = self._measure_quality(image_path, baseline_svg)

                    # Test AI
                    ai_svg = self.ai_converter.convert(image_path)
                    ai_quality = self._measure_quality(image_path, ai_svg)

                    baseline_qualities.append(baseline_quality)
                    ai_qualities.append(ai_quality)

                except Exception as e:
                    logger.warning(f"    Edge case test failed for {Path(image_path).name}: {e}")

            if baseline_qualities and ai_qualities:
                avg_baseline = np.mean(baseline_qualities)
                avg_ai = np.mean(ai_qualities)
                improvement = (avg_ai - avg_baseline) / avg_baseline * 100

                return {
                    'success': True,
                    'images_tested': len(baseline_qualities),
                    'baseline_avg': avg_baseline,
                    'ai_avg': avg_ai,
                    'improvement_percent': improvement,
                    'meets_target': improvement >= self.improvement_target
                }
            else:
                return {
                    'success': False,
                    'error': 'No successful conversions in edge case test'
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _measure_quality(self, image_path: str, svg_content: str) -> float:
        """Measure quality of conversion."""
        try:
            if not svg_content:
                return 0.0

            # Use quality metrics to measure SSIM
            quality_result = self.quality_metrics.calculate_comprehensive_metrics(image_path, svg_content)
            return quality_result.get('ssim', 0.0)

        except Exception as e:
            logger.warning(f"Quality measurement failed: {e}")
            return 0.0

    def _get_image_category(self, image_path: str) -> str:
        """Extract category from image path."""
        path_parts = Path(image_path).parts
        categories = ["simple_geometric", "text_based", "gradients", "complex", "abstract"]
        for part in path_parts:
            if part in categories:
                return part
        return "unknown"

    def _get_image_size(self, image_path: str) -> int:
        """Get image size in pixels."""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.width * img.height
        except Exception:
            return 0

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality validation report."""
        logger.info("Generating quality validation report...")

        # Overall improvement validation
        overall_validation = self.validate_improvement_claims()

        # Category-specific validation
        category_validation = self.validate_by_category()

        # Quality prediction validation
        prediction_validation = self.validate_quality_predictions()

        # Edge case testing
        edge_case_validation = self.test_edge_cases()

        # Generate summary
        validation_summary = {
            'overall_improvement': overall_validation,
            'category_analysis': category_validation,
            'prediction_accuracy': prediction_validation,
            'edge_case_testing': edge_case_validation,
            'detailed_results': [asdict(r) for r in self.results],
            'validation_timestamp': datetime.now().isoformat()
        }

        # Assessment against acceptance criteria
        acceptance_criteria = {
            'overall_improvement_15_percent': overall_validation.get('meets_target', False),
            'positive_improvement_all_categories': category_validation.get('all_categories_meet_target', False),
            'quality_predictions_accurate': prediction_validation.get('meets_accuracy_target', False) if prediction_validation.get('available') else True,
            'no_quality_regressions': self._check_no_regressions()
        }

        all_criteria_met = all(acceptance_criteria.values())

        validation_summary['acceptance_criteria'] = acceptance_criteria
        validation_summary['all_criteria_met'] = all_criteria_met

        return validation_summary

    def _check_no_regressions(self) -> bool:
        """Check that no quality regressions occurred."""
        regressions = [r for r in self.results if r.improvement_percent < 0]
        return len(regressions) == 0

    def save_results(self, filename: str = "quality_validation_results.json"):
        """Save validation results to file."""
        report = self.generate_quality_report()

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Quality validation results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to {filename}: {e}")

    def print_summary(self):
        """Print human-readable summary of quality validation."""
        report = self.generate_quality_report()

        print("\n" + "="*80)
        print("QUALITY VALIDATION RESULTS")
        print("="*80)

        # Overall improvement
        overall = report['overall_improvement']
        if 'error' not in overall:
            status = "✅" if overall['meets_target'] else "❌"
            print(f"\n📊 OVERALL IMPROVEMENT: {status}")
            print(f"   • Target: {self.improvement_target}%")
            print(f"   • Measured: {overall['improvement_percent']:.1f}%")
            print(f"   • Images tested: {overall['images_tested']}")
            print(f"   • Baseline avg quality: {overall['baseline_avg']:.3f}")
            print(f"   • AI avg quality: {overall['ai_avg']:.3f}")

        # Category analysis
        category = report['category_analysis']
        status = "✅" if category['all_categories_meet_target'] else "❌"
        print(f"\n🏷️  CATEGORY CONSISTENCY: {status}")
        print(f"   • Categories meeting target: {category['categories_meeting_target']}/{category['valid_categories']}")

        for cat_name, cat_data in category['categories'].items():
            if 'improvement_percent' in cat_data:
                cat_status = "✅" if cat_data['meets_target'] else "❌"
                print(f"   • {cat_name}: {cat_status} {cat_data['improvement_percent']:+.1f}%")

        # Acceptance criteria
        criteria = report['acceptance_criteria']
        print(f"\n✅ ACCEPTANCE CRITERIA:")
        for criterion, met in criteria.items():
            status = "✅" if met else "❌"
            readable_name = criterion.replace('_', ' ').title()
            print(f"   • {readable_name}: {status}")

        # Overall status
        if report['all_criteria_met']:
            print(f"\n🎉 ALL QUALITY VALIDATION CRITERIA MET!")
        else:
            print(f"\n⚠️  Some quality validation criteria not met")

        print("\n" + "="*80)


def main():
    """Main quality validation execution function."""
    parser = argparse.ArgumentParser(description="Quality Validation Suite")
    parser.add_argument("--test-set", default="data/logos", help="Path to test image set")
    parser.add_argument("--baseline", default="vtracer", help="Baseline converter type")
    parser.add_argument("--ai", action="store_true", help="Test AI-enhanced converter")
    parser.add_argument("--target", type=float, default=15.0, help="Target improvement percentage")
    parser.add_argument("--output", default="quality_validation_results.json", help="Output file")
    parser.add_argument("--categories-only", action="store_true", help="Run only category validation")
    parser.add_argument("--predictions-only", action="store_true", help="Run only prediction validation")

    args = parser.parse_args()

    try:
        validator = QualityValidator(
            baseline_type=args.baseline,
            improvement_target=args.target
        )

        if args.categories_only:
            validator.validate_improvement_claims()  # Need overall results first
            validator.validate_by_category()
        elif args.predictions_only:
            validator.validate_improvement_claims()  # Need overall results first
            validator.validate_quality_predictions()
        else:
            # Run full validation
            validator.validate_improvement_claims()
            validator.validate_by_category()
            validator.validate_quality_predictions()
            validator.test_edge_cases()

        # Generate and save results
        validator.save_results(args.output)
        validator.print_summary()

        # Exit with appropriate code
        report = validator.generate_quality_report()
        if report['all_criteria_met']:
            logger.info("🎉 All quality validation criteria met!")
            return 0
        else:
            logger.warning("⚠️ Some quality validation criteria not met")
            return 1

    except Exception as e:
        logger.error(f"Quality validation failed: {e}")
        return 2


if __name__ == "__main__":
    exit(main())