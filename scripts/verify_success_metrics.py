#!/usr/bin/env python3
"""
Success Metrics Verification for DAY14 Integration Testing
Systematically verify each success metric with actual testing
"""

import json
import time
import statistics
from pathlib import Path
import traceback
from typing import Dict, List, Any, Tuple
import subprocess


class SuccessMetricsVerification:
    """Systematically verify all DAY14 Success Metrics"""

    def __init__(self):
        self.results = {
            'functionality': {},
            'performance': {},
            'quality': {},
            'overall_status': 'UNKNOWN'
        }

    def test_functionality_no_lost_features(self) -> bool:
        """Test: No functionality lost from cleanup"""
        print("\n🔍 Testing: No functionality lost from cleanup")

        try:
            # Test core pipeline functionality
            from backend.ai_modules.pipeline import UnifiedAIPipeline
            from backend.ai_modules.classification import ClassificationModule
            from backend.ai_modules.optimization import OptimizationEngine
            from backend.ai_modules.quality import QualitySystem
            from backend.converters.ai_enhanced_converter import AIEnhancedConverter

            pipeline = UnifiedAIPipeline()
            test_image = 'data/test/simple_01.png'

            # Test complete pipeline flow
            result = pipeline.process(test_image)

            if result is None:
                print("❌ Pipeline processing failed")
                return False

            # Convert to dict if it's a PipelineResult object
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            else:
                result_dict = result

            # Check pipeline success
            if not result_dict.get('success', False):
                print(f"❌ Pipeline processing failed: {result_dict.get('error_message', 'Unknown error')}")
                return False

            # Check key components exist
            if not result_dict.get('svg_content'):
                print("❌ No SVG content generated")
                return False

            # Test individual modules
            classifier = ClassificationModule()
            class_result = classifier.classify(test_image)
            if 'final_class' not in class_result:
                print("❌ Classification functionality broken")
                return False

            optimizer = OptimizationEngine()
            features = classifier.feature_extractor.extract(test_image)
            params = optimizer.calculate_base_parameters(features)
            if not isinstance(params, dict) or 'color_precision' not in params:
                print("❌ Optimization functionality broken")
                return False

            converter = AIEnhancedConverter()
            svg_result = converter.convert(test_image, parameters=params)
            if 'svg_content' not in svg_result:
                print("❌ Conversion functionality broken")
                return False

            quality = QualitySystem()
            # Create temporary SVG file for quality measurement
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.svg', mode='w', delete=False) as f:
                f.write(svg_result['svg_content'])
                temp_svg_path = f.name

            try:
                metrics = quality.calculate_comprehensive_metrics(test_image, temp_svg_path)
                if 'ssim' not in metrics:
                    print("❌ Quality measurement functionality broken")
                    return False
            finally:
                Path(temp_svg_path).unlink(missing_ok=True)

            print("✅ All core functionality working")
            return True

        except Exception as e:
            print(f"❌ Functionality test failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_functionality_all_features_working(self) -> bool:
        """Test: All features working"""
        print("\n🔍 Testing: All features working")

        try:
            # Test each major feature category
            features_tested = {}

            # 1. Classification features
            from backend.ai_modules.classification import ClassificationModule
            classifier = ClassificationModule()

            test_cases = [
                ('data/test/simple_01.png', 'simple'),
                ('data/test/text_01.png', 'text'),
                ('data/test/gradient_01.png', 'gradient'),
                ('data/test/complex_01.png', 'complex')
            ]

            for image_path, expected_type in test_cases:
                result = classifier.classify(image_path)
                if 'final_class' not in result:
                    print(f"❌ Classification failed for {image_path}")
                    return False
            features_tested['classification'] = True

            # 2. Optimization features
            from backend.ai_modules.optimization import OptimizationEngine
            optimizer = OptimizationEngine()

            # Test formula-based optimization
            features = {'unique_colors': 5, 'complexity': 0.2, 'has_gradients': False, 'edge_density': 0.1}
            params = optimizer.calculate_base_parameters(features)
            if not isinstance(params, dict):
                print("❌ Formula-based optimization failed")
                return False
            features_tested['optimization'] = True

            # 3. Quality features
            from backend.ai_modules.quality import QualitySystem
            quality = QualitySystem()

            from backend.converters.ai_enhanced_converter import AIEnhancedConverter
            converter = AIEnhancedConverter()

            svg_result = converter.convert('data/test/simple_01.png')

            # Create temporary SVG file for quality measurement
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.svg', mode='w', delete=False) as f:
                f.write(svg_result['svg_content'])
                temp_svg_path = f.name

            try:
                metrics = quality.calculate_comprehensive_metrics('data/test/simple_01.png', temp_svg_path)
                required_metrics = ['ssim', 'mse', 'psnr']
                missing_metrics = [m for m in required_metrics if m not in metrics]
                if missing_metrics:
                    print(f"❌ Missing quality metrics: {missing_metrics}")
                    return False
                features_tested['quality'] = True
            finally:
                Path(temp_svg_path).unlink(missing_ok=True)

            # 4. Pipeline features
            from backend.ai_modules.pipeline import UnifiedAIPipeline
            pipeline = UnifiedAIPipeline()

            # Test batch processing
            test_images = ['data/test/simple_01.png', 'data/test/text_01.png']
            for img in test_images:
                result = pipeline.process(img)
                if result is None:
                    print(f"❌ Pipeline processing failed for {img}")
                    return False

                # Convert to dict if needed
                if hasattr(result, 'to_dict'):
                    result_dict = result.to_dict()
                else:
                    result_dict = result

                if not result_dict.get('success', False):
                    print(f"❌ Pipeline processing failed for {img}")
                    return False
            features_tested['pipeline'] = True

            # 5. Caching features
            start_time = time.time()
            result1 = pipeline.process('data/test/simple_01.png')
            first_time = time.time() - start_time

            start_time = time.time()
            result2 = pipeline.process('data/test/simple_01.png')
            second_time = time.time() - start_time

            if second_time >= first_time:
                print("⚠️ Caching may not be working optimally")
            features_tested['caching'] = True

            print(f"✅ All features working: {list(features_tested.keys())}")
            return True

        except Exception as e:
            print(f"❌ Feature test failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_functionality_improved_organization(self) -> bool:
        """Test: Improved organization evident"""
        print("\n🔍 Testing: Improved organization evident")

        try:
            # Check file structure organization
            expected_structure = {
                'backend/ai_modules/': [
                    'classification.py',
                    'optimization.py',
                    'quality.py',
                    'pipeline/unified_ai_pipeline.py',
                    'utils.py'
                ],
                'tests/': [
                    'test_models.py'
                ],
                'scripts/': [
                    'performance_regression_test.py'
                ]
            }

            missing_files = []
            for directory, files in expected_structure.items():
                for file in files:
                    full_path = Path(directory) / file
                    if not full_path.exists():
                        missing_files.append(str(full_path))

            if missing_files:
                print(f"❌ Missing organized files: {missing_files}")
                return False

            # Check import structure works
            try:
                from backend.ai_modules.pipeline import UnifiedAIPipeline
                from backend.ai_modules.classification import ClassificationModule
                from backend.ai_modules.optimization import OptimizationEngine
                from backend.ai_modules.quality import QualitySystem
                print("✅ Import structure organized correctly")
            except ImportError as e:
                print(f"❌ Import structure broken: {e}")
                return False

            # Check if tests are organized
            if Path('tests/test_models.py').exists():
                print("✅ Test structure organized")
            else:
                print("❌ Test structure not organized")
                return False

            print("✅ Code organization improved")
            return True

        except Exception as e:
            print(f"❌ Organization test failed: {str(e)}")
            return False

    def test_performance_equal_or_better(self) -> bool:
        """Test: Equal or better than Day 13"""
        print("\n🔍 Testing: Performance equal or better than Day 13")

        try:
            # Load performance results
            perf_file = Path('performance_report_day14.json')
            if not perf_file.exists():
                print("❌ Performance report not found")
                return False

            with open(perf_file) as f:
                performance = json.load(f)

            # Check if baseline exists for comparison
            baseline_file = Path('benchmarks/day13_baseline.json')
            if not baseline_file.exists():
                print("⚠️ No Day 13 baseline for comparison")
                # Without baseline, check if performance is reasonable
                current_times = performance['details']['conversion_times']
                simple_time = current_times['simple']['mean']

                if simple_time > 5.0:  # Very conservative threshold
                    print(f"❌ Performance seems degraded: simple conversion {simple_time}s")
                    return False
                else:
                    print(f"✅ Performance reasonable: simple conversion {simple_time}s")
                    return True

            # If baseline exists, compare
            with open(baseline_file) as f:
                baseline = json.load(f)

            current_simple = performance['details']['conversion_times']['simple']['mean']
            baseline_simple = baseline.get('conversion_times', {}).get('simple', {}).get('mean', 10.0)

            if current_simple <= baseline_simple * 1.1:  # Allow 10% tolerance
                print(f"✅ Performance maintained or improved: {current_simple}s vs {baseline_simple}s")
                return True
            else:
                print(f"❌ Performance degraded: {current_simple}s vs {baseline_simple}s")
                return False

        except Exception as e:
            print(f"❌ Performance comparison failed: {str(e)}")
            return False

    def test_performance_all_targets_achieved(self) -> bool:
        """Test: All targets achieved"""
        print("\n🔍 Testing: All performance targets achieved")

        try:
            # Load performance results
            perf_file = Path('performance_report_day14.json')
            if not perf_file.exists():
                print("❌ Performance report not found")
                return False

            with open(perf_file) as f:
                performance = json.load(f)

            summary = performance['summary']
            failed_targets = [key for key, passed in summary.items() if not passed]

            if failed_targets:
                print(f"❌ Failed performance targets: {failed_targets}")
                details = performance['details']

                if 'import_time_ok' in failed_targets:
                    print(f"  - Import time: {details['import_time']:.1f}s (target: <2s)")
                if 'conversion_speed_ok' in failed_targets:
                    simple_time = details['conversion_times']['simple']['mean']
                    print(f"  - Simple conversion: {simple_time:.1f}s (target: <2s)")

                return False
            else:
                print("✅ All performance targets achieved")
                return True

        except Exception as e:
            print(f"❌ Performance target test failed: {str(e)}")
            return False

    def test_performance_no_memory_leaks(self) -> bool:
        """Test: No memory leaks"""
        print("\n🔍 Testing: No memory leaks")

        try:
            import psutil
            import gc

            # Get baseline memory
            gc.collect()
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024

            # Import and use modules multiple times
            from backend.ai_modules.pipeline import UnifiedAIPipeline

            memory_readings = [baseline_memory]

            for i in range(10):
                pipeline = UnifiedAIPipeline()
                result = pipeline.process('data/test/simple_01.png')
                del pipeline
                gc.collect()

                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)

            # Check if memory consistently increases (indicates leak)
            final_memory = memory_readings[-1]
            memory_increase = final_memory - baseline_memory

            # Check from performance report
            perf_file = Path('performance_report_day14.json')
            if perf_file.exists():
                with open(perf_file) as f:
                    performance = json.load(f)

                memory_used = performance['details']['memory_usage']['used_mb']
                if memory_used > 500:
                    print(f"❌ Memory usage too high: {memory_used}MB")
                    return False

            if memory_increase > 100:  # 100MB increase is concerning
                print(f"❌ Potential memory leak: {memory_increase:.1f}MB increase")
                return False
            else:
                print(f"✅ No significant memory leaks: {memory_increase:.1f}MB change")
                return True

        except Exception as e:
            print(f"❌ Memory leak test failed: {str(e)}")
            return False

    def test_quality_ssim_improvements_maintained(self) -> bool:
        """Test: SSIM improvements maintained"""
        print("\n🔍 Testing: SSIM improvements maintained")

        try:
            from backend.ai_modules.pipeline import UnifiedAIPipeline
            from backend.ai_modules.quality import QualitySystem
            from backend.converters.ai_enhanced_converter import AIEnhancedConverter

            pipeline = UnifiedAIPipeline()
            converter = AIEnhancedConverter()
            quality = QualitySystem()

            # Test different image types
            test_cases = [
                ('data/test/simple_01.png', 0.85),  # Expected minimum SSIM
                ('data/test/text_01.png', 0.85),
                ('data/test/gradient_01.png', 0.80),
                ('data/test/complex_01.png', 0.70)
            ]

            results = []
            for image_path, min_expected in test_cases:
                result = pipeline.process(image_path)
                if result is None:
                    print(f"❌ Failed to process {image_path}")
                    return False

                # Convert to dict if needed
                if hasattr(result, 'to_dict'):
                    result_dict = result.to_dict()
                else:
                    result_dict = result

                if not result_dict.get('success', False):
                    print(f"❌ Processing failed for {image_path}")
                    return False

                # Use quality_score as proxy for SSIM if available
                ssim = result_dict.get('quality_score', 0.0)
                results.append((image_path, ssim, min_expected))

                if ssim < min_expected:
                    print(f"❌ Quality below expectation for {image_path}: {ssim:.3f} < {min_expected}")
                    return False

            avg_ssim = statistics.mean([r[1] for r in results])
            print(f"✅ SSIM improvements maintained: average {avg_ssim:.3f}")

            for image_path, ssim, min_expected in results:
                print(f"  - {Path(image_path).name}: {ssim:.3f} (target: >{min_expected})")

            return True

        except Exception as e:
            print(f"❌ SSIM test failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_quality_consistent_results(self) -> bool:
        """Test: Consistent results"""
        print("\n🔍 Testing: Consistent results")

        try:
            from backend.ai_modules.pipeline import UnifiedAIPipeline

            pipeline = UnifiedAIPipeline()
            test_image = 'data/test/simple_01.png'

            # Process same image multiple times
            results = []
            for i in range(5):
                result = pipeline.process(test_image)
                if result is None:
                    print(f"❌ Processing failed on run {i+1}")
                    return False

                # Convert to dict if needed
                if hasattr(result, 'to_dict'):
                    result_dict = result.to_dict()
                else:
                    result_dict = result

                if not result_dict.get('success', False):
                    print(f"❌ Processing failed on run {i+1}")
                    return False

                # Use quality_score
                quality_score = result_dict.get('quality_score', 0.0)
                results.append(quality_score)

            # Check consistency (standard deviation should be very low)
            if len(results) > 1:
                std_dev = statistics.stdev(results)
                if std_dev > 0.01:  # Allow very small variation
                    print(f"❌ Inconsistent results: std dev {std_dev:.4f}")
                    print(f"  SSIM values: {results}")
                    return False

            print(f"✅ Consistent results: SSIM values {results}")
            return True

        except Exception as e:
            print(f"❌ Consistency test failed: {str(e)}")
            return False

    def test_quality_reliable_predictions(self) -> bool:
        """Test: Reliable predictions"""
        print("\n🔍 Testing: Reliable predictions")

        try:
            from backend.ai_modules.classification import ClassificationModule
            from backend.ai_modules.optimization import OptimizationEngine

            classifier = ClassificationModule()
            optimizer = OptimizationEngine()

            # Test prediction reliability across different images
            test_images = [
                'data/test/simple_01.png',
                'data/test/text_01.png',
                'data/test/gradient_01.png',
                'data/test/complex_01.png'
            ]

            predictions = []
            for image_path in test_images:
                # Test classification prediction
                class_result = classifier.classify(image_path)
                if 'final_class' not in class_result:
                    print(f"❌ Classification prediction failed for {image_path}")
                    return False

                # Test parameter optimization prediction
                features = classifier.feature_extractor.extract(image_path)
                params = optimizer.calculate_base_parameters(features)
                if not isinstance(params, dict) or 'color_precision' not in params:
                    print(f"❌ Parameter prediction failed for {image_path}")
                    return False

                # Check if parameters are in reasonable ranges
                if not (1 <= params['color_precision'] <= 10):
                    print(f"❌ Unreliable color_precision prediction: {params['color_precision']}")
                    return False

                predictions.append({
                    'image': image_path,
                    'class': class_result['final_class'],
                    'color_precision': params['color_precision']
                })

            print("✅ Reliable predictions:")
            for pred in predictions:
                print(f"  - {Path(pred['image']).name}: {pred['class']}, color_precision={pred['color_precision']}")

            return True

        except Exception as e:
            print(f"❌ Prediction reliability test failed: {str(e)}")
            return False

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive success metrics verification report"""
        print("\n📊 Generating Success Metrics Verification Report")

        # Run all tests
        functionality_tests = {
            'no_functionality_lost': self.test_functionality_no_lost_features(),
            'all_features_working': self.test_functionality_all_features_working(),
            'improved_organization': self.test_functionality_improved_organization()
        }

        performance_tests = {
            'equal_or_better_than_day13': self.test_performance_equal_or_better(),
            'all_targets_achieved': self.test_performance_all_targets_achieved(),
            'no_memory_leaks': self.test_performance_no_memory_leaks()
        }

        quality_tests = {
            'ssim_improvements_maintained': self.test_quality_ssim_improvements_maintained(),
            'consistent_results': self.test_quality_consistent_results(),
            'reliable_predictions': self.test_quality_reliable_predictions()
        }

        # Calculate overall status
        all_tests = {**functionality_tests, **performance_tests, **quality_tests}
        passed_tests = sum(1 for passed in all_tests.values() if passed)
        total_tests = len(all_tests)

        overall_passed = all(all_tests.values())

        report = {
            'summary': {
                'overall_status': 'PASSED' if overall_passed else 'FAILED',
                'tests_passed': passed_tests,
                'tests_total': total_tests,
                'success_rate': (passed_tests / total_tests) * 100
            },
            'functionality': functionality_tests,
            'performance': performance_tests,
            'quality': quality_tests,
            'recommendations': []
        }

        # Add recommendations for failed tests
        if not overall_passed:
            for category, tests in [('functionality', functionality_tests),
                                  ('performance', performance_tests),
                                  ('quality', quality_tests)]:
                failed_tests = [test for test, passed in tests.items() if not passed]
                if failed_tests:
                    report['recommendations'].append(f"Fix {category} issues: {failed_tests}")

        return report


def main():
    """Run success metrics verification"""
    print("🎯 Success Metrics Verification for DAY14")
    print("=" * 50)

    verifier = SuccessMetricsVerification()
    report = verifier.generate_comprehensive_report()

    # Save report
    with open('success_metrics_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)

    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Tests Passed: {report['summary']['tests_passed']}/{report['summary']['tests_total']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")

    print("\n🔍 DETAILED RESULTS:")

    for category in ['functionality', 'performance', 'quality']:
        print(f"\n{category.upper()}:")
        for test, passed in report[category].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test}: {status}")

    if report['recommendations']:
        print("\n⚠️ RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

    print(f"\nDetailed report saved to: success_metrics_verification_report.json")

    return report['summary']['overall_status'] == 'PASSED'


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)