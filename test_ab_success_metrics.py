#!/usr/bin/env python3
"""
Test Success Metrics for Day 9 A/B Testing Framework
Run empirical validation using real AI-enhanced vs baseline converters
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Import real converters
from converters.ai_enhanced_converter import AIEnhancedConverter
from converters.vtracer_converter import VTracerConverter

# Import A/B testing framework
from ai_modules.testing.ab_framework import ABTestFramework, TestConfig
from ai_modules.testing.statistical_analysis import StatisticalAnalyzer
from ai_modules.testing.test_orchestrator import ABTestOrchestrator, CampaignConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuccessMetricsValidator:
    """Validates Day 9 Success Metrics through empirical testing"""

    def __init__(self):
        self.ai_converter = AIEnhancedConverter()
        self.baseline_converter = VTracerConverter()
        self.ab_framework = ABTestFramework()
        self.analyzer = StatisticalAnalyzer()

        # Replace mock converters with real ones
        self.ab_framework.test_groups = {
            'control': self._baseline_convert,
            'treatment': self._ai_enhanced_convert
        }

        # Test dataset
        self.test_images = self._get_test_images()

        # Results storage
        self.results = []

    def _baseline_convert(self, image_path: str) -> Dict[str, Any]:
        """Baseline converter wrapper for A/B framework"""
        try:
            start_time = time.time()
            svg_content = self.baseline_converter.convert(image_path)
            conversion_time = time.time() - start_time

            return {
                'svg': svg_content,
                'processing_time': conversion_time,
                'parameters': {
                    'method': 'baseline_vtracer',
                    'color_precision': 6,
                    'layer_difference': 16
                },
                'success': True
            }
        except Exception as e:
            return {
                'svg': None,
                'processing_time': 0.0,
                'parameters': {'method': 'baseline_vtracer'},
                'success': False,
                'error': str(e)
            }

    def _ai_enhanced_convert(self, image_path: str) -> Dict[str, Any]:
        """AI-enhanced converter wrapper for A/B framework"""
        try:
            start_time = time.time()
            svg_content = self.ai_converter.convert(image_path)
            conversion_time = time.time() - start_time

            return {
                'svg': svg_content,
                'processing_time': conversion_time,
                'parameters': {
                    'method': 'ai_enhanced',
                    'optimization': 'method_1_correlation'
                },
                'success': True
            }
        except Exception as e:
            return {
                'svg': None,
                'processing_time': 0.0,
                'parameters': {'method': 'ai_enhanced'},
                'success': False,
                'error': str(e)
            }

    def _get_test_images(self) -> List[str]:
        """Get list of test images across all categories"""
        base_path = Path("data/logos")
        images = []

        categories = ["simple_geometric", "text_based", "gradients", "complex", "abstract"]

        for category in categories:
            category_path = base_path / category
            if category_path.exists():
                category_images = list(category_path.glob("*.png"))
                # Filter out processed images
                category_images = [
                    str(img) for img in category_images
                    if "optimized" not in str(img) and ".cache" not in str(img)
                ]
                images.extend(category_images)

        logger.info(f"Found {len(images)} test images across {len(categories)} categories")
        return images

    def run_empirical_validation(self) -> Dict[str, Any]:
        """Run complete empirical validation of Success Metrics"""
        logger.info("Starting empirical validation of Success Metrics...")

        validation_results = {
            'total_images_tested': len(self.test_images),
            'categories_tested': self._get_category_breakdown(),
            'success_metrics': {},
            'detailed_results': [],
            'statistical_analysis': {},
            'recommendations': []
        }

        # Run A/B tests on all images
        logger.info(f"Running A/B tests on {len(self.test_images)} images...")

        test_config = TestConfig(
            assignment_method='sequential',  # Ensure even split
            min_sample_size=len(self.test_images),
            enable_quality_measurement=True
        )

        for i, image_path in enumerate(self.test_images):
            logger.info(f"Testing image {i+1}/{len(self.test_images)}: {Path(image_path).name}")

            try:
                result = self.ab_framework.run_test(image_path, test_config)
                self.results.append(result)
                validation_results['detailed_results'].append({
                    'image': image_path,
                    'category': self._get_image_category(image_path),
                    'group': result.group,
                    'quality': result.quality,
                    'processing_time': result.processing_time,
                    'success': result.success
                })
            except Exception as e:
                logger.error(f"Test failed for {image_path}: {e}")
                validation_results['detailed_results'].append({
                    'image': image_path,
                    'category': self._get_image_category(image_path),
                    'error': str(e),
                    'success': False
                })

        # Analyze results for Success Metrics
        validation_results['success_metrics'] = self._validate_success_metrics()
        validation_results['statistical_analysis'] = self._perform_statistical_analysis()
        validation_results['recommendations'] = self._generate_recommendations()

        logger.info("Empirical validation completed!")
        return validation_results

    def _validate_success_metrics(self) -> Dict[str, Any]:
        """Validate each Success Metric against actual results"""
        logger.info("Validating Success Metrics...")

        # Split results by group
        control_results = [r for r in self.results if r.group == 'control' and r.success]
        treatment_results = [r for r in self.results if r.group == 'treatment' and r.success]

        if not control_results or not treatment_results:
            return {
                'error': 'Insufficient successful conversions for analysis',
                'control_count': len(control_results),
                'treatment_count': len(treatment_results)
            }

        # Calculate quality improvements
        control_quality = [r.quality.get('ssim', 0.0) for r in control_results]
        treatment_quality = [r.quality.get('ssim', 0.0) for r in treatment_results]

        control_avg = sum(control_quality) / len(control_quality)
        treatment_avg = sum(treatment_quality) / len(treatment_quality)
        improvement_pct = ((treatment_avg - control_avg) / control_avg) * 100

        # Performance analysis
        control_times = [r.processing_time for r in control_results]
        treatment_times = [r.processing_time for r in treatment_results]

        control_avg_time = sum(control_times) / len(control_times)
        treatment_avg_time = sum(treatment_times) / len(treatment_times)
        time_change_pct = ((treatment_avg_time - control_avg_time) / control_avg_time) * 100

        # Statistical significance
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(treatment_quality, control_quality)
            is_significant = p_value < 0.05
        except ImportError:
            logger.warning("scipy not available for t-test")
            t_stat, p_value, is_significant = None, None, False

        # Category consistency
        category_analysis = self._analyze_category_consistency()

        success_metrics = {
            'quality_improvement': {
                'measured': improvement_pct,
                'target': 15.0,
                'meets_target': improvement_pct > 15.0,
                'control_avg_ssim': control_avg,
                'treatment_avg_ssim': treatment_avg
            },
            'statistical_significance': {
                'p_value': p_value,
                'target': 0.05,
                'meets_target': is_significant,
                't_statistic': t_stat,
                'sample_sizes': {
                    'control': len(control_results),
                    'treatment': len(treatment_results)
                }
            },
            'category_consistency': {
                'analysis': category_analysis,
                'meets_target': category_analysis.get('all_categories_improved', False)
            },
            'performance_regression': {
                'time_change_pct': time_change_pct,
                'target': 'no_significant_increase',
                'meets_target': time_change_pct < 50.0,  # Allow up to 50% increase for AI processing
                'control_avg_time': control_avg_time,
                'treatment_avg_time': treatment_avg_time
            }
        }

        return success_metrics

    def _analyze_category_consistency(self) -> Dict[str, Any]:
        """Analyze improvement consistency across image categories"""
        categories = ["simple_geometric", "text_based", "gradients", "complex", "abstract"]
        category_results = {}

        for category in categories:
            category_control = [
                r for r in self.results
                if r.group == 'control' and r.success and category in r.image_path
            ]
            category_treatment = [
                r for r in self.results
                if r.group == 'treatment' and r.success and category in r.image_path
            ]

            if category_control and category_treatment:
                control_avg = sum(r.quality.get('ssim', 0.0) for r in category_control) / len(category_control)
                treatment_avg = sum(r.quality.get('ssim', 0.0) for r in category_treatment) / len(category_treatment)
                improvement = ((treatment_avg - control_avg) / control_avg) * 100

                category_results[category] = {
                    'control_avg': control_avg,
                    'treatment_avg': treatment_avg,
                    'improvement_pct': improvement,
                    'sample_sizes': {
                        'control': len(category_control),
                        'treatment': len(category_treatment)
                    }
                }
            else:
                category_results[category] = {
                    'error': 'Insufficient samples',
                    'control_count': len(category_control),
                    'treatment_count': len(category_treatment)
                }

        # Check if all categories show improvement
        valid_categories = [k for k, v in category_results.items() if 'improvement_pct' in v]
        improved_categories = [k for k, v in category_results.items() if v.get('improvement_pct', 0) > 0]

        return {
            'categories': category_results,
            'total_categories': len(categories),
            'valid_categories': len(valid_categories),
            'improved_categories': len(improved_categories),
            'all_categories_improved': len(improved_categories) == len(valid_categories) and len(valid_categories) > 0
        }

    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        logger.info("Performing statistical analysis...")

        # Convert results to format expected by StatisticalAnalyzer
        analysis_results = []
        for result in self.results:
            if result.success:
                analysis_results.append({
                    'group': result.group,
                    'quality': result.quality,
                    'processing_time': result.processing_time,
                    'image': result.image_path
                })

        try:
            return self.analyzer.analyze_results(analysis_results)
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []

        success_metrics = self._validate_success_metrics()

        # Quality improvement recommendations
        quality = success_metrics.get('quality_improvement', {})
        if quality.get('meets_target'):
            recommendations.append(f"✅ Quality improvement target met: {quality['measured']:.1f}% > 15%")
        else:
            recommendations.append(f"❌ Quality improvement below target: {quality['measured']:.1f}% < 15%")
            recommendations.append("Consider: Tuning AI optimization parameters or adding more training data")

        # Statistical significance recommendations
        stats = success_metrics.get('statistical_significance', {})
        if stats.get('meets_target'):
            recommendations.append(f"✅ Statistical significance achieved: p = {stats['p_value']:.4f} < 0.05")
        else:
            recommendations.append(f"❌ No statistical significance: p = {stats.get('p_value', 'N/A')}")
            recommendations.append("Consider: Increasing sample size or improving AI model performance")

        # Category consistency recommendations
        consistency = success_metrics.get('category_consistency', {})
        if consistency.get('meets_target'):
            recommendations.append("✅ Consistent improvements across all image categories")
        else:
            recommendations.append("❌ Inconsistent improvements across image categories")
            recommendations.append("Consider: Category-specific optimization or model fine-tuning")

        # Performance recommendations
        performance = success_metrics.get('performance_regression', {})
        if performance.get('meets_target'):
            recommendations.append(f"✅ No significant performance regression: {performance['time_change_pct']:+.1f}%")
        else:
            recommendations.append(f"❌ Performance regression detected: {performance['time_change_pct']:+.1f}%")
            recommendations.append("Consider: Optimizing AI inference speed or caching strategies")

        return recommendations

    def _get_category_breakdown(self) -> Dict[str, int]:
        """Get breakdown of images by category"""
        breakdown = {}
        for image_path in self.test_images:
            category = self._get_image_category(image_path)
            breakdown[category] = breakdown.get(category, 0) + 1
        return breakdown

    def _get_image_category(self, image_path: str) -> str:
        """Extract category from image path"""
        path_parts = Path(image_path).parts
        for part in path_parts:
            if part in ["simple_geometric", "text_based", "gradients", "complex", "abstract"]:
                return part
        return "unknown"

    def save_results(self, results: Dict[str, Any], output_path: str = "success_metrics_validation.json"):
        """Save validation results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary of results"""
        print("\n" + "="*80)
        print("SUCCESS METRICS VALIDATION SUMMARY")
        print("="*80)

        metrics = results.get('success_metrics', {})

        print(f"\n📊 DATASET:")
        print(f"   • Total images tested: {results['total_images_tested']}")
        print(f"   • Categories: {results['categories_tested']}")

        print(f"\n🎯 SUCCESS METRICS RESULTS:")

        # Quality improvement
        quality = metrics.get('quality_improvement', {})
        status = "✅ PASS" if quality.get('meets_target') else "❌ FAIL"
        print(f"   • Quality improvement: {status}")
        print(f"     Target: >15% | Measured: {quality.get('measured', 0):.1f}%")

        # Statistical significance
        stats = metrics.get('statistical_significance', {})
        status = "✅ PASS" if stats.get('meets_target') else "❌ FAIL"
        print(f"   • Statistical significance: {status}")
        print(f"     Target: p < 0.05 | Measured: p = {stats.get('p_value', 'N/A')}")

        # Category consistency
        consistency = metrics.get('category_consistency', {})
        status = "✅ PASS" if consistency.get('meets_target') else "❌ FAIL"
        print(f"   • Category consistency: {status}")

        # Performance regression
        performance = metrics.get('performance_regression', {})
        status = "✅ PASS" if performance.get('meets_target') else "❌ FAIL"
        print(f"   • Performance regression: {status}")
        print(f"     Time change: {performance.get('time_change_pct', 0):+.1f}%")

        print(f"\n💡 RECOMMENDATIONS:")
        for rec in results.get('recommendations', []):
            print(f"   • {rec}")

        print("\n" + "="*80)

def main():
    """Run Success Metrics validation"""
    try:
        validator = SuccessMetricsValidator()
        results = validator.run_empirical_validation()

        # Save and display results
        validator.save_results(results)
        validator.print_summary(results)

        # Determine overall success
        metrics = results.get('success_metrics', {})
        all_passed = all([
            metrics.get('quality_improvement', {}).get('meets_target', False),
            metrics.get('statistical_significance', {}).get('meets_target', False),
            metrics.get('category_consistency', {}).get('meets_target', False),
            metrics.get('performance_regression', {}).get('meets_target', False)
        ])

        if all_passed:
            print("\n🎉 ALL SUCCESS METRICS PASSED! Day 9 objectives achieved.")
            return 0
        else:
            print("\n⚠️  Some Success Metrics failed. Review recommendations above.")
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ VALIDATION ERROR: {e}")
        return 2

if __name__ == "__main__":
    exit(main())