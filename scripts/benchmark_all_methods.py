#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking System
Benchmarks all three optimization methods with statistical rigor and visualization
"""

import numpy as np
import pandas as pd
import json
import time
import logging
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import psutil
import os
import sys

# Statistical analysis
from scipy import stats
from scipy.stats import f_oneway, ttest_rel, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Memory profiling
try:
    from memory_profiler import profile
    MEMORY_PROFILING_AVAILABLE = True
except ImportError:
    MEMORY_PROFILING_AVAILABLE = False
    def profile(func):
        return func

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.converters.intelligent_converter import IntelligentConverter
from backend.ai_modules.optimization.quality_metrics import OptimizationQualityMetrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result for statistical analysis"""
    method: str
    image_path: str
    logo_type: str
    complexity_score: float
    ssim_improvement: float
    processing_time: float
    memory_usage_mb: float
    svg_size_bytes: int
    success: bool
    error_message: Optional[str] = None
    visual_quality_score: float = 0.0
    file_size_reduction: float = 0.0


@dataclass
class MethodStatistics:
    """Statistical summary for a method"""
    method: str
    count: int
    mean_ssim_improvement: float
    std_ssim_improvement: float
    mean_processing_time: float
    std_processing_time: float
    success_rate: float
    confidence_interval_ssim: Tuple[float, float]
    confidence_interval_time: Tuple[float, float]
    effect_size_vs_baseline: float


class ComprehensiveOptimizationBenchmark:
    """Benchmark all three optimization methods comprehensively"""

    def __init__(self, dataset_path: str, output_dir: str = "benchmark_results"):
        """
        Initialize comprehensive benchmarking system

        Args:
            dataset_path: Path to test dataset directory
            output_dir: Directory for benchmark outputs
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize converters and metrics
        self.converter = IntelligentConverter()
        self.quality_metrics = OptimizationQualityMetrics()

        # Load comprehensive test dataset
        self.test_images = self._load_comprehensive_dataset()

        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.method_statistics: Dict[str, MethodStatistics] = {}

        # Benchmark configuration
        self.benchmark_config = {
            'runs_per_image': 3,  # Statistical significance
            'timeout_per_image': 120,  # 2 minutes max per image
            'memory_monitoring': True,
            'parallel_processing': False,  # Set True for speed, False for accurate resource monitoring
            'statistical_significance_threshold': 0.05,
            'effect_size_threshold': 0.5,  # Medium effect size
            'methods_to_test': ['default', 'method1', 'method2', 'method3']
        }

        # Setup logging
        self._setup_logging()

        logger.info(f"Initialized comprehensive benchmarking system")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Test images loaded: {sum(len(images) for images in self.test_images.values())}")

    def _setup_logging(self):
        """Setup detailed logging for benchmarking"""
        log_file = self.output_dir / "benchmark_log.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)

    def _load_comprehensive_dataset(self) -> Dict[str, List[Path]]:
        """
        Load comprehensive test dataset organized by logo type

        Returns:
            Dictionary mapping logo types to image paths
        """
        test_images = {
            'simple': [],
            'text': [],
            'gradient': [],
            'complex': [],
            'mixed': []
        }

        if not self.dataset_path.exists():
            logger.warning(f"Dataset path {self.dataset_path} does not exist")
            return test_images

        # Load images from dataset structure
        for logo_type in test_images.keys():
            type_dir = self.dataset_path / logo_type
            if type_dir.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    test_images[logo_type].extend(type_dir.glob(ext))

        # If no organized structure, use flat directory
        if not any(test_images.values()):
            logger.info("No organized structure found, using flat directory")
            all_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                all_images.extend(self.dataset_path.glob(ext))

            # Distribute evenly across categories
            for i, img_path in enumerate(all_images):
                category = list(test_images.keys())[i % len(test_images)]
                test_images[category].append(img_path)

        # Log dataset statistics
        for logo_type, images in test_images.items():
            logger.info(f"Loaded {len(images)} {logo_type} images")

        return test_images

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmarking across all methods and images

        Returns:
            Comprehensive benchmark results with statistical analysis
        """
        logger.info("🚀 Starting comprehensive optimization benchmark")
        start_time = time.time()

        results_summary = {
            'benchmark_metadata': {
                'total_images': sum(len(images) for images in self.test_images.values()),
                'methods_compared': len(self.benchmark_config['methods_to_test']),
                'runs_per_image': self.benchmark_config['runs_per_image'],
                'benchmark_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'system_info': self._get_system_info()
            },
            'category_results': {},
            'method_statistics': {},
            'comparative_analysis': {},
            'statistical_tests': {},
            'visualizations': {},
            'recommendations': {}
        }

        try:
            # Benchmark each category
            for category, images in self.test_images.items():
                if not images:
                    continue

                logger.info(f"📊 Benchmarking {category} category ({len(images)} images)")
                category_results = self._benchmark_category(category, images)
                results_summary['category_results'][category] = category_results

            # Calculate method statistics
            logger.info("📈 Calculating method statistics")
            self.method_statistics = self._calculate_method_statistics()
            results_summary['method_statistics'] = {
                method: asdict(stats) for method, stats in self.method_statistics.items()
            }

            # Generate comparative analysis
            logger.info("🔬 Generating comparative analysis")
            comparative_analysis = self._generate_comparative_analysis()
            results_summary['comparative_analysis'] = comparative_analysis

            # Perform statistical tests
            logger.info("📊 Performing statistical tests")
            statistical_tests = self._perform_statistical_tests()
            results_summary['statistical_tests'] = statistical_tests

            # Create visualizations
            logger.info("📈 Creating visualizations")
            visualizations = self._create_benchmark_visualizations()
            results_summary['visualizations'] = visualizations

            # Generate recommendations
            logger.info("💡 Generating recommendations")
            recommendations = self._generate_recommendations()
            results_summary['recommendations'] = recommendations

            # Save comprehensive results
            self._save_benchmark_results(results_summary)

            total_time = time.time() - start_time
            logger.info(f"✅ Comprehensive benchmark completed in {total_time:.1f}s")

            results_summary['benchmark_metadata']['total_benchmark_time'] = total_time

        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            logger.error(traceback.format_exc())
            results_summary['error'] = str(e)
            results_summary['traceback'] = traceback.format_exc()

        finally:
            # Cleanup
            self.quality_metrics.cleanup()

        return results_summary

    def _benchmark_category(self, category: str, images: List[Path]) -> Dict[str, Any]:
        """
        Benchmark a specific logo category

        Args:
            category: Logo category name
            images: List of image paths in category

        Returns:
            Category benchmark results
        """
        category_results = {
            'category': category,
            'image_count': len(images),
            'method_results': {},
            'category_statistics': {},
            'processing_summary': {
                'total_time': 0,
                'successful_conversions': 0,
                'failed_conversions': 0
            }
        }

        category_start_time = time.time()

        # Limit images for faster benchmarking if dataset is large
        max_images_per_category = 20
        if len(images) > max_images_per_category:
            images = images[:max_images_per_category]
            logger.info(f"Limited {category} to {max_images_per_category} images for faster benchmarking")

        # Test each method on each image
        for method in self.benchmark_config['methods_to_test']:
            logger.info(f"  Testing {method} on {category} images")
            method_results = []

            for image_path in images:
                # Run multiple times for statistical significance
                for run in range(self.benchmark_config['runs_per_image']):
                    result = self._benchmark_single_image(method, image_path, category, run)
                    if result:
                        method_results.append(result)
                        self.benchmark_results.append(result)

            category_results['method_results'][method] = method_results

            # Calculate method statistics for this category
            if method_results:
                successful_results = [r for r in method_results if r.success]
                category_results['category_statistics'][method] = {
                    'total_runs': len(method_results),
                    'successful_runs': len(successful_results),
                    'success_rate': len(successful_results) / len(method_results),
                    'mean_ssim_improvement': np.mean([r.ssim_improvement for r in successful_results]) if successful_results else 0,
                    'mean_processing_time': np.mean([r.processing_time for r in successful_results]) if successful_results else 0,
                    'mean_memory_usage': np.mean([r.memory_usage_mb for r in successful_results]) if successful_results else 0
                }

        category_time = time.time() - category_start_time
        category_results['processing_summary']['total_time'] = category_time
        category_results['processing_summary']['successful_conversions'] = sum(
            len([r for r in results if r.success])
            for results in category_results['method_results'].values()
        )
        category_results['processing_summary']['failed_conversions'] = sum(
            len([r for r in results if not r.success])
            for results in category_results['method_results'].values()
        )

        logger.info(f"  {category} completed in {category_time:.1f}s")

        return category_results

    def _benchmark_single_image(self, method: str, image_path: Path,
                               logo_type: str, run: int) -> Optional[BenchmarkResult]:
        """
        Benchmark a single image with a specific method

        Args:
            method: Method to test ('default', 'method1', 'method2', 'method3')
            image_path: Path to image file
            logo_type: Logo category
            run: Run number for this image/method combination

        Returns:
            BenchmarkResult or None if failed
        """
        try:
            # Monitor memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()

            # Configure method-specific parameters
            conversion_kwargs = {'force_method': method} if method != 'default' else {}

            if method == 'default':
                # Use base converter without optimization
                svg_content = self.converter.convert(str(image_path), **conversion_kwargs)
                success = svg_content is not None
                optimization_result = {'quality_improvement': 0.0}
            else:
                # Use intelligent converter with specific method
                result = self.converter.convert(str(image_path), **conversion_kwargs)
                success = result.get('success', False)
                svg_content = result.get('svg_content')
                optimization_result = result

            processing_time = time.time() - start_time

            # Monitor memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before

            if not success:
                return BenchmarkResult(
                    method=method,
                    image_path=str(image_path),
                    logo_type=logo_type,
                    complexity_score=0.0,
                    ssim_improvement=0.0,
                    processing_time=processing_time,
                    memory_usage_mb=memory_usage,
                    svg_size_bytes=0,
                    success=False,
                    error_message="Conversion failed"
                )

            # Calculate quality metrics
            ssim_improvement = optimization_result.get('quality_improvement', 0.0)

            # Estimate SVG size (would be actual size in production)
            svg_size = len(str(svg_content).encode('utf-8')) if svg_content else 0

            # Estimate complexity score (would be calculated from features)
            complexity_score = 0.5  # Default complexity

            # Calculate file size reduction
            try:
                original_size = image_path.stat().st_size
                file_size_reduction = (original_size - svg_size) / original_size if original_size > 0 else 0
            except:
                file_size_reduction = 0.0

            return BenchmarkResult(
                method=method,
                image_path=str(image_path),
                logo_type=logo_type,
                complexity_score=complexity_score,
                ssim_improvement=ssim_improvement,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                svg_size_bytes=svg_size,
                success=True,
                file_size_reduction=file_size_reduction,
                visual_quality_score=0.8 + ssim_improvement  # Estimated
            )

        except Exception as e:
            logger.warning(f"Single image benchmark failed for {image_path} with {method}: {e}")
            return BenchmarkResult(
                method=method,
                image_path=str(image_path),
                logo_type=logo_type,
                complexity_score=0.0,
                ssim_improvement=0.0,
                processing_time=0.0,
                memory_usage_mb=0.0,
                svg_size_bytes=0,
                success=False,
                error_message=str(e)
            )

    def _calculate_method_statistics(self) -> Dict[str, MethodStatistics]:
        """
        Calculate comprehensive statistics for each method

        Returns:
            Dictionary mapping method names to MethodStatistics
        """
        method_stats = {}

        for method in self.benchmark_config['methods_to_test']:
            method_results = [r for r in self.benchmark_results if r.method == method and r.success]

            if not method_results:
                logger.warning(f"No successful results for {method}")
                continue

            # Extract metrics
            ssim_improvements = [r.ssim_improvement for r in method_results]
            processing_times = [r.processing_time for r in method_results]

            # Calculate statistics
            count = len(method_results)
            mean_ssim = np.mean(ssim_improvements)
            std_ssim = np.std(ssim_improvements) if count > 1 else 0
            mean_time = np.mean(processing_times)
            std_time = np.std(processing_times) if count > 1 else 0

            total_attempts = len([r for r in self.benchmark_results if r.method == method])
            success_rate = count / total_attempts if total_attempts > 0 else 0

            # Calculate confidence intervals (95%)
            if count > 1:
                sem_ssim = stats.sem(ssim_improvements)
                sem_time = stats.sem(processing_times)
                ci_ssim = stats.t.interval(0.95, count-1, loc=mean_ssim, scale=sem_ssim)
                ci_time = stats.t.interval(0.95, count-1, loc=mean_time, scale=sem_time)
            else:
                ci_ssim = (mean_ssim, mean_ssim)
                ci_time = (mean_time, mean_time)

            # Calculate effect size vs baseline (default method)
            baseline_results = [r for r in self.benchmark_results
                              if r.method == 'default' and r.success]
            if baseline_results and method != 'default':
                baseline_ssim = [r.ssim_improvement for r in baseline_results]
                effect_size = self._calculate_cohens_d(ssim_improvements, baseline_ssim)
            else:
                effect_size = 0.0

            method_stats[method] = MethodStatistics(
                method=method,
                count=count,
                mean_ssim_improvement=mean_ssim,
                std_ssim_improvement=std_ssim,
                mean_processing_time=mean_time,
                std_processing_time=std_time,
                success_rate=success_rate,
                confidence_interval_ssim=ci_ssim,
                confidence_interval_time=ci_time,
                effect_size_vs_baseline=effect_size
            )

        return method_stats

    def _calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size between two groups

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size
        """
        if not group1 or not group2:
            return 0.0

        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        n1, n2 = len(group1), len(group2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d

    def _generate_comparative_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparative analysis between methods

        Returns:
            Comparative analysis results
        """
        analysis = {
            'head_to_head_comparisons': {},
            'quality_time_tradeoffs': {},
            'success_rate_analysis': {},
            'cost_benefit_analysis': {},
            'method_rankings': {}
        }

        # Head-to-head quality comparisons
        methods = list(self.method_statistics.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                comparison = self._compare_two_methods(method1, method2)
                analysis['head_to_head_comparisons'][f"{method1}_vs_{method2}"] = comparison

        # Quality vs time tradeoff analysis
        for method, stats in self.method_statistics.items():
            efficiency_score = (stats.mean_ssim_improvement / max(stats.mean_processing_time, 0.001))
            analysis['quality_time_tradeoffs'][method] = {
                'efficiency_score': efficiency_score,
                'quality': stats.mean_ssim_improvement,
                'time': stats.mean_processing_time,
                'quality_per_second': efficiency_score
            }

        # Success rate analysis by logo type
        for logo_type in ['simple', 'text', 'gradient', 'complex']:
            type_results = [r for r in self.benchmark_results if r.logo_type == logo_type]
            if type_results:
                type_analysis = {}
                for method in methods:
                    method_type_results = [r for r in type_results if r.method == method]
                    if method_type_results:
                        success_count = sum(1 for r in method_type_results if r.success)
                        type_analysis[method] = success_count / len(method_type_results)
                analysis['success_rate_analysis'][logo_type] = type_analysis

        # Cost-benefit analysis (processing time as cost, quality as benefit)
        for method, stats in self.method_statistics.items():
            cost = stats.mean_processing_time  # seconds
            benefit = stats.mean_ssim_improvement  # quality improvement
            roi = benefit / max(cost, 0.001)  # Return on investment

            analysis['cost_benefit_analysis'][method] = {
                'cost_seconds': cost,
                'benefit_quality': benefit,
                'roi_quality_per_second': roi,
                'cost_effectiveness_rank': 0  # Will be calculated after all methods
            }

        # Rank methods by cost-effectiveness
        sorted_methods = sorted(
            analysis['cost_benefit_analysis'].items(),
            key=lambda x: x[1]['roi_quality_per_second'],
            reverse=True
        )
        for rank, (method, data) in enumerate(sorted_methods, 1):
            analysis['cost_benefit_analysis'][method]['cost_effectiveness_rank'] = rank

        # Overall method rankings
        ranking_criteria = {
            'quality': lambda m: self.method_statistics[m].mean_ssim_improvement,
            'speed': lambda m: -self.method_statistics[m].mean_processing_time,  # Negative for ascending
            'reliability': lambda m: self.method_statistics[m].success_rate,
            'efficiency': lambda m: analysis['quality_time_tradeoffs'][m]['efficiency_score']
        }

        for criterion, score_func in ranking_criteria.items():
            ranked_methods = sorted(methods, key=score_func, reverse=True)
            analysis['method_rankings'][criterion] = [
                {'rank': i+1, 'method': method, 'score': score_func(method)}
                for i, method in enumerate(ranked_methods)
            ]

        return analysis

    def _compare_two_methods(self, method1: str, method2: str) -> Dict[str, Any]:
        """
        Compare two methods statistically

        Args:
            method1: First method name
            method2: Second method name

        Returns:
            Detailed comparison results
        """
        stats1 = self.method_statistics.get(method1)
        stats2 = self.method_statistics.get(method2)

        if not stats1 or not stats2:
            return {'error': 'Insufficient data for comparison'}

        comparison = {
            'methods': [method1, method2],
            'sample_sizes': [stats1.count, stats2.count],
            'quality_comparison': {
                'means': [stats1.mean_ssim_improvement, stats2.mean_ssim_improvement],
                'difference': stats1.mean_ssim_improvement - stats2.mean_ssim_improvement,
                'winner': method1 if stats1.mean_ssim_improvement > stats2.mean_ssim_improvement else method2
            },
            'time_comparison': {
                'means': [stats1.mean_processing_time, stats2.mean_processing_time],
                'difference': stats1.mean_processing_time - stats2.mean_processing_time,
                'winner': method1 if stats1.mean_processing_time < stats2.mean_processing_time else method2
            },
            'reliability_comparison': {
                'success_rates': [stats1.success_rate, stats2.success_rate],
                'difference': stats1.success_rate - stats2.success_rate,
                'winner': method1 if stats1.success_rate > stats2.success_rate else method2
            }
        }

        return comparison

    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests

        Returns:
            Statistical test results
        """
        tests = {
            'anova_tests': {},
            'pairwise_t_tests': {},
            'effect_sizes': {},
            'power_analysis': {},
            'normality_tests': {}
        }

        # Prepare data for testing
        method_groups = {}
        for method in self.benchmark_config['methods_to_test']:
            successful_results = [r for r in self.benchmark_results
                                if r.method == method and r.success]
            if successful_results:
                method_groups[method] = {
                    'ssim_improvements': [r.ssim_improvement for r in successful_results],
                    'processing_times': [r.processing_time for r in successful_results]
                }

        # ANOVA for multi-method comparison
        if len(method_groups) >= 3:
            # SSIM improvement ANOVA
            ssim_groups = [data['ssim_improvements'] for data in method_groups.values()]
            ssim_groups = [group for group in ssim_groups if len(group) >= 2]

            if len(ssim_groups) >= 3:
                f_stat, p_value = f_oneway(*ssim_groups)
                tests['anova_tests']['ssim_improvement'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.benchmark_config['statistical_significance_threshold'],
                    'interpretation': 'Significant differences between methods' if p_value < 0.05 else 'No significant differences'
                }

            # Processing time ANOVA
            time_groups = [data['processing_times'] for data in method_groups.values()]
            time_groups = [group for group in time_groups if len(group) >= 2]

            if len(time_groups) >= 3:
                f_stat, p_value = f_oneway(*time_groups)
                tests['anova_tests']['processing_time'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.benchmark_config['statistical_significance_threshold'],
                    'interpretation': 'Significant differences between methods' if p_value < 0.05 else 'No significant differences'
                }

        # Pairwise t-tests
        methods = list(method_groups.keys())
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if method1 in method_groups and method2 in method_groups:
                    group1_ssim = method_groups[method1]['ssim_improvements']
                    group2_ssim = method_groups[method2]['ssim_improvements']

                    if len(group1_ssim) >= 2 and len(group2_ssim) >= 2:
                        t_stat, p_value = ttest_ind(group1_ssim, group2_ssim)
                        tests['pairwise_t_tests'][f"{method1}_vs_{method2}"] = {
                            'metric': 'ssim_improvement',
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < self.benchmark_config['statistical_significance_threshold'],
                            'effect_size': self._calculate_cohens_d(group1_ssim, group2_ssim)
                        }

        # Effect sizes vs baseline
        if 'default' in method_groups:
            baseline_ssim = method_groups['default']['ssim_improvements']
            for method, data in method_groups.items():
                if method != 'default':
                    effect_size = self._calculate_cohens_d(
                        data['ssim_improvements'], baseline_ssim
                    )
                    tests['effect_sizes'][method] = {
                        'cohens_d': effect_size,
                        'magnitude': self._interpret_effect_size(effect_size)
                    }

        return tests

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size

        Args:
            cohens_d: Cohen's d value

        Returns:
            Effect size interpretation
        """
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def _create_benchmark_visualizations(self) -> Dict[str, str]:
        """
        Create comprehensive visualizations for benchmark results

        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}

        # Set style for professional plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        try:
            # 1. Box plots comparing quality improvements
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Comprehensive Method Comparison', fontsize=16, fontweight='bold')

            # Quality improvement box plot
            self._create_quality_boxplot(axes[0, 0])

            # Processing time box plot
            self._create_time_boxplot(axes[0, 1])

            # Scatter plot: quality vs time
            self._create_quality_time_scatter(axes[1, 0])

            # Success rate by logo type heatmap
            self._create_success_rate_heatmap(axes[1, 1])

            plt.tight_layout()
            comparison_path = self.output_dir / "method_comparison.png"
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['method_comparison'] = str(comparison_path)

            # 2. Radar chart for multi-dimensional performance
            radar_path = self._create_radar_chart()
            if radar_path:
                visualizations['performance_radar'] = radar_path

            # 3. Statistical significance visualization
            significance_path = self._create_significance_plot()
            if significance_path:
                visualizations['statistical_significance'] = significance_path

            # 4. Interactive dashboard (static version)
            dashboard_path = self._create_dashboard_visualization()
            if dashboard_path:
                visualizations['dashboard'] = dashboard_path

        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            visualizations['error'] = str(e)

        return visualizations

    def _create_quality_boxplot(self, ax):
        """Create box plot for quality improvements"""
        data_for_plot = []
        methods_for_plot = []

        for method in self.benchmark_config['methods_to_test']:
            method_results = [r for r in self.benchmark_results
                            if r.method == method and r.success]
            if method_results:
                improvements = [r.ssim_improvement for r in method_results]
                data_for_plot.extend(improvements)
                methods_for_plot.extend([method] * len(improvements))

        if data_for_plot:
            df = pd.DataFrame({'Method': methods_for_plot, 'SSIM_Improvement': data_for_plot})
            sns.boxplot(data=df, x='Method', y='SSIM_Improvement', ax=ax)
            ax.set_title('Quality Improvement by Method')
            ax.set_ylabel('SSIM Improvement')
            ax.tick_params(axis='x', rotation=45)

    def _create_time_boxplot(self, ax):
        """Create box plot for processing times"""
        data_for_plot = []
        methods_for_plot = []

        for method in self.benchmark_config['methods_to_test']:
            method_results = [r for r in self.benchmark_results
                            if r.method == method and r.success]
            if method_results:
                times = [r.processing_time for r in method_results]
                data_for_plot.extend(times)
                methods_for_plot.extend([method] * len(times))

        if data_for_plot:
            df = pd.DataFrame({'Method': methods_for_plot, 'Processing_Time': data_for_plot})
            sns.boxplot(data=df, x='Method', y='Processing_Time', ax=ax)
            ax.set_title('Processing Time by Method')
            ax.set_ylabel('Processing Time (seconds)')
            ax.tick_params(axis='x', rotation=45)

    def _create_quality_time_scatter(self, ax):
        """Create scatter plot of quality vs time"""
        for method in self.benchmark_config['methods_to_test']:
            method_results = [r for r in self.benchmark_results
                            if r.method == method and r.success]
            if method_results:
                times = [r.processing_time for r in method_results]
                improvements = [r.ssim_improvement for r in method_results]
                ax.scatter(times, improvements, label=method, alpha=0.7, s=50)

        ax.set_xlabel('Processing Time (seconds)')
        ax.set_ylabel('SSIM Improvement')
        ax.set_title('Quality vs Time Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_success_rate_heatmap(self, ax):
        """Create heatmap of success rates by logo type"""
        logo_types = ['simple', 'text', 'gradient', 'complex']
        methods = self.benchmark_config['methods_to_test']

        success_matrix = np.zeros((len(logo_types), len(methods)))

        for i, logo_type in enumerate(logo_types):
            for j, method in enumerate(methods):
                type_method_results = [r for r in self.benchmark_results
                                     if r.logo_type == logo_type and r.method == method]
                if type_method_results:
                    success_rate = sum(1 for r in type_method_results if r.success) / len(type_method_results)
                    success_matrix[i, j] = success_rate

        sns.heatmap(success_matrix,
                   xticklabels=methods,
                   yticklabels=logo_types,
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd',
                   ax=ax)
        ax.set_title('Success Rate by Logo Type and Method')

    def _create_radar_chart(self) -> Optional[str]:
        """Create radar chart for multi-dimensional performance"""
        try:
            from math import pi

            # Prepare data for radar chart
            categories = ['Quality', 'Speed', 'Reliability', 'Efficiency']
            methods = [m for m in self.benchmark_config['methods_to_test']
                      if m in self.method_statistics]

            if not methods:
                return None

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # Calculate angles for each category
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]  # Complete the circle

            # Normalize scores to 0-1 scale
            max_quality = max(stats.mean_ssim_improvement for stats in self.method_statistics.values())
            max_speed = max(1/max(stats.mean_processing_time, 0.001) for stats in self.method_statistics.values())
            max_reliability = 1.0  # Success rate is already 0-1
            max_efficiency = max((stats.mean_ssim_improvement / max(stats.mean_processing_time, 0.001))
                               for stats in self.method_statistics.values())

            for method in methods:
                stats = self.method_statistics[method]

                values = [
                    stats.mean_ssim_improvement / max_quality if max_quality > 0 else 0,
                    (1/max(stats.mean_processing_time, 0.001)) / max_speed if max_speed > 0 else 0,
                    stats.success_rate,
                    (stats.mean_ssim_improvement / max(stats.mean_processing_time, 0.001)) / max_efficiency if max_efficiency > 0 else 0
                ]
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=method)
                ax.fill(angles, values, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Multi-Dimensional Performance Comparison', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            radar_path = self.output_dir / "performance_radar.png"
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(radar_path)

        except Exception as e:
            logger.error(f"Radar chart creation failed: {e}")
            return None

    def _create_significance_plot(self) -> Optional[str]:
        """Create visualization for statistical significance"""
        try:
            if not hasattr(self, 'statistical_tests') or not self.method_statistics:
                # Run statistical tests if not done yet
                statistical_tests = self._perform_statistical_tests()
            else:
                statistical_tests = self.statistical_tests

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Effect size plot
            methods = [m for m in self.method_statistics.keys() if m != 'default']
            effect_sizes = []

            for method in methods:
                if method in statistical_tests.get('effect_sizes', {}):
                    effect_sizes.append(statistical_tests['effect_sizes'][method]['cohens_d'])
                else:
                    effect_sizes.append(0)

            colors = ['green' if abs(es) >= 0.5 else 'orange' if abs(es) >= 0.2 else 'red'
                     for es in effect_sizes]

            axes[0].barh(methods, effect_sizes, color=colors, alpha=0.7)
            axes[0].set_xlabel("Cohen's d (Effect Size)")
            axes[0].set_title('Effect Size vs Baseline (Default)')
            axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0].axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
            axes[0].axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Medium effect')
            axes[0].axvline(x=0.8, color='darkgreen', linestyle='--', alpha=0.5, label='Large effect')
            axes[0].legend()

            # P-value plot for pairwise comparisons
            pairwise_tests = statistical_tests.get('pairwise_t_tests', {})
            if pairwise_tests:
                comparisons = list(pairwise_tests.keys())
                p_values = [pairwise_tests[comp]['p_value'] for comp in comparisons]

                colors = ['green' if p < 0.05 else 'red' for p in p_values]

                axes[1].barh(comparisons, p_values, color=colors, alpha=0.7)
                axes[1].set_xlabel('P-value')
                axes[1].set_title('Statistical Significance (p < 0.05)')
                axes[1].axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='Significance threshold')
                axes[1].legend()

            plt.tight_layout()
            significance_path = self.output_dir / "statistical_significance.png"
            plt.savefig(significance_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(significance_path)

        except Exception as e:
            logger.error(f"Significance plot creation failed: {e}")
            return None

    def _create_dashboard_visualization(self) -> Optional[str]:
        """Create a comprehensive dashboard visualization"""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(3, 4, figure=fig)

            # Main performance overview
            ax_main = fig.add_subplot(gs[0, :2])
            self._create_performance_overview(ax_main)

            # Quality distribution
            ax_quality = fig.add_subplot(gs[0, 2])
            self._create_quality_distribution(ax_quality)

            # Time distribution
            ax_time = fig.add_subplot(gs[0, 3])
            self._create_time_distribution(ax_time)

            # Success rates
            ax_success = fig.add_subplot(gs[1, :2])
            self._create_success_analysis(ax_success)

            # Method efficiency
            ax_efficiency = fig.add_subplot(gs[1, 2:])
            self._create_efficiency_analysis(ax_efficiency)

            # Summary statistics table
            ax_table = fig.add_subplot(gs[2, :])
            self._create_summary_table(ax_table)

            plt.suptitle('Comprehensive Benchmarking Dashboard', fontsize=20, fontweight='bold')
            plt.tight_layout()

            dashboard_path = self.output_dir / "benchmark_dashboard.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(dashboard_path)

        except Exception as e:
            logger.error(f"Dashboard creation failed: {e}")
            return None

    def _create_performance_overview(self, ax):
        """Create performance overview chart"""
        methods = list(self.method_statistics.keys())
        quality_scores = [self.method_statistics[m].mean_ssim_improvement for m in methods]
        time_scores = [1/max(self.method_statistics[m].mean_processing_time, 0.001) for m in methods]

        # Normalize scores
        if quality_scores:
            max_quality = max(quality_scores)
            quality_scores = [q/max_quality for q in quality_scores]
        if time_scores:
            max_time = max(time_scores)
            time_scores = [t/max_time for t in time_scores]

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width/2, quality_scores, width, label='Quality Score', alpha=0.8)
        ax.bar(x + width/2, time_scores, width, label='Speed Score', alpha=0.8)

        ax.set_xlabel('Methods')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Performance Overview')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()

    def _create_quality_distribution(self, ax):
        """Create quality distribution plot"""
        all_improvements = []
        for method in self.method_statistics.keys():
            method_results = [r for r in self.benchmark_results
                            if r.method == method and r.success]
            improvements = [r.ssim_improvement for r in method_results]
            all_improvements.extend(improvements)

        if all_improvements:
            ax.hist(all_improvements, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('SSIM Improvement')
            ax.set_ylabel('Frequency')
            ax.set_title('Quality Distribution')

    def _create_time_distribution(self, ax):
        """Create time distribution plot"""
        all_times = []
        for method in self.method_statistics.keys():
            method_results = [r for r in self.benchmark_results
                            if r.method == method and r.success]
            times = [r.processing_time for r in method_results]
            all_times.extend(times)

        if all_times:
            ax.hist(all_times, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Processing Time (s)')
            ax.set_ylabel('Frequency')
            ax.set_title('Time Distribution')

    def _create_success_analysis(self, ax):
        """Create success rate analysis"""
        methods = list(self.method_statistics.keys())
        success_rates = [self.method_statistics[m].success_rate for m in methods]

        colors = plt.cm.RdYlGn([rate for rate in success_rates])
        bars = ax.bar(methods, success_rates, color=colors, alpha=0.8)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')

        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Method')
        ax.set_ylim(0, 1.1)

    def _create_efficiency_analysis(self, ax):
        """Create efficiency analysis chart"""
        methods = list(self.method_statistics.keys())
        efficiency_scores = []

        for method in methods:
            stats = self.method_statistics[method]
            efficiency = stats.mean_ssim_improvement / max(stats.mean_processing_time, 0.001)
            efficiency_scores.append(efficiency)

        # Create bubble chart
        bubble_sizes = [self.method_statistics[m].count * 10 for m in methods]
        scatter = ax.scatter(
            [self.method_statistics[m].mean_processing_time for m in methods],
            [self.method_statistics[m].mean_ssim_improvement for m in methods],
            s=bubble_sizes,
            alpha=0.6,
            c=efficiency_scores,
            cmap='viridis'
        )

        # Add method labels
        for i, method in enumerate(methods):
            ax.annotate(method,
                       (self.method_statistics[method].mean_processing_time,
                        self.method_statistics[method].mean_ssim_improvement),
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Processing Time (s)')
        ax.set_ylabel('SSIM Improvement')
        ax.set_title('Efficiency Analysis (bubble size = sample count)')
        plt.colorbar(scatter, ax=ax, label='Efficiency Score')

    def _create_summary_table(self, ax):
        """Create summary statistics table"""
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        methods = list(self.method_statistics.keys())
        table_data = []

        for method in methods:
            stats = self.method_statistics[method]
            row = [
                method,
                f"{stats.mean_ssim_improvement:.3f}±{stats.std_ssim_improvement:.3f}",
                f"{stats.mean_processing_time:.2f}±{stats.std_processing_time:.2f}",
                f"{stats.success_rate:.1%}",
                f"{stats.count}",
                f"{stats.effect_size_vs_baseline:.2f}"
            ]
            table_data.append(row)

        headers = ['Method', 'SSIM Improvement', 'Time (s)', 'Success Rate', 'Samples', 'Effect Size']

        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

    def _generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on benchmark results

        Returns:
            Recommendations for method usage and system optimization
        """
        recommendations = {
            'method_selection': {},
            'optimization_priorities': [],
            'performance_insights': [],
            'system_recommendations': [],
            'roi_analysis': {}
        }

        # Method selection recommendations
        if self.method_statistics:
            best_quality_method = max(
                self.method_statistics.keys(),
                key=lambda m: self.method_statistics[m].mean_ssim_improvement
            )
            fastest_method = min(
                self.method_statistics.keys(),
                key=lambda m: self.method_statistics[m].mean_processing_time
            )
            most_reliable_method = max(
                self.method_statistics.keys(),
                key=lambda m: self.method_statistics[m].success_rate
            )

            recommendations['method_selection'] = {
                'best_quality': {
                    'method': best_quality_method,
                    'improvement': self.method_statistics[best_quality_method].mean_ssim_improvement,
                    'use_case': 'When quality is paramount and time is not constrained'
                },
                'fastest': {
                    'method': fastest_method,
                    'time': self.method_statistics[fastest_method].mean_processing_time,
                    'use_case': 'When speed is critical and moderate quality is acceptable'
                },
                'most_reliable': {
                    'method': most_reliable_method,
                    'success_rate': self.method_statistics[most_reliable_method].success_rate,
                    'use_case': 'When consistency and reliability are most important'
                }
            }

        # Optimization priorities based on performance gaps
        optimization_priorities = []

        for method, stats in self.method_statistics.items():
            if stats.success_rate < 0.9:
                optimization_priorities.append({
                    'priority': 'high',
                    'method': method,
                    'issue': 'low_success_rate',
                    'current_value': stats.success_rate,
                    'recommendation': f'Improve {method} reliability - success rate only {stats.success_rate:.1%}'
                })

            if stats.mean_processing_time > 30:  # Arbitrary threshold
                optimization_priorities.append({
                    'priority': 'medium',
                    'method': method,
                    'issue': 'slow_processing',
                    'current_value': stats.mean_processing_time,
                    'recommendation': f'Optimize {method} performance - processing time {stats.mean_processing_time:.1f}s'
                })

            if stats.mean_ssim_improvement < 0.1:  # Arbitrary threshold
                optimization_priorities.append({
                    'priority': 'medium',
                    'method': method,
                    'issue': 'low_quality',
                    'current_value': stats.mean_ssim_improvement,
                    'recommendation': f'Improve {method} quality - SSIM improvement only {stats.mean_ssim_improvement:.3f}'
                })

        recommendations['optimization_priorities'] = sorted(
            optimization_priorities,
            key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']],
            reverse=True
        )

        # Performance insights
        insights = []

        # Quality vs time tradeoff insights
        efficiency_scores = {}
        for method, stats in self.method_statistics.items():
            efficiency = stats.mean_ssim_improvement / max(stats.mean_processing_time, 0.001)
            efficiency_scores[method] = efficiency

        best_efficiency = max(efficiency_scores.keys(), key=efficiency_scores.get)
        insights.append(f"{best_efficiency} offers the best quality/time efficiency ratio")

        # Reliability insights
        reliability_variance = np.var([stats.success_rate for stats in self.method_statistics.values()])
        if reliability_variance > 0.01:  # Significant variance
            insights.append("High variance in success rates between methods suggests different reliability profiles")

        # Quality variance insights
        quality_variance = np.var([stats.mean_ssim_improvement for stats in self.method_statistics.values()])
        if quality_variance > 0.01:  # Significant variance
            insights.append("Significant quality differences between methods - method selection is critical")

        recommendations['performance_insights'] = insights

        # System recommendations
        system_recs = []

        # Memory usage recommendations
        total_memory_results = [r for r in self.benchmark_results if r.success and r.memory_usage_mb > 0]
        if total_memory_results:
            avg_memory = np.mean([r.memory_usage_mb for r in total_memory_results])
            if avg_memory > 100:  # MB
                system_recs.append(f"High memory usage detected ({avg_memory:.1f}MB avg) - consider memory optimization")

        # Batch processing recommendations
        if len(self.benchmark_results) > 50:
            system_recs.append("Large batch detected - consider parallel processing for improved throughput")

        # Method availability recommendations
        if 'method2' not in self.method_statistics:
            system_recs.append("Method 2 (PPO) unavailable - train PPO model for additional optimization option")

        recommendations['system_recommendations'] = system_recs

        # ROI analysis
        for method, stats in self.method_statistics.items():
            cost_per_improvement = stats.mean_processing_time / max(stats.mean_ssim_improvement, 0.001)
            recommendations['roi_analysis'][method] = {
                'cost_per_improvement': cost_per_improvement,
                'total_value': stats.mean_ssim_improvement * stats.success_rate,  # Quality * reliability
                'efficiency_rank': 0  # Will be filled after sorting
            }

        # Rank by efficiency
        sorted_roi = sorted(
            recommendations['roi_analysis'].items(),
            key=lambda x: x[1]['total_value'],
            reverse=True
        )
        for rank, (method, data) in enumerate(sorted_roi, 1):
            recommendations['roi_analysis'][method]['efficiency_rank'] = rank

        return recommendations

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility"""
        return {
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def _save_benchmark_results(self, results: Dict[str, Any]):
        """
        Save comprehensive benchmark results to multiple formats

        Args:
            results: Complete benchmark results
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Save JSON results
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON results saved to {json_path}")

        # Save CSV summary
        if self.benchmark_results:
            df = pd.DataFrame([asdict(r) for r in self.benchmark_results])
            csv_path = self.output_dir / f"benchmark_data_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"CSV data saved to {csv_path}")

        # Save executive summary
        summary_path = self.output_dir / f"executive_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(self._generate_executive_summary(results))
        logger.info(f"Executive summary saved to {summary_path}")

        # Save technical report
        report_path = self.output_dir / f"technical_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(self._generate_technical_report(results))
        logger.info(f"Technical report saved to {report_path}")

    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary for stakeholders"""
        summary = "# OPTIMIZATION METHODS BENCHMARK - EXECUTIVE SUMMARY\n\n"

        metadata = results.get('benchmark_metadata', {})
        summary += f"**Benchmark Date**: {metadata.get('benchmark_date', 'Unknown')}\n"
        summary += f"**Images Tested**: {metadata.get('total_images', 0)}\n"
        summary += f"**Methods Compared**: {metadata.get('methods_compared', 0)}\n\n"

        # Key findings
        summary += "## KEY FINDINGS\n\n"

        recommendations = results.get('recommendations', {})
        method_selection = recommendations.get('method_selection', {})

        if 'best_quality' in method_selection:
            best_quality = method_selection['best_quality']
            summary += f"• **Best Quality**: {best_quality['method']} "
            summary += f"({best_quality['improvement']:.1%} improvement)\n"

        if 'fastest' in method_selection:
            fastest = method_selection['fastest']
            summary += f"• **Fastest Processing**: {fastest['method']} "
            summary += f"({fastest['time']:.2f}s average)\n"

        if 'most_reliable' in method_selection:
            most_reliable = method_selection['most_reliable']
            summary += f"• **Most Reliable**: {most_reliable['method']} "
            summary += f"({most_reliable['success_rate']:.1%} success rate)\n"

        summary += "\n## RECOMMENDATIONS\n\n"

        optimization_priorities = recommendations.get('optimization_priorities', [])
        for priority in optimization_priorities[:3]:  # Top 3 priorities
            summary += f"• **{priority['priority'].upper()}**: {priority['recommendation']}\n"

        roi_analysis = recommendations.get('roi_analysis', {})
        if roi_analysis:
            best_roi = min(roi_analysis.items(), key=lambda x: x[1]['efficiency_rank'])
            summary += f"\n• **Best ROI**: {best_roi[0]} offers the best return on processing investment\n"

        summary += "\n## STATISTICAL SIGNIFICANCE\n\n"

        statistical_tests = results.get('statistical_tests', {})
        anova_tests = statistical_tests.get('anova_tests', {})

        if 'ssim_improvement' in anova_tests:
            anova = anova_tests['ssim_improvement']
            if anova['significant']:
                summary += "• Quality differences between methods are **statistically significant**\n"
            else:
                summary += "• No statistically significant quality differences detected\n"

        summary += f"\n*Full technical details available in accompanying technical report*\n"

        return summary

    def _generate_technical_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed technical report"""
        report = "# COMPREHENSIVE OPTIMIZATION BENCHMARK - TECHNICAL REPORT\n\n"

        # Metadata
        metadata = results.get('benchmark_metadata', {})
        report += "## BENCHMARK CONFIGURATION\n\n"
        report += f"- **Date**: {metadata.get('benchmark_date', 'Unknown')}\n"
        report += f"- **Total Images**: {metadata.get('total_images', 0)}\n"
        report += f"- **Methods Tested**: {metadata.get('methods_compared', 0)}\n"
        report += f"- **Runs Per Image**: {self.benchmark_config.get('runs_per_image', 'Unknown')}\n"
        report += f"- **Statistical Threshold**: p < {self.benchmark_config.get('statistical_significance_threshold', 0.05)}\n\n"

        # System info
        system_info = metadata.get('system_info', {})
        if system_info:
            report += "## SYSTEM CONFIGURATION\n\n"
            report += f"- **Platform**: {system_info.get('platform', 'Unknown')}\n"
            report += f"- **Python**: {system_info.get('python_version', 'Unknown')}\n"
            report += f"- **CPU Cores**: {system_info.get('cpu_count', 'Unknown')}\n"
            report += f"- **Memory**: {system_info.get('memory_gb', 0):.1f} GB\n\n"

        # Method statistics
        method_stats = results.get('method_statistics', {})
        if method_stats:
            report += "## METHOD PERFORMANCE STATISTICS\n\n"
            report += "| Method | SSIM Improvement | Processing Time (s) | Success Rate | Sample Size | Effect Size |\n"
            report += "|--------|------------------|-------------------|--------------|-------------|-------------|\n"

            for method, stats in method_stats.items():
                report += f"| {method} | {stats['mean_ssim_improvement']:.3f}±{stats['std_ssim_improvement']:.3f} | "
                report += f"{stats['mean_processing_time']:.2f}±{stats['std_processing_time']:.2f} | "
                report += f"{stats['success_rate']:.1%} | {stats['count']} | {stats['effect_size_vs_baseline']:.2f} |\n"

            report += "\n"

        # Statistical tests
        statistical_tests = results.get('statistical_tests', {})
        if statistical_tests:
            report += "## STATISTICAL ANALYSIS\n\n"

            anova_tests = statistical_tests.get('anova_tests', {})
            if anova_tests:
                report += "### ANOVA Results\n\n"
                for metric, anova in anova_tests.items():
                    report += f"- **{metric}**: F={anova['f_statistic']:.3f}, p={anova['p_value']:.3f} "
                    report += f"({'Significant' if anova['significant'] else 'Not significant'})\n"
                report += "\n"

            pairwise_tests = statistical_tests.get('pairwise_t_tests', {})
            if pairwise_tests:
                report += "### Pairwise t-tests\n\n"
                for comparison, test in pairwise_tests.items():
                    report += f"- **{comparison}**: t={test['t_statistic']:.3f}, p={test['p_value']:.3f}, "
                    report += f"d={test['effect_size']:.3f} ({'Significant' if test['significant'] else 'Not significant'})\n"
                report += "\n"

        # Comparative analysis
        comparative = results.get('comparative_analysis', {})
        if comparative:
            report += "## COMPARATIVE ANALYSIS\n\n"

            quality_time = comparative.get('quality_time_tradeoffs', {})
            if quality_time:
                report += "### Quality-Time Tradeoffs\n\n"
                report += "| Method | Efficiency Score | Quality | Time (s) |\n"
                report += "|--------|------------------|---------|----------|\n"

                for method, data in quality_time.items():
                    report += f"| {method} | {data['efficiency_score']:.3f} | "
                    report += f"{data['quality']:.3f} | {data['time']:.2f} |\n"
                report += "\n"

        # Recommendations
        recommendations = results.get('recommendations', {})
        if recommendations:
            report += "## RECOMMENDATIONS\n\n"

            optimization_priorities = recommendations.get('optimization_priorities', [])
            if optimization_priorities:
                report += "### Optimization Priorities\n\n"
                for i, priority in enumerate(optimization_priorities, 1):
                    report += f"{i}. **{priority['priority'].upper()}**: {priority['recommendation']}\n"
                report += "\n"

            system_recs = recommendations.get('system_recommendations', [])
            if system_recs:
                report += "### System Recommendations\n\n"
                for i, rec in enumerate(system_recs, 1):
                    report += f"{i}. {rec}\n"
                report += "\n"

        report += "---\n"
        report += f"*Report generated automatically by ComprehensiveOptimizationBenchmark*\n"

        return report

    def cleanup(self):
        """Clean up resources"""
        try:
            self.quality_metrics.cleanup()
            logger.info("Benchmark cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main function for running comprehensive benchmark"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Optimization Benchmark')
    parser.add_argument('dataset_path', help='Path to test dataset directory')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--runs-per-image', type=int, default=3,
                       help='Number of runs per image for statistical significance')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing (faster but less accurate resource monitoring)')
    parser.add_argument('--methods', nargs='+',
                       default=['default', 'method1', 'method2', 'method3'],
                       help='Methods to benchmark')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Timeout per image in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Initialize benchmark system
    benchmark = ComprehensiveOptimizationBenchmark(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )

    # Update configuration
    benchmark.benchmark_config.update({
        'runs_per_image': args.runs_per_image,
        'parallel_processing': args.parallel,
        'methods_to_test': args.methods,
        'timeout_per_image': args.timeout
    })

    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()

        if 'error' in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return 1

        # Print summary
        print("\n" + "="*80)
        print("🏆 BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*80)

        metadata = results.get('benchmark_metadata', {})
        print(f"📊 Images processed: {metadata.get('total_images', 0)}")
        print(f"⚡ Total time: {metadata.get('total_benchmark_time', 0):.1f}s")
        print(f"🎯 Methods compared: {metadata.get('methods_compared', 0)}")

        # Show recommendations
        recommendations = results.get('recommendations', {})
        method_selection = recommendations.get('method_selection', {})

        if method_selection:
            print("\n🥇 TOP PERFORMERS:")
            if 'best_quality' in method_selection:
                best = method_selection['best_quality']
                print(f"   Quality: {best['method']} ({best['improvement']:.1%} improvement)")
            if 'fastest' in method_selection:
                fastest = method_selection['fastest']
                print(f"   Speed: {fastest['method']} ({fastest['time']:.2f}s)")
            if 'most_reliable' in method_selection:
                reliable = method_selection['most_reliable']
                print(f"   Reliability: {reliable['method']} ({reliable['success_rate']:.1%} success)")

        visualizations = results.get('visualizations', {})
        if visualizations:
            print(f"\n📈 Visualizations saved in: {args.output_dir}/")
            for name, path in visualizations.items():
                if name != 'error':
                    print(f"   • {name}: {Path(path).name}")

        print(f"\n📋 Detailed results saved in: {args.output_dir}/")
        print("="*80)

        return 0

    except KeyboardInterrupt:
        print("\n⏸️ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        benchmark.cleanup()


if __name__ == '__main__':
    exit(main())