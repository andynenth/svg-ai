"""Validation Pipeline for Method 1 Parameter Optimization"""
import logging
import time
import json
import statistics
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

from .feature_mapping import FeatureMappingOptimizer
from .quality_metrics import OptimizationQualityMetrics
from .parameter_bounds import VTracerParameterBounds
from ..feature_extraction import ImageFeatureExtractor


@dataclass
class ValidationResult:
    """Structure for validation results"""
    image_path: str
    features: Dict[str, float]
    optimized_params: Dict[str, Any]
    quality_improvement: float
    processing_time: float
    success: bool
    error_message: str = ""
    logo_type: str = ""
    confidence: float = 0.0
    default_ssim: float = 0.0
    optimized_ssim: float = 0.0
    file_size_improvement: float = 0.0
    conversion_time: float = 0.0


class Method1ValidationPipeline:
    """Systematic validation of Method 1 optimization"""

    def __init__(self, enable_quality_measurement: bool = True):
        """Initialize validation pipeline

        Args:
            enable_quality_measurement: If False, skips VTracer quality tests (faster)
        """
        self.optimizer = FeatureMappingOptimizer()
        self.feature_extractor = ImageFeatureExtractor()
        self.quality_metrics = OptimizationQualityMetrics() if enable_quality_measurement else None
        self.bounds = VTracerParameterBounds()
        self.results: List[ValidationResult] = []
        self.logger = logging.getLogger(__name__)
        self.enable_quality_measurement = enable_quality_measurement

    def validate_dataset(self, dataset_path: str, max_images_per_category: int = 10) -> Dict[str, Any]:
        """Run validation on complete test dataset

        Args:
            dataset_path: Path to test dataset with categorized subdirectories
            max_images_per_category: Limit images per category for speed

        Returns:
            Comprehensive validation results
        """
        dataset_path = Path(dataset_path)
        self.results = []

        self.logger.info(f"Starting validation pipeline on dataset: {dataset_path}")

        # Define expected categories
        categories = ["simple", "text", "gradient", "complex"]

        # Collect images by category
        images_by_category = {}
        total_images = 0

        for category in categories:
            category_path = dataset_path / category
            if category_path.exists():
                # Get image files
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    image_files.extend(category_path.glob(ext))

                # Limit images per category
                if len(image_files) > max_images_per_category:
                    image_files = image_files[:max_images_per_category]

                images_by_category[category] = image_files
                total_images += len(image_files)
                self.logger.info(f"Found {len(image_files)} images in {category} category")
            else:
                images_by_category[category] = []
                self.logger.warning(f"Category directory not found: {category_path}")

        if total_images == 0:
            return {
                "error": "No images found in dataset",
                "dataset_path": str(dataset_path),
                "timestamp": datetime.now().isoformat()
            }

        # Run validation on each image
        processed = 0
        for category, image_files in images_by_category.items():
            for image_path in image_files:
                self.logger.info(f"Validating image {processed + 1}/{total_images}: {image_path.name}")

                try:
                    result = self._validate_single_image(str(image_path), category)
                    self.results.append(result)

                    if result.success:
                        self.logger.info(f"✅ Success: {result.quality_improvement:.1f}% improvement")
                    else:
                        self.logger.warning(f"❌ Failed: {result.error_message}")

                except Exception as e:
                    self.logger.error(f"Validation error for {image_path}: {e}")
                    # Add failed result
                    self.results.append(ValidationResult(
                        image_path=str(image_path),
                        features={},
                        optimized_params={},
                        quality_improvement=0.0,
                        processing_time=0.0,
                        success=False,
                        error_message=str(e),
                        logo_type=category
                    ))

                processed += 1

        # Generate comprehensive analysis
        analysis = self._generate_statistical_analysis()

        self.logger.info(f"Validation complete: {len(self.results)} images processed")
        self.logger.info(f"Overall success rate: {analysis['validation_summary']['overall_success_rate']:.1f}%")

        return analysis

    def _validate_single_image(self, image_path: str, logo_type: str) -> ValidationResult:
        """Validate Method 1 optimization on a single image"""
        start_time = time.time()

        try:
            # Extract features
            features = self.feature_extractor.extract_features(image_path)

            if not features:
                return ValidationResult(
                    image_path=image_path,
                    features={},
                    optimized_params={},
                    quality_improvement=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Feature extraction failed",
                    logo_type=logo_type
                )

            # Run optimization
            optimization_result = self.optimizer.optimize(features)

            if not optimization_result or 'parameters' not in optimization_result:
                return ValidationResult(
                    image_path=image_path,
                    features=features,
                    optimized_params={},
                    quality_improvement=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Parameter optimization failed",
                    logo_type=logo_type
                )

            optimized_params = optimization_result['parameters']
            confidence = optimization_result.get('confidence', 0.0)

            # Validate parameters are within bounds
            validation_result = self.bounds.validate_parameters(optimized_params)
            if not validation_result['valid']:
                return ValidationResult(
                    image_path=image_path,
                    features=features,
                    optimized_params=optimized_params,
                    quality_improvement=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Invalid parameters: {validation_result['errors']}",
                    logo_type=logo_type,
                    confidence=confidence
                )

            # Measure quality improvement if enabled
            quality_improvement = 0.0
            default_ssim = 0.0
            optimized_ssim = 0.0
            file_size_improvement = 0.0
            conversion_time = 0.0

            if self.enable_quality_measurement and self.quality_metrics:
                try:
                    default_params = self.bounds.get_default_parameters()
                    quality_result = self.quality_metrics.measure_improvement(
                        image_path, default_params, optimized_params, runs=1
                    )

                    if quality_result and 'improvements' in quality_result:
                        quality_improvement = quality_result['improvements'].get('ssim_improvement', 0.0)
                        file_size_improvement = quality_result['improvements'].get('file_size_improvement', 0.0)

                        default_ssim = quality_result.get('default_metrics', {}).get('ssim', 0.0)
                        optimized_ssim = quality_result.get('optimized_metrics', {}).get('ssim', 0.0)
                        conversion_time = quality_result.get('optimized_metrics', {}).get('conversion_time', 0.0)

                except Exception as e:
                    self.logger.warning(f"Quality measurement failed for {image_path}: {e}")
                    # Don't fail the validation, just note the issue
                    quality_improvement = 0.0

            processing_time = time.time() - start_time

            # Consider validation successful if parameters are valid
            # Quality improvement > 0 is a bonus but not required for success
            success = True
            error_message = ""

            return ValidationResult(
                image_path=image_path,
                features=features,
                optimized_params=optimized_params,
                quality_improvement=quality_improvement,
                processing_time=processing_time,
                success=success,
                error_message=error_message,
                logo_type=logo_type,
                confidence=confidence,
                default_ssim=default_ssim,
                optimized_ssim=optimized_ssim,
                file_size_improvement=file_size_improvement,
                conversion_time=conversion_time
            )

        except Exception as e:
            return ValidationResult(
                image_path=image_path,
                features={},
                optimized_params={},
                quality_improvement=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
                logo_type=logo_type
            )

    def _generate_statistical_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis of validation results"""
        if not self.results:
            return {
                "error": "No validation results to analyze",
                "timestamp": datetime.now().isoformat()
            }

        total_results = len(self.results)
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]

        # Calculate overall metrics
        overall_success_rate = len(successful_results) / total_results * 100

        # Quality improvements (only for successful results with quality measurement)
        quality_improvements = [r.quality_improvement for r in successful_results if r.quality_improvement != 0.0]

        # Processing times
        processing_times = [r.processing_time for r in self.results]

        # Analysis by category
        categories = ["simple", "text", "gradient", "complex"]
        category_analysis = {}

        for category in categories:
            category_results = [r for r in self.results if r.logo_type == category]

            if category_results:
                category_successful = [r for r in category_results if r.success]
                success_rate = len(category_successful) / len(category_results) * 100

                # Quality improvements for this category
                category_quality = [r.quality_improvement for r in category_successful if r.quality_improvement != 0.0]

                category_analysis[category] = {
                    "total_images": len(category_results),
                    "successful": len(category_successful),
                    "success_rate": success_rate,
                    "quality_improvements": {
                        "count": len(category_quality),
                        "mean": statistics.mean(category_quality) if category_quality else 0.0,
                        "median": statistics.median(category_quality) if category_quality else 0.0,
                        "std_dev": statistics.stdev(category_quality) if len(category_quality) > 1 else 0.0,
                        "min": min(category_quality) if category_quality else 0.0,
                        "max": max(category_quality) if category_quality else 0.0
                    },
                    "processing_times": {
                        "mean": statistics.mean([r.processing_time for r in category_results]),
                        "median": statistics.median([r.processing_time for r in category_results]),
                        "max": max([r.processing_time for r in category_results])
                    }
                }
            else:
                category_analysis[category] = {
                    "total_images": 0,
                    "successful": 0,
                    "success_rate": 0.0,
                    "quality_improvements": {"count": 0, "mean": 0.0},
                    "processing_times": {"mean": 0.0}
                }

        # Failure analysis
        failure_analysis = {}
        if failed_results:
            error_types = {}
            for result in failed_results:
                error_type = result.error_message.split(':')[0] if result.error_message else "Unknown error"
                error_types[error_type] = error_types.get(error_type, 0) + 1

            failure_analysis = {
                "total_failures": len(failed_results),
                "failure_rate": len(failed_results) / total_results * 100,
                "error_types": error_types,
                "failed_images": [r.image_path for r in failed_results]
            }

        # Feature correlation analysis
        feature_correlations = self._analyze_feature_correlations()

        # Generate summary
        analysis = {
            "validation_summary": {
                "total_images": total_results,
                "successful_optimizations": len(successful_results),
                "failed_optimizations": len(failed_results),
                "overall_success_rate": overall_success_rate,
                "quality_measurement_enabled": self.enable_quality_measurement
            },
            "performance_metrics": {
                "processing_time": {
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "max": max(processing_times),
                    "min": min(processing_times)
                }
            },
            "quality_analysis": {
                "improvements_measured": len(quality_improvements),
                "mean_improvement": statistics.mean(quality_improvements) if quality_improvements else 0.0,
                "median_improvement": statistics.median(quality_improvements) if quality_improvements else 0.0,
                "std_dev": statistics.stdev(quality_improvements) if len(quality_improvements) > 1 else 0.0,
                "min_improvement": min(quality_improvements) if quality_improvements else 0.0,
                "max_improvement": max(quality_improvements) if quality_improvements else 0.0,
                "improvements_above_15_percent": len([q for q in quality_improvements if q >= 15.0])
            },
            "category_analysis": category_analysis,
            "failure_analysis": failure_analysis,
            "feature_correlations": feature_correlations,
            "success_criteria_check": self._check_success_criteria(category_analysis, overall_success_rate),
            "timestamp": datetime.now().isoformat(),
            "dataset_path": self.results[0].image_path.split('/')[:-2] if self.results else "unknown"
        }

        return analysis

    def _analyze_feature_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between features and optimization success"""
        successful_results = [r for r in self.results if r.success and r.features]

        if len(successful_results) < 3:
            return {"error": "Insufficient data for correlation analysis"}

        # Extract feature values
        feature_names = list(successful_results[0].features.keys())
        correlations = {}

        for feature_name in feature_names:
            feature_values = [r.features.get(feature_name, 0.0) for r in successful_results]
            quality_improvements = [r.quality_improvement for r in successful_results]

            # Calculate correlation if we have quality data
            if any(q != 0.0 for q in quality_improvements) and len(feature_values) > 2:
                try:
                    correlation = np.corrcoef(feature_values, quality_improvements)[0, 1]
                    if not np.isnan(correlation):
                        correlations[feature_name] = {
                            "correlation": correlation,
                            "sample_size": len(feature_values),
                            "feature_range": {
                                "min": min(feature_values),
                                "max": max(feature_values),
                                "mean": statistics.mean(feature_values)
                            }
                        }
                except:
                    pass

        return correlations

    def _check_success_criteria(self, category_analysis: Dict, overall_success_rate: float) -> Dict[str, Any]:
        """Check if validation meets success criteria"""
        criteria = {
            "overall_success_rate_80_percent": overall_success_rate >= 80.0,
            "simple_success_rate_95_percent": category_analysis.get("simple", {}).get("success_rate", 0) >= 95.0,
            "text_success_rate_90_percent": category_analysis.get("text", {}).get("success_rate", 0) >= 90.0,
            "gradient_success_rate_85_percent": category_analysis.get("gradient", {}).get("success_rate", 0) >= 85.0,
            "complex_success_rate_80_percent": category_analysis.get("complex", {}).get("success_rate", 0) >= 80.0
        }

        total_criteria = len(criteria)
        met_criteria = sum(criteria.values())

        return {
            "criteria_met": criteria,
            "total_criteria": total_criteria,
            "met_criteria": met_criteria,
            "success_percentage": met_criteria / total_criteria * 100,
            "overall_success": met_criteria >= (total_criteria * 0.8)  # 80% of criteria must be met
        }

    def export_results(self, output_dir: str = "validation_results") -> Dict[str, str]:
        """Export validation results to multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export detailed results as JSON
        json_path = output_dir / f"validation_results_{timestamp}.json"
        results_data = [asdict(result) for result in self.results]

        with open(json_path, 'w') as f:
            json.dump({
                "results": results_data,
                "analysis": self._generate_statistical_analysis()
            }, f, indent=2)

        # Export summary as CSV
        csv_path = output_dir / f"validation_summary_{timestamp}.csv"

        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path", "logo_type", "success", "quality_improvement",
                "processing_time", "confidence", "error_message"
            ])

            for result in self.results:
                writer.writerow([
                    Path(result.image_path).name,
                    result.logo_type,
                    result.success,
                    f"{result.quality_improvement:.2f}",
                    f"{result.processing_time:.3f}",
                    f"{result.confidence:.3f}",
                    result.error_message
                ])

        # Generate HTML report
        html_path = self._generate_html_report(output_dir, timestamp)

        return {
            "json": str(json_path),
            "csv": str(csv_path),
            "html": str(html_path)
        }

    def _generate_html_report(self, output_dir: Path, timestamp: str) -> Path:
        """Generate comprehensive HTML validation report"""
        html_path = output_dir / f"validation_report_{timestamp}.html"

        analysis = self._generate_statistical_analysis()

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Method 1 Validation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .success {{ color: green; font-weight: bold; }}
        .failure {{ color: red; font-weight: bold; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Method 1 Validation Report</h1>
    <p>Generated: {analysis['timestamp']}</p>

    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Overall Success Rate:</strong>
            <span class="{'success' if analysis['validation_summary']['overall_success_rate'] >= 80 else 'failure'}">
                {analysis['validation_summary']['overall_success_rate']:.1f}%
            </span>
        </div>
        <div class="metric">
            <strong>Images Processed:</strong> {analysis['validation_summary']['total_images']}
        </div>
        <div class="metric">
            <strong>Average Quality Improvement:</strong> {analysis['quality_analysis']['mean_improvement']:.1f}%
        </div>
        <div class="metric">
            <strong>Average Processing Time:</strong> {analysis['performance_metrics']['processing_time']['mean']:.3f}s
        </div>
    </div>

    <h2>Success Criteria Assessment</h2>
    <table>
        <tr><th>Criteria</th><th>Target</th><th>Actual</th><th>Status</th></tr>
"""

        # Add success criteria rows
        criteria_names = {
            "overall_success_rate_80_percent": ("Overall Success Rate", "≥80%"),
            "simple_success_rate_95_percent": ("Simple Logos", "≥95%"),
            "text_success_rate_90_percent": ("Text Logos", "≥90%"),
            "gradient_success_rate_85_percent": ("Gradient Logos", "≥85%"),
            "complex_success_rate_80_percent": ("Complex Logos", "≥80%")
        }

        success_criteria = analysis.get('success_criteria_check', {})
        criteria_met = success_criteria.get('criteria_met', {})

        for key, (name, target) in criteria_names.items():
            status = criteria_met.get(key, False)
            # Get actual value
            if key == "overall_success_rate_80_percent":
                actual = f"{analysis['validation_summary']['overall_success_rate']:.1f}%"
            else:
                category = key.split('_')[0]
                actual_rate = analysis['category_analysis'].get(category, {}).get('success_rate', 0)
                actual = f"{actual_rate:.1f}%"

            html_content += f"""
        <tr>
            <td>{name}</td>
            <td>{target}</td>
            <td>{actual}</td>
            <td class="{'success' if status else 'failure'}">{'✅ Pass' if status else '❌ Fail'}</td>
        </tr>"""

        html_content += """
    </table>

    <h2>Performance by Logo Type</h2>
    <table>
        <tr><th>Category</th><th>Images</th><th>Success Rate</th><th>Avg Quality Improvement</th><th>Avg Processing Time</th></tr>
"""

        for category, data in analysis['category_analysis'].items():
            if data['total_images'] > 0:
                html_content += f"""
        <tr>
            <td>{category.title()}</td>
            <td>{data['total_images']}</td>
            <td class="{'success' if data['success_rate'] >= 80 else 'failure'}">{data['success_rate']:.1f}%</td>
            <td>{data['quality_improvements']['mean']:.1f}%</td>
            <td>{data['processing_times']['mean']:.3f}s</td>
        </tr>"""

        html_content += """
    </table>
"""

        # Add failure analysis if there are failures
        if 'failure_analysis' in analysis and analysis['failure_analysis']:
            html_content += """
    <h2>Failure Analysis</h2>
    <div class="summary">
"""
            failure_data = analysis['failure_analysis']
            for error_type, count in failure_data.get('error_types', {}).items():
                html_content += f"<p><strong>{error_type}:</strong> {count} occurrences</p>"

            html_content += "</div>"

        html_content += """
    <h2>Quality Improvement Distribution</h2>
    <div id="qualityChart" class="chart"></div>

    <script>
        // Quality improvement histogram
        var improvements = ["""

        quality_improvements = [r.quality_improvement for r in self.results if r.success and r.quality_improvement != 0.0]
        html_content += ', '.join([str(q) for q in quality_improvements])

        html_content += f"""
        ];

        var trace = {{
            x: improvements,
            type: 'histogram',
            nbinsx: 20,
            name: 'Quality Improvements'
        }};

        var layout = {{
            title: 'Distribution of Quality Improvements',
            xaxis: {{ title: 'SSIM Improvement (%)' }},
            yaxis: {{ title: 'Count' }}
        }};

        Plotly.newPlot('qualityChart', [trace], layout);
    </script>

</body>
</html>"""

        with open(html_path, 'w') as f:
            f.write(html_content)

        return html_path

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of validation results"""
        if not self.results:
            return {"message": "No validation results available"}

        successful = len([r for r in self.results if r.success])
        total = len(self.results)

        return {
            "total_images": total,
            "successful_optimizations": successful,
            "success_rate": successful / total * 100 if total > 0 else 0,
            "quality_measurement_enabled": self.enable_quality_measurement,
            "average_processing_time": statistics.mean([r.processing_time for r in self.results]),
            "results_available": len(self.results)
        }

    def cleanup(self):
        """Cleanup validation resources"""
        if self.quality_metrics:
            self.quality_metrics.cleanup()