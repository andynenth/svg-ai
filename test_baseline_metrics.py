#!/usr/bin/env python3
"""
Baseline metrics collection for SVG conversion optimization.

This script establishes baseline performance metrics across all logo categories
to track improvements during optimization.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineMetrics:
    """Collect baseline metrics for conversion quality and performance."""

    def __init__(self, dataset_dir: str = "data/logos"):
        """
        Initialize baseline metrics collector.

        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.categories = ['simple_geometric', 'text_based', 'gradients', 'abstract', 'complex']
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'categories': {},
            'summary': {}
        }

    def test_ai_detection(self, image_path: str) -> Dict:
        """
        Test AI detection on an image.

        Args:
            image_path: Path to test image

        Returns:
            Detection results dictionary
        """
        try:
            from utils.ai_detector import create_detector

            detector = create_detector()
            logo_type, confidence, scores = detector.detect_logo_type(str(image_path))

            return {
                'detected_type': logo_type,
                'confidence': confidence,
                'scores': scores,
                'success': True
            }
        except Exception as e:
            logger.warning(f"AI detection failed for {image_path}: {e}")
            return {
                'detected_type': 'unknown',
                'confidence': 0.0,
                'scores': {},
                'success': False,
                'error': str(e)
            }

    def test_conversion(self, image_path: str) -> Dict:
        """
        Test conversion of an image.

        Args:
            image_path: Path to test image

        Returns:
            Conversion results dictionary
        """
        try:
            from converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()
            output_path = image_path.with_suffix('.baseline.svg')

            # Time the conversion
            start_time = time.time()
            svg_content = converter.convert(str(image_path))
            conversion_time = time.time() - start_time

            # Save SVG
            with open(output_path, 'w') as f:
                f.write(svg_content)

            # Get file sizes
            png_size = os.path.getsize(image_path)
            svg_size = os.path.getsize(output_path)

            return {
                'conversion_time': conversion_time,
                'png_size': png_size,
                'svg_size': svg_size,
                'size_reduction': (1 - svg_size / png_size) * 100,
                'output_path': str(output_path),
                'success': True
            }
        except Exception as e:
            logger.warning(f"Conversion failed for {image_path}: {e}")
            return {
                'conversion_time': 0,
                'png_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                'svg_size': 0,
                'size_reduction': 0,
                'success': False,
                'error': str(e)
            }

    def test_quality(self, png_path: str, svg_path: str) -> Dict:
        """
        Test quality metrics between PNG and SVG.

        Args:
            png_path: Path to original PNG
            svg_path: Path to converted SVG

        Returns:
            Quality metrics dictionary
        """
        try:
            from utils.image_loader import QualityMetricsWrapper

            wrapper = QualityMetricsWrapper()

            ssim = wrapper.calculate_ssim_from_paths(str(png_path), str(svg_path))
            mse = wrapper.calculate_mse_from_paths(str(png_path), str(svg_path))
            psnr = wrapper.calculate_psnr_from_paths(str(png_path), str(svg_path))

            return {
                'ssim': max(0, ssim),  # Ensure non-negative
                'mse': max(0, mse),
                'psnr': max(0, psnr),
                'success': ssim >= 0
            }
        except Exception as e:
            logger.warning(f"Quality metrics failed: {e}")
            return {
                'ssim': 0,
                'mse': -1,
                'psnr': 0,
                'success': False,
                'error': str(e)
            }

    def process_category(self, category: str, max_files: int = 5) -> Dict:
        """
        Process all images in a category.

        Args:
            category: Category name
            max_files: Maximum files to process per category

        Returns:
            Category results dictionary
        """
        category_path = self.dataset_dir / category
        if not category_path.exists():
            logger.warning(f"Category directory not found: {category_path}")
            return {'error': 'Directory not found'}

        # Get PNG files
        png_files = list(category_path.glob("*.png"))[:max_files]
        if not png_files:
            logger.warning(f"No PNG files found in {category_path}")
            return {'error': 'No PNG files'}

        logger.info(f"\nProcessing {category} ({len(png_files)} files)...")

        results = []
        for png_file in png_files:
            logger.info(f"  Testing {png_file.name}...")

            file_results = {
                'filename': png_file.name,
                'path': str(png_file)
            }

            # Test AI detection
            detection = self.test_ai_detection(png_file)
            file_results['detection'] = detection

            # Test conversion
            conversion = self.test_conversion(png_file)
            file_results['conversion'] = conversion

            # Test quality if conversion succeeded
            if conversion.get('success') and conversion.get('output_path'):
                quality = self.test_quality(png_file, conversion['output_path'])
                file_results['quality'] = quality
            else:
                file_results['quality'] = {'success': False}

            results.append(file_results)

        # Calculate category statistics
        stats = self.calculate_stats(results, category)

        return {
            'files': results,
            'stats': stats
        }

    def calculate_stats(self, results: List[Dict], expected_type: str) -> Dict:
        """
        Calculate statistics for a set of results.

        Args:
            results: List of file results
            expected_type: Expected detection type for this category

        Returns:
            Statistics dictionary
        """
        total = len(results)
        if total == 0:
            return {}

        # Detection stats
        detections_correct = sum(
            1 for r in results
            if r['detection']['detected_type'] == self.map_category_to_type(expected_type)
        )
        detection_accuracy = detections_correct / total * 100

        # Average confidence
        confidences = [r['detection']['confidence'] for r in results if r['detection']['success']]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Conversion stats
        conversions_success = sum(1 for r in results if r['conversion']['success'])
        conversion_rate = conversions_success / total * 100

        # Average times
        times = [r['conversion']['conversion_time'] for r in results if r['conversion']['success']]
        avg_time = sum(times) / len(times) if times else 0

        # Size reduction
        reductions = [r['conversion']['size_reduction'] for r in results if r['conversion']['success']]
        avg_reduction = sum(reductions) / len(reductions) if reductions else 0

        # Quality stats
        ssims = [r['quality']['ssim'] for r in results if r['quality'].get('success')]
        avg_ssim = sum(ssims) / len(ssims) if ssims else 0

        return {
            'total_files': total,
            'detection_accuracy': detection_accuracy,
            'avg_confidence': avg_confidence,
            'conversion_success_rate': conversion_rate,
            'avg_conversion_time': avg_time,
            'avg_size_reduction': avg_reduction,
            'avg_ssim': avg_ssim
        }

    def map_category_to_type(self, category: str) -> str:
        """Map directory category to expected detection type."""
        mapping = {
            'simple_geometric': 'simple',
            'text_based': 'text',
            'gradients': 'gradient',
            'abstract': 'complex',
            'complex': 'complex'
        }
        return mapping.get(category, 'complex')

    def run_baseline_tests(self, max_files_per_category: int = 5):
        """
        Run baseline tests on all categories.

        Args:
            max_files_per_category: Maximum files to test per category
        """
        logger.info("="*60)
        logger.info("COLLECTING BASELINE METRICS")
        logger.info("="*60)

        for category in self.categories:
            self.results['categories'][category] = self.process_category(
                category, max_files_per_category
            )

        # Calculate overall summary
        self.calculate_summary()

        # Save results
        output_path = Path("baseline_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\n✅ Baseline metrics saved to: {output_path}")
        self.print_summary()

    def calculate_summary(self):
        """Calculate overall summary statistics."""
        all_stats = []
        for category, data in self.results['categories'].items():
            if 'stats' in data:
                all_stats.append(data['stats'])

        if not all_stats:
            return

        # Calculate overall averages
        self.results['summary'] = {
            'overall_detection_accuracy': sum(s['detection_accuracy'] for s in all_stats) / len(all_stats),
            'overall_avg_confidence': sum(s['avg_confidence'] for s in all_stats) / len(all_stats),
            'overall_conversion_rate': sum(s['conversion_success_rate'] for s in all_stats) / len(all_stats),
            'overall_avg_time': sum(s['avg_conversion_time'] for s in all_stats) / len(all_stats),
            'overall_size_reduction': sum(s['avg_size_reduction'] for s in all_stats) / len(all_stats),
            'overall_avg_ssim': sum(s['avg_ssim'] for s in all_stats) / len(all_stats),
            'categories_tested': len(all_stats)
        }

    def print_summary(self):
        """Print summary of baseline metrics."""
        print("\n" + "="*60)
        print("BASELINE METRICS SUMMARY")
        print("="*60)

        if 'summary' in self.results and self.results['summary']:
            s = self.results['summary']
            print(f"\n📊 Overall Performance:")
            print(f"  Detection Accuracy: {s['overall_detection_accuracy']:.1f}%")
            print(f"  Average Confidence: {s['overall_avg_confidence']:.1%}")
            print(f"  Conversion Success: {s['overall_conversion_rate']:.1f}%")
            print(f"  Average Time: {s['overall_avg_time']:.3f}s")
            print(f"  Size Reduction: {s['overall_size_reduction']:.1f}%")
            print(f"  Average SSIM: {s['overall_avg_ssim']:.4f}")

        print("\n📁 Per Category:")
        for category, data in self.results['categories'].items():
            if 'stats' in data:
                stats = data['stats']
                print(f"\n  {category}:")
                print(f"    Detection: {stats['detection_accuracy']:.0f}% accurate")
                print(f"    Quality: {stats['avg_ssim']:.3f} SSIM")
                print(f"    Size: {stats['avg_size_reduction']:.1f}% smaller")

        print("\n" + "="*60)
        print("Use these metrics as baseline for optimization tracking")
        print("="*60)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect baseline metrics for SVG conversion")
    parser.add_argument("--dataset", default="data/logos", help="Dataset directory")
    parser.add_argument("--max-files", type=int, default=5,
                       help="Maximum files per category")
    parser.add_argument("--output", default="baseline_metrics.json",
                       help="Output JSON file")

    args = parser.parse_args()

    # Run baseline tests
    collector = BaselineMetrics(args.dataset)
    collector.run_baseline_tests(args.max_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())