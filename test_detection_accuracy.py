#!/usr/bin/env python3
"""
Test detection accuracy across all logo categories.

This script tests the AI detection accuracy on the full dataset
and generates a detailed accuracy report.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ai_detector import create_detector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionAccuracyTester:
    """Test and report detection accuracy across logo categories."""

    # Expected types for each category
    CATEGORY_MAPPING = {
        'simple_geometric': 'simple',
        'text_based': 'text',
        'gradients': 'gradient',
        'abstract': 'complex',
        'complex': 'complex'
    }

    def __init__(self, dataset_dir: str = "data/logos"):
        """Initialize the accuracy tester."""
        self.dataset_dir = Path(dataset_dir)
        self.results = {
            'by_category': {},
            'overall': {},
            'confusion_matrix': {}
        }

    def test_category(self, category: str, max_files: int = None) -> Dict:
        """
        Test detection accuracy for a category.

        Args:
            category: Category name
            max_files: Maximum number of files to test

        Returns:
            Category results dictionary
        """
        category_path = self.dataset_dir / category
        if not category_path.exists():
            logger.warning(f"Category not found: {category_path}")
            return {}

        # Get PNG files
        png_files = list(category_path.glob("*.png"))
        if max_files:
            png_files = png_files[:max_files]

        if not png_files:
            logger.warning(f"No PNG files in {category_path}")
            return {}

        expected_type = self.CATEGORY_MAPPING.get(category, 'complex')
        logger.info(f"\nTesting {category} ({len(png_files)} files, expecting '{expected_type}')...")

        # Initialize detector
        detector = create_detector()

        results = []
        correct = 0
        total_confidence = 0

        for png_file in png_files:
            # Detect type
            detected_type, confidence, scores = detector.detect_logo_type(str(png_file))

            is_correct = detected_type == expected_type
            if is_correct:
                correct += 1

            total_confidence += confidence

            results.append({
                'file': png_file.name,
                'expected': expected_type,
                'detected': detected_type,
                'correct': is_correct,
                'confidence': confidence,
                'scores': scores
            })

            # Update confusion matrix
            key = f"{expected_type}->{detected_type}"
            if key not in self.results['confusion_matrix']:
                self.results['confusion_matrix'][key] = 0
            self.results['confusion_matrix'][key] += 1

        accuracy = (correct / len(results)) * 100 if results else 0
        avg_confidence = (total_confidence / len(results)) if results else 0

        return {
            'total': len(results),
            'correct': correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'expected_type': expected_type,
            'details': results
        }

    def run_full_test(self, max_files_per_category: int = None):
        """
        Run detection test on all categories.

        Args:
            max_files_per_category: Maximum files to test per category
        """
        logger.info("="*60)
        logger.info("DETECTION ACCURACY TEST")
        logger.info("="*60)

        categories = ['simple_geometric', 'text_based', 'gradients', 'abstract', 'complex']

        total_correct = 0
        total_files = 0
        total_confidence = 0

        for category in categories:
            results = self.test_category(category, max_files_per_category)
            if results:
                self.results['by_category'][category] = results
                total_correct += results['correct']
                total_files += results['total']
                total_confidence += results['avg_confidence'] * results['total']

        # Calculate overall statistics
        if total_files > 0:
            self.results['overall'] = {
                'total_files': total_files,
                'total_correct': total_correct,
                'overall_accuracy': (total_correct / total_files) * 100,
                'overall_avg_confidence': total_confidence / total_files
            }

    def print_report(self):
        """Print a formatted accuracy report."""
        print("\n" + "="*60)
        print("DETECTION ACCURACY REPORT")
        print("="*60)

        if 'overall' in self.results and self.results['overall']:
            o = self.results['overall']
            print(f"\n📊 Overall Performance:")
            print(f"  Files Tested: {o['total_files']}")
            print(f"  Correct: {o['total_correct']}/{o['total_files']}")
            print(f"  Accuracy: {o['overall_accuracy']:.1f}%")
            print(f"  Avg Confidence: {o['overall_avg_confidence']:.1%}")

        print("\n📁 Per Category Accuracy:")
        for category, data in self.results['by_category'].items():
            print(f"\n  {category}:")
            print(f"    Expected: {data['expected_type']}")
            print(f"    Accuracy: {data['accuracy']:.1f}% ({data['correct']}/{data['total']})")
            print(f"    Confidence: {data['avg_confidence']:.1%}")

        print("\n🔄 Confusion Matrix:")
        print("  (expected->detected: count)")
        for key, count in sorted(self.results['confusion_matrix'].items()):
            print(f"    {key}: {count}")

        # Find most common misclassifications
        misclassifications = {k: v for k, v in self.results['confusion_matrix'].items()
                              if not k.startswith(k.split('->')[1])}
        if misclassifications:
            print("\n⚠️ Common Misclassifications:")
            for key, count in sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {key}: {count} times")

    def save_results(self, output_path: str = "detection_accuracy_report.json"):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test AI detection accuracy")
    parser.add_argument("--dataset", default="data/logos", help="Dataset directory")
    parser.add_argument("--max-files", type=int, help="Max files per category")
    parser.add_argument("--output", default="detection_accuracy_report.json",
                       help="Output JSON file")
    parser.add_argument("--compare-prompts", action="store_true",
                       help="Compare different prompt variations")

    args = parser.parse_args()

    # Run accuracy test
    tester = DetectionAccuracyTester(args.dataset)
    tester.run_full_test(args.max_files)
    tester.print_report()
    tester.save_results(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())