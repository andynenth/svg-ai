#!/usr/bin/env python3
"""
Test OCR-enhanced detection accuracy.
"""

import os
import sys
from pathlib import Path
import json
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.ai_detector import create_detector
from utils.ocr_detector import OCRDetector
import logging

# Disable verbose logging
logging.getLogger('easyocr').setLevel(logging.ERROR)


def test_ocr_accuracy():
    """Test OCR-enhanced detection accuracy."""
    print("="*60)
    print("OCR-ENHANCED DETECTION ACCURACY TEST")
    print("="*60)

    # Define expected types
    expected_types = {
        'simple_geometric': 'simple',
        'text_based': 'text',
        'gradient_shaded': 'gradient',
        'complex_artistic': 'complex',
        'mixed_elements': 'complex'
    }

    base_detector = create_detector()
    ocr_detector = OCRDetector()

    results = {
        'base': {'correct': 0, 'total': 0},
        'ocr': {'correct': 0, 'total': 0},
        'by_category': {}
    }

    # Process each category
    for category, expected_type in expected_types.items():
        category_dir = Path(f"data/logos/{category}")
        if not category_dir.exists():
            continue

        images = list(category_dir.glob("*.png"))[:5]  # Test first 5
        category_results = {'base': 0, 'ocr': 0, 'total': len(images)}

        print(f"\n📂 {category}:")
        for img_path in images:
            # Base detection
            base_result = base_detector.detect_logo_type(str(img_path))
            base_type = base_result[0]

            # OCR-enhanced detection
            enhanced_result = ocr_detector.classify_with_ocr(str(img_path), base_result)
            ocr_type = enhanced_result[0]

            # Check accuracy
            base_correct = base_type == expected_type
            ocr_correct = ocr_type == expected_type

            if base_correct:
                results['base']['correct'] += 1
                category_results['base'] += 1

            if ocr_correct:
                results['ocr']['correct'] += 1
                category_results['ocr'] += 1

            results['base']['total'] += 1
            results['ocr']['total'] += 1

            # Print individual result if changed
            if base_type != ocr_type:
                print(f"  {img_path.name}: {base_type} → {ocr_type} {'✅' if ocr_correct else '❌'}")

        # Category summary
        if category_results['total'] > 0:
            base_acc = category_results['base'] / category_results['total'] * 100
            ocr_acc = category_results['ocr'] / category_results['total'] * 100
            print(f"  Accuracy: Base={base_acc:.0f}%, OCR={ocr_acc:.0f}%")
            results['by_category'][category] = {
                'base_accuracy': base_acc,
                'ocr_accuracy': ocr_acc
            }

    # Overall summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    base_accuracy = results['base']['correct'] / results['base']['total'] * 100 if results['base']['total'] > 0 else 0
    ocr_accuracy = results['ocr']['correct'] / results['ocr']['total'] * 100 if results['ocr']['total'] > 0 else 0

    print(f"\n📊 Overall Accuracy:")
    print(f"  Base Detection: {base_accuracy:.1f}%")
    print(f"  OCR-Enhanced:   {ocr_accuracy:.1f}%")
    print(f"  Improvement:    {ocr_accuracy - base_accuracy:+.1f}%")

    # Save results
    with open('ocr_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to ocr_accuracy_results.json")


if __name__ == "__main__":
    test_ocr_accuracy()