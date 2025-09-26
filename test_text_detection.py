#!/usr/bin/env python3
"""Test the improved text detection algorithm."""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iterative_optimizer_standalone import IterativeOptimizer

def test_text_detection():
    """Test detection on text logos to verify improvement."""

    # List of text logos to test
    text_logos = [
        "data/logos/text_based/text_tech_00.png",
        "data/logos/text_based/text_ai_04.png",
        "data/logos/text_based/text_web_05.png",
        "data/logos/text_based/text_corp_01.png",
        "data/logos/text_based/text_data_02.png",
        "data/logos/text_based/text_net_07.png",
        "data/logos/text_based/text_soft_08.png",
        "data/logos/text_based/text_app_06.png",
        "data/logos/text_based/text_code_09.png",
        "data/logos/text_based/text_cloud_03.png"
    ]

    print("=" * 60)
    print("TESTING IMPROVED TEXT DETECTION ALGORITHM")
    print("=" * 60)

    correct = 0
    total = 0
    results = []

    for logo_path in text_logos:
        if not Path(logo_path).exists():
            print(f"\n⚠️ Skipping {logo_path} - file not found")
            continue

        total += 1

        # Create optimizer instance (we only need the detection)
        optimizer = IterativeOptimizer(
            input_path=logo_path,
            output_dir="temp_test",
            target_ssim=0.98
        )

        # Run detection
        detected_type = optimizer.detect_logo_type()

        # Check if correct
        is_correct = detected_type == 'text'
        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"

        results.append({
            'file': Path(logo_path).name,
            'detected': detected_type,
            'correct': is_correct
        })

        print(f"\n{status} - {Path(logo_path).name}")
        print(f"  Expected: text")
        print(f"  Detected: {detected_type}")
        print("-" * 40)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nTotal tested: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {total - correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Compare with previous results
    print("\n" + "=" * 60)
    print("IMPROVEMENT COMPARISON")
    print("=" * 60)
    print(f"Previous detection accuracy: 0% (0/10)")
    print(f"Current detection accuracy: {accuracy:.1f}% ({correct}/{total})")

    if accuracy > 0:
        print(f"\n🎉 IMPROVEMENT: Detection accuracy increased from 0% to {accuracy:.1f}%!")
    else:
        print(f"\n⚠️ No improvement detected - algorithm may need further adjustment")

    # Show detailed results table
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    print(f"{'Logo':<25} {'Detected':<15} {'Result':<10}")
    print("-" * 50)

    for result in results:
        status = "✅" if result['correct'] else "❌"
        print(f"{result['file']:<25} {result['detected']:<15} {status:<10}")

    return accuracy

if __name__ == "__main__":
    accuracy = test_text_detection()

    # Exit with status code based on improvement
    if accuracy > 50:
        print("\n✅ SUCCESS: Significant improvement achieved!")
        sys.exit(0)
    else:
        print("\n⚠️ PARTIAL SUCCESS: Some improvement but needs more work")
        sys.exit(1)