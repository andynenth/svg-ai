#!/usr/bin/env python3
"""
End-to-End Test Runner for Classification System
Executes comprehensive E2E tests as specified in Day 9 plan
"""

import sys
import time
import pytest
import subprocess
import requests
from pathlib import Path

def check_api_availability(base_url="http://localhost:8001"):
    """Check if the API server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_e2e_tests():
    """Run comprehensive E2E tests"""
    print("=" * 60)
    print("CLASSIFICATION SYSTEM - END-TO-END TESTING")
    print("=" * 60)

    # Check if API server is running
    print("Checking API server availability...")
    if not check_api_availability():
        print("❌ API server not available at http://localhost:8001")
        print("Please start the Flask server first:")
        print("  python backend/app.py")
        return False

    print("✅ API server is available")

    # Check if test images exist
    test_dir = Path('data/test')
    required_images = [
        'simple_geometric_logo.png',
        'text_based_logo.png',
        'gradient_logo.png',
        'complex_logo.png'
    ]

    print("Checking test images...")
    missing_images = []
    for img in required_images:
        if not (test_dir / img).exists():
            missing_images.append(img)

    if missing_images:
        print(f"❌ Missing test images: {missing_images}")
        print("Run: python scripts/create_test_images.py")
        return False

    print("✅ All test images available")

    # Run the E2E tests
    print("\nStarting End-to-End Tests...")
    print("-" * 40)

    try:
        # Run pytest with verbose output
        result = pytest.main([
            'tests/test_e2e_classification_integration.py',
            '-v',
            '--tb=short',
            '--no-header'
        ])

        if result == 0:
            print("\n" + "=" * 60)
            print("✅ ALL E2E TESTS PASSED")
            print("=" * 60)
            print("Classification system is ready for production!")
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ SOME E2E TESTS FAILED")
            print("=" * 60)
            print("Please review the test failures above.")
            return False

    except Exception as e:
        print(f"❌ Error running E2E tests: {e}")
        return False

def main():
    """Main function"""
    success = run_e2e_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()