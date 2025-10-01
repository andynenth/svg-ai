#!/usr/bin/env python3
"""
Production E2E Test Runner
Executes comprehensive end-to-end tests for production validation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from tests.e2e.test_production_workflows import TestProductionWorkflows

    def main():
        print("🚀 SVG-AI Production Validation Suite")
        print("=====================================")
        print()

        # Check if system is accessible
        try:
            import requests
            response = requests.get("http://localhost/health", timeout=5)
            if response.status_code != 200:
                print("❌ System not accessible at http://localhost")
                print("   Please ensure the application is running")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to system: {e}")
            print("   Please ensure the application is running at http://localhost")
            return False

        # Run validation tests
        validator = TestProductionWorkflows()

        try:
            success = validator.run_all_tests()

            if success:
                print("\n🎉 PRODUCTION VALIDATION SUCCESSFUL!")
                print("✅ System is ready for production deployment")
                return True
            else:
                print("\n⚠️  PRODUCTION VALIDATION INCOMPLETE")
                print("❌ Address failing tests before production deployment")
                return False

        except Exception as e:
            print(f"\n💥 Validation suite crashed: {e}")
            print("❌ System not ready for production")
            return False

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install required packages: pip install requests pytest")
    sys.exit(1)