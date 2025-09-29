#!/usr/bin/env python3
"""
Comprehensive Test Orchestrator for Day 9 Testing
Runs all automated tests in sequence as specified in Day 9 plan
"""

import sys
import time
import subprocess
import requests
from pathlib import Path

class ComprehensiveTestRunner:
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()

    def check_prerequisites(self):
        """Check if all prerequisites are met for testing"""
        print("=" * 60)
        print("DAY 9 COMPREHENSIVE TESTING SUITE")
        print("=" * 60)
        print("Checking prerequisites...")

        # Check API server
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Flask API server is running")
            else:
                print(f"❌ Flask API server returned {response.status_code}")
                return False
        except:
            print("❌ Flask API server not accessible at http://localhost:8001")
            print("Please start the server with: python backend/app.py")
            return False

        # Check test images
        test_dir = Path('data/test')
        required_images = [
            'simple_geometric_logo.png',
            'text_based_logo.png',
            'gradient_logo.png',
            'complex_logo.png'
        ]

        missing_images = []
        for img in required_images:
            if not (test_dir / img).exists():
                missing_images.append(img)

        if missing_images:
            print(f"❌ Missing test images: {missing_images}")
            print("Run: python scripts/create_test_images.py")
            return False
        else:
            print("✅ All test images available")

        # Check test scripts
        test_scripts = [
            'scripts/run_e2e_tests.py',
            'scripts/load_test_classification.py',
            'scripts/user_scenario_tests.py',
            'scripts/security_tests.py'
        ]

        missing_scripts = []
        for script in test_scripts:
            if not Path(script).exists():
                missing_scripts.append(script)

        if missing_scripts:
            print(f"❌ Missing test scripts: {missing_scripts}")
            return False
        else:
            print("✅ All test scripts available")

        return True

    def run_test_suite(self, test_name, script_path, description):
        """Run a specific test suite and capture results"""
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        print(f"Description: {description}")
        print(f"Script: {script_path}")
        print("-" * 60)

        start_time = time.time()

        try:
            # Run the test script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

            duration = time.time() - start_time
            success = result.returncode == 0

            # Store results
            self.test_results[test_name] = {
                'success': success,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'description': description
            }

            if success:
                print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            else:
                print(f"❌ {test_name} FAILED ({duration:.1f}s)")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")

            # Show key output lines
            if result.stdout:
                lines = result.stdout.split('\n')
                summary_lines = [line for line in lines if any(keyword in line.lower()
                    for keyword in ['passed', 'failed', 'success', 'error', '✅', '❌', 'summary'])]

                if summary_lines:
                    print("Key results:")
                    for line in summary_lines[-5:]:  # Show last 5 relevant lines
                        if line.strip():
                            print(f"  {line.strip()}")

            return success

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"❌ {test_name} TIMEOUT ({duration:.1f}s)")
            self.test_results[test_name] = {
                'success': False,
                'duration': duration,
                'error': 'Test timed out',
                'description': description
            }
            return False

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {test_name} ERROR: {e}")
            self.test_results[test_name] = {
                'success': False,
                'duration': duration,
                'error': str(e),
                'description': description
            }
            return False

    def run_all_tests(self):
        """Run all test suites in sequence"""
        if not self.check_prerequisites():
            return False

        # Define test suites in order
        test_suites = [
            {
                'name': 'e2e_workflow',
                'script': 'scripts/run_e2e_tests.py',
                'description': 'End-to-end workflow validation'
            },
            {
                'name': 'load_testing',
                'script': 'scripts/load_test_classification.py',
                'description': 'Performance and load testing'
            },
            {
                'name': 'user_scenarios',
                'script': 'scripts/user_scenario_tests.py',
                'description': 'User acceptance testing'
            },
            {
                'name': 'security_testing',
                'script': 'scripts/security_tests.py',
                'description': 'Security and edge case testing'
            }
        ]

        passed_tests = 0
        total_tests = len(test_suites)

        print(f"\nStarting comprehensive test execution...")
        print(f"Total test suites: {total_tests}")

        for test_suite in test_suites:
            success = self.run_test_suite(
                test_suite['name'],
                test_suite['script'],
                test_suite['description']
            )

            if success:
                passed_tests += 1

            # Small delay between test suites
            time.sleep(2)

        return self.generate_final_report(passed_tests, total_tests)

    def generate_final_report(self, passed_tests, total_tests):
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("COMPREHENSIVE TESTING RESULTS")
        print("=" * 60)

        # Overall summary
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total execution time: {total_duration:.1f}s")

        # Detailed results
        print(f"\nDetailed Results:")
        print("-" * 40)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            description = result['description']

            print(f"{status} {test_name:<20} ({duration:>5.1f}s) - {description}")

            if not result['success'] and 'error' in result:
                print(f"     Error: {result['error']}")

        # Production readiness assessment
        print(f"\n" + "=" * 60)
        print("PRODUCTION READINESS ASSESSMENT")
        print("=" * 60)

        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
            print("✅ System is PRODUCTION READY")
            print("✅ All Day 9 success criteria met")
            print("✅ Classification system validated end-to-end")
            print("\nThe system has successfully completed comprehensive testing and")
            print("is ready for production deployment.")

            # Save success report
            self.save_test_report(True)
            return True

        else:
            print("⚠️  SOME TESTS FAILED")
            print("❌ System NOT ready for production")
            print(f"❌ {total_tests - passed_tests} test suite(s) failed")
            print("\nPlease review and fix the failed tests before production deployment.")

            # Save failure report
            self.save_test_report(False)
            return False

    def save_test_report(self, overall_success):
        """Save detailed test report to file"""
        import json
        from datetime import datetime

        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': overall_success,
            'total_duration': time.time() - self.start_time,
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(1 for r in self.test_results.values() if r['success']),
                'success_rate': (sum(1 for r in self.test_results.values() if r['success']) / len(self.test_results)) * 100
            }
        }

        filename = f"day9_comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed test report saved to: {filename}")

def main():
    """Main function"""
    runner = ComprehensiveTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()