#!/usr/bin/env python3
"""Continuous Testing Workflow for AI Modules"""

import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import argparse

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"🏃 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True, result.stdout
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False, str(e)

def run_tests():
    """Run the complete test suite"""
    print("🧪 Running AI Module Test Suite")
    print("=" * 50)

    # Test commands in order of importance
    test_commands = [
        ("python3 -m pytest tests/ai_modules/ -x", "Unit Tests"),
        ("coverage run -m pytest tests/ai_modules/", "Tests with Coverage"),
        ("coverage report --fail-under=60", "Coverage Check (60% minimum)"),
        ("python3 scripts/test_ai_imports.py", "Import Validation"),
        ("python3 scripts/test_performance_monitoring.py", "Performance Monitoring"),
        ("python3 scripts/test_logging_config.py", "Logging Configuration")
    ]

    results = []
    for command, description in test_commands:
        success, output = run_command(command, description)
        results.append((description, success, output))

        if not success and "Coverage Check" not in description:
            print(f"🛑 Stopping due to failure in: {description}")
            break

    return results

def generate_test_report(results):
    """Generate a test report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []
    report.append("# AI Modules Test Report")
    report.append(f"Generated: {timestamp}")
    report.append("")

    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    report.append(f"## Summary: {passed}/{total} tests passed")
    report.append("")

    # Detailed results
    report.append("## Detailed Results")
    report.append("")

    for test_name, success, output in results:
        status = "✅ PASS" if success else "❌ FAIL"
        report.append(f"### {test_name}: {status}")
        if not success:
            report.append("```")
            report.append(output[:500] + "..." if len(output) > 500 else output)
            report.append("```")
        report.append("")

    # Coverage information
    report.append("## Coverage Information")
    coverage_success, coverage_output = run_command("coverage report", "Coverage Report")
    if coverage_success:
        report.append("```")
        report.append(coverage_output)
        report.append("```")

    return "\n".join(report)

def watch_mode():
    """Run tests in watch mode"""
    print("👀 Starting watch mode - tests will run when files change")
    print("Press Ctrl+C to stop")

    last_run = 0
    ai_modules_path = Path("backend/ai_modules")
    tests_path = Path("tests/ai_modules")

    def get_latest_mtime():
        """Get the latest modification time of relevant files"""
        latest = 0
        for path in [ai_modules_path, tests_path]:
            if path.exists():
                for file_path in path.rglob("*.py"):
                    latest = max(latest, file_path.stat().st_mtime)
        return latest

    try:
        while True:
            current_mtime = get_latest_mtime()
            if current_mtime > last_run:
                print(f"\n🔄 Files changed, running tests... ({datetime.now().strftime('%H:%M:%S')})")
                results = run_tests()

                # Quick summary
                passed = sum(1 for _, success, _ in results if success)
                total = len(results)
                print(f"\n📊 Quick Summary: {passed}/{total} tests passed")

                last_run = time.time()

            time.sleep(2)  # Check every 2 seconds

    except KeyboardInterrupt:
        print("\n🛑 Watch mode stopped")

def quick_test():
    """Run a quick subset of tests for fast feedback"""
    print("⚡ Running Quick Test Suite")
    print("=" * 30)

    quick_commands = [
        ("python3 -m pytest tests/ai_modules/test_classification.py -x", "Classification Tests"),
        ("python3 -m pytest tests/ai_modules/test_integration.py -x", "Integration Tests"),
        ("python3 scripts/test_ai_imports.py", "Import Check")
    ]

    for command, description in quick_commands:
        success, output = run_command(command, description)
        if not success:
            print(f"🛑 Quick test failed: {description}")
            return False

    print("🎉 Quick tests passed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Continuous Testing for AI Modules')
    parser.add_argument('--watch', action='store_true', help='Run in watch mode')
    parser.add_argument('--quick', action='store_true', help='Run quick test suite')
    parser.add_argument('--report', type=str, help='Save test report to file')
    parser.add_argument('--coverage-html', action='store_true', help='Generate HTML coverage report')

    args = parser.parse_args()

    if args.watch:
        watch_mode()
    elif args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)
    else:
        results = run_tests()

        # Generate HTML coverage report if requested
        if args.coverage_html:
            run_command("coverage html", "HTML Coverage Report")
            print("📄 HTML coverage report generated in coverage_html_report/")

        # Generate and save report if requested
        if args.report:
            report_content = generate_test_report(results)
            with open(args.report, 'w') as f:
                f.write(report_content)
            print(f"📄 Test report saved to: {args.report}")

        # Exit with appropriate code
        passed = sum(1 for _, success, _ in results if success)
        total = len(results)
        success_rate = passed / total if total > 0 else 0

        print(f"\n🎯 Final Result: {passed}/{total} tests passed ({success_rate:.1%})")

        if success_rate >= 0.8:
            print("🏆 Excellent! All critical tests passed")
            sys.exit(0)
        elif success_rate >= 0.6:
            print("⚠️  Most tests passed, but improvements needed")
            sys.exit(0)
        else:
            print("💥 Too many test failures - please fix issues")
            sys.exit(1)

if __name__ == "__main__":
    main()