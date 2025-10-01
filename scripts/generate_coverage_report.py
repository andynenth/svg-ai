#!/usr/bin/env python3
"""
Generate test coverage report as specified in DAY14 Subtask 5.2
Exactly implements the specification from the documentation
"""

import subprocess
import json
from pathlib import Path


def generate_coverage_report():
    """Generate test coverage report"""

    import subprocess

    # Run tests with coverage
    result = subprocess.run([
        'pytest',
        'tests/',
        '--cov=backend',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-report=json'
    ], capture_output=True, text=True)

    print("Coverage test execution:")
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"Return code: {result.returncode}")

    # Parse coverage report
    coverage_file = Path('coverage.json')
    if not coverage_file.exists():
        print("❌ Coverage report not generated")
        return False

    try:
        with open('coverage.json', 'r') as f:
            coverage = json.load(f)

        total_coverage = coverage['totals']['percent_covered']

        print(f"Total coverage: {total_coverage:.1f}%")

        # Check critical files
        critical_files = [
            'backend/ai_modules/classification.py',
            'backend/ai_modules/optimization.py',
            'backend/ai_modules/pipeline.py'
        ]

        for file in critical_files:
            if file in coverage['files']:
                file_cov = coverage['files'][file]['summary']['percent_covered']
                if file_cov < 80:
                    print(f"⚠️ Low coverage in {file}: {file_cov:.1f}%")
                else:
                    print(f"✅ Good coverage in {file}: {file_cov:.1f}%")
            else:
                print(f"⚠️ File not found in coverage: {file}")

        return total_coverage >= 80

    except Exception as e:
        print(f"❌ Error parsing coverage report: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Generating Coverage Report - DAY14 Subtask 5.2")
    print("=" * 50)

    success = generate_coverage_report()

    print("\n" + "=" * 50)
    if success:
        print("✅ Coverage target achieved (>80%)")
    else:
        print("❌ Coverage target not met (<80%)")

    print("\nCoverage reports generated:")
    print("- Terminal: shown above")
    print("- HTML: htmlcov/index.html")
    print("- JSON: coverage.json")