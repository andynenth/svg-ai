#!/usr/bin/env python3
"""
Final Validation Script for Day 13
Validates system functionality after cleanup and merging
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_final_validation():
    """Complete validation after cleanup"""

    validation_results = {}

    print("🧪 Running Final Validation - Day 13")
    print("=" * 50)

    # Test 1: Import all modules
    print("Test 1: Module imports...")
    try:
        # Test individual imports with error handling
        import_tests = [
            ("ClassificationModule", "from backend.ai_modules.classification import ClassificationModule"),
            ("OptimizationEngine", "from backend.ai_modules.optimization import OptimizationEngine"),
            ("QualitySystem", "from backend.ai_modules.quality import QualitySystem"),
            ("UnifiedUtils", "from backend.ai_modules.utils import UnifiedUtils")
        ]

        import_results = {}
        for module_name, import_stmt in import_tests:
            try:
                exec(import_stmt)
                import_results[module_name] = 'PASS'
                print(f"  ✓ {module_name}")
            except Exception as e:
                import_results[module_name] = f'FAIL: {e}'
                print(f"  ✗ {module_name}: {e}")

        validation_results['imports'] = import_results

    except Exception as e:
        validation_results['imports'] = f'FAIL: {e}'

    # Test 2: Check file structure
    print("\nTest 2: File structure validation...")
    essential_files = [
        'backend/ai_modules/classification.py',
        'backend/ai_modules/optimization.py',
        'backend/ai_modules/quality.py',
        'backend/ai_modules/utils.py',
        'scripts/train_models.py'
    ]

    structure_results = {}
    for file_path in essential_files:
        if Path(file_path).exists():
            structure_results[file_path] = 'PASS'
            print(f"  ✓ {file_path}")
        else:
            structure_results[file_path] = 'FAIL: Missing'
            print(f"  ✗ {file_path}")

    validation_results['structure'] = structure_results

    # Test 3: Basic functionality test
    print("\nTest 3: Basic functionality...")
    try:
        # Test unified utils
        from backend.ai_modules.utils import UnifiedUtils
        utils = UnifiedUtils()
        utils.cache_set("test_key", "test_value")
        cache_result = utils.cache_get("test_key")

        if cache_result == "test_value":
            validation_results['functionality'] = 'PASS'
            print("  ✓ Basic caching functionality works")
        else:
            validation_results['functionality'] = 'FAIL: Cache test failed'
            print("  ✗ Cache test failed")

    except Exception as e:
        validation_results['functionality'] = f'FAIL: {e}'
        print(f"  ✗ Functionality test failed: {e}")

    # Test 4: Check for syntax errors
    print("\nTest 4: Syntax validation...")
    syntax_results = {}

    for file_path in essential_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Try to compile the code
                compile(content, file_path, 'exec')
                syntax_results[file_path] = 'PASS'
                print(f"  ✓ {file_path}")

            except SyntaxError as e:
                syntax_results[file_path] = f'FAIL: Line {e.lineno}: {e.msg}'
                print(f"  ✗ {file_path}: Syntax error on line {e.lineno}")
            except Exception as e:
                syntax_results[file_path] = f'FAIL: {e}'
                print(f"  ✗ {file_path}: {e}")

    validation_results['syntax'] = syntax_results

    # Test 5: Performance check
    print("\nTest 5: Performance check...")
    try:
        # Run benchmark if available
        benchmark_path = Path('scripts/benchmark.py')
        if benchmark_path.exists():
            result = subprocess.run(
                [sys.executable, str(benchmark_path), '--quick'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                validation_results['performance'] = 'PASS'
                print("  ✓ Benchmark completed successfully")
            else:
                validation_results['performance'] = f'FAIL: {result.stderr}'
                print(f"  ✗ Benchmark failed: {result.stderr}")
        else:
            validation_results['performance'] = 'SKIP: No benchmark script'
            print("  - Benchmark script not found")

    except subprocess.TimeoutExpired:
        validation_results['performance'] = 'FAIL: Timeout'
        print("  ✗ Benchmark timed out")
    except Exception as e:
        validation_results['performance'] = f'FAIL: {e}'
        print(f"  ✗ Performance test failed: {e}")

    # Generate summary
    print(f"\n📊 Validation Summary:")
    total_tests = 0
    passed_tests = 0

    for test_name, results in validation_results.items():
        if isinstance(results, dict):
            # Count individual sub-tests
            for sub_test, result in results.items():
                total_tests += 1
                if 'PASS' in str(result):
                    passed_tests += 1
        else:
            total_tests += 1
            if 'PASS' in str(results):
                passed_tests += 1

    print(f"  Tests passed: {passed_tests}/{total_tests}")
    print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")

    # Overall status
    if passed_tests == total_tests:
        print("  🎉 All tests passed!")
        overall_status = "PASS"
    elif passed_tests >= total_tests * 0.8:
        print("  ⚠️  Most tests passed")
        overall_status = "PARTIAL"
    else:
        print("  ❌ Many tests failed")
        overall_status = "FAIL"

    validation_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests/total_tests*100,
        'overall_status': overall_status
    }

    return validation_results


def generate_validation_report(results: Dict):
    """Generate detailed validation report"""

    report = {
        'timestamp': '2025-09-30',
        'validation_results': results,
        'recommendations': []
    }

    # Generate recommendations
    if 'imports' in results:
        import_results = results['imports']
        if isinstance(import_results, dict):
            failed_imports = [k for k, v in import_results.items() if 'FAIL' in str(v)]
            if failed_imports:
                report['recommendations'].append(f"Fix import issues in: {', '.join(failed_imports)}")

    if 'syntax' in results:
        syntax_results = results['syntax']
        if isinstance(syntax_results, dict):
            syntax_errors = [k for k, v in syntax_results.items() if 'FAIL' in str(v)]
            if syntax_errors:
                report['recommendations'].append(f"Fix syntax errors in: {', '.join(syntax_errors)}")

    # Save report
    import json
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Validation report saved to: validation_report.json")

    return report


def main():
    """Run final validation"""
    results = run_final_validation()
    report = generate_validation_report(results)

    # Return exit code based on validation results
    if results.get('summary', {}).get('overall_status') == 'PASS':
        return 0
    elif results.get('summary', {}).get('overall_status') == 'PARTIAL':
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)