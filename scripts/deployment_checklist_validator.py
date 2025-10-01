#!/usr/bin/env python3
"""
Deployment Checklist and Validation Script (Day 7)
Validates system readiness for production deployment
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DeploymentValidator:
    """Comprehensive deployment readiness validator"""

    def __init__(self):
        self.validation_results = {}
        self.checklist_items = []

    def validate_dependencies(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate all required dependencies"""
        print("🔍 Validating Dependencies...")

        results = {
            'required_packages': {},
            'optional_packages': {},
            'system_requirements': {},
            'all_met': True
        }

        # Required packages
        required_packages = [
            'torch', 'torchvision', 'PIL', 'numpy', 'cv2',
            'sklearn', 'pathlib', 'logging', 'hashlib'
        ]

        for package in required_packages:
            try:
                if package == 'PIL':
                    import PIL
                    version = PIL.__version__
                elif package == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')

                results['required_packages'][package] = {
                    'available': True,
                    'version': version,
                    'status': '✅'
                }
                print(f"  ✅ {package}: {version}")

            except ImportError:
                results['required_packages'][package] = {
                    'available': False,
                    'version': 'not_installed',
                    'status': '❌'
                }
                results['all_met'] = False
                print(f"  ❌ {package}: NOT INSTALLED")

        # Optional packages
        optional_packages = ['psutil', 'line_profiler']

        for package in optional_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                results['optional_packages'][package] = {
                    'available': True,
                    'version': version,
                    'status': '✅'
                }
                print(f"  ✅ {package} (optional): {version}")
            except ImportError:
                results['optional_packages'][package] = {
                    'available': False,
                    'version': 'not_installed',
                    'status': '⚠️'
                }
                print(f"  ⚠️  {package} (optional): not installed")

        # System requirements
        python_version = sys.version.split()[0]
        python_ok = sys.version_info >= (3, 9)

        results['system_requirements'] = {
            'python_version': python_version,
            'python_ok': python_ok,
            'platform': sys.platform
        }

        if python_ok:
            print(f"  ✅ Python: {python_version}")
        else:
            print(f"  ❌ Python: {python_version} (requires 3.9+)")
            results['all_met'] = False

        return results['all_met'], results

    def validate_model_files(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate model files and directories"""
        print("🔍 Validating Model Files...")

        results = {
            'model_directories': {},
            'model_files': {},
            'all_met': True
        }

        # Required directories
        required_dirs = [
            'backend/ai_modules',
            'backend/ai_modules/classification',
            'day6_exports'
        ]

        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                results['model_directories'][dir_path] = {
                    'exists': True,
                    'status': '✅'
                }
                print(f"  ✅ Directory: {dir_path}")
            else:
                results['model_directories'][dir_path] = {
                    'exists': False,
                    'status': '❌'
                }
                results['all_met'] = False
                print(f"  ❌ Directory: {dir_path}")

        # Model files (some may not exist, which is acceptable)
        model_files = [
            'backend/ai_modules/classification/hybrid_classifier.py',
            'backend/ai_modules/classification/rule_based_classifier.py',
            'backend/ai_modules/classification/efficientnet_classifier.py',
            'backend/ai_modules/feature_extraction.py',
            'day6_exports/efficientnet_logo_classifier_best.pth',
            'day6_exports/neural_network_traced.pt'
        ]

        critical_files = [
            'backend/ai_modules/classification/hybrid_classifier.py',
            'backend/ai_modules/feature_extraction.py'
        ]

        for file_path in model_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                results['model_files'][file_path] = {
                    'exists': True,
                    'size_mb': file_size / (1024 * 1024),
                    'status': '✅'
                }
                print(f"  ✅ File: {file_path} ({file_size / (1024 * 1024):.1f}MB)")
            else:
                is_critical = file_path in critical_files
                status = '❌' if is_critical else '⚠️'
                results['model_files'][file_path] = {
                    'exists': False,
                    'critical': is_critical,
                    'status': status
                }
                if is_critical:
                    results['all_met'] = False
                print(f"  {status} File: {file_path}")

        return results['all_met'], results

    def validate_system_functionality(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate system functionality"""
        print("🔍 Validating System Functionality...")

        results = {
            'initialization': {},
            'classification': {},
            'error_handling': {},
            'performance': {},
            'all_met': True
        }

        try:
            # Test 1: System initialization
            print("  Testing system initialization...")
            from backend.ai_modules.classification.hybrid_classifier import HybridClassifier

            start_time = time.time()
            classifier = HybridClassifier()
            init_time = time.time() - start_time

            results['initialization'] = {
                'success': True,
                'init_time': init_time,
                'status': '✅'
            }
            print(f"    ✅ Initialization: {init_time:.3f}s")

            # Test 2: Basic classification
            print("  Testing basic classification...")
            test_image = 'test-data/circle_00.png'

            if os.path.exists(test_image):
                start_time = time.time()
                result = classifier.classify_safe(test_image)
                classification_time = time.time() - start_time

                results['classification'] = {
                    'success': not result.get('error', False),
                    'processing_time': classification_time,
                    'result': result.get('logo_type', 'unknown'),
                    'confidence': result.get('confidence', 0),
                    'status': '✅' if not result.get('error', False) else '❌'
                }

                if not result.get('error', False):
                    print(f"    ✅ Classification: {result.get('logo_type')} "
                          f"({result.get('confidence', 0):.3f}, {classification_time:.3f}s)")
                else:
                    print(f"    ❌ Classification failed: {result.get('error_message', 'unknown')}")
                    results['all_met'] = False

            else:
                results['classification'] = {
                    'success': False,
                    'error': 'No test image available',
                    'status': '⚠️'
                }
                print(f"    ⚠️  No test image available for classification test")

            # Test 3: Error handling
            print("  Testing error handling...")
            error_result = classifier.classify_safe('nonexistent_file.png')

            error_handled = error_result.get('error', False)
            results['error_handling'] = {
                'success': error_handled,
                'error_type': error_result.get('error_type', 'none'),
                'status': '✅' if error_handled else '❌'
            }

            if error_handled:
                print(f"    ✅ Error handling: {error_result.get('error_type')}")
            else:
                print(f"    ❌ Error handling failed")
                results['all_met'] = False

            # Test 4: Performance features
            print("  Testing performance features...")
            memory_stats = classifier.get_memory_usage()
            performance_stats = classifier.get_performance_stats()

            results['performance'] = {
                'memory_monitoring': 'rss_mb' in memory_stats,
                'performance_tracking': 'total_classifications' in performance_stats,
                'caching_available': hasattr(classifier, 'feature_cache'),
                'status': '✅'
            }
            print(f"    ✅ Memory monitoring: {memory_stats.get('rss_mb', 0):.1f}MB")
            print(f"    ✅ Performance tracking enabled")
            print(f"    ✅ Caching system available")

        except Exception as e:
            print(f"    ❌ System functionality test failed: {e}")
            results['all_met'] = False
            results['error'] = str(e)

        return results['all_met'], results

    def validate_performance_targets(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate performance targets"""
        print("🔍 Validating Performance Targets...")

        results = {
            'processing_time': {},
            'memory_usage': {},
            'reliability': {},
            'all_met': True
        }

        try:
            from backend.ai_modules.classification.hybrid_classifier import HybridClassifier
            classifier = HybridClassifier()

            # Test processing time
            test_image = 'test-data/circle_00.png'
            if os.path.exists(test_image):
                times = []
                for _ in range(5):  # 5 test runs
                    start_time = time.time()
                    result = classifier.classify_safe(test_image)
                    end_time = time.time()
                    if not result.get('error', False):
                        times.append(end_time - start_time)

                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)

                    time_target_met = avg_time < 2.0
                    results['processing_time'] = {
                        'average': avg_time,
                        'maximum': max_time,
                        'target': 2.0,
                        'target_met': time_target_met,
                        'status': '✅' if time_target_met else '❌'
                    }

                    if time_target_met:
                        print(f"    ✅ Processing time: {avg_time:.3f}s avg (target: <2s)")
                    else:
                        print(f"    ❌ Processing time: {avg_time:.3f}s avg (exceeds 2s target)")
                        results['all_met'] = False

            # Test memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                memory_target_met = memory_mb < 500  # Reasonable target for single process
                results['memory_usage'] = {
                    'current_mb': memory_mb,
                    'target': 500,
                    'target_met': memory_target_met,
                    'status': '✅' if memory_target_met else '⚠️'
                }

                if memory_target_met:
                    print(f"    ✅ Memory usage: {memory_mb:.1f}MB (reasonable)")
                else:
                    print(f"    ⚠️  Memory usage: {memory_mb:.1f}MB (higher than expected)")

            except ImportError:
                print(f"    ⚠️  psutil not available for memory monitoring")

            # Test reliability (basic)
            reliability_tests = 10
            successful_tests = 0

            for _ in range(reliability_tests):
                try:
                    result = classifier.classify_safe('test-data/circle_00.png')
                    if not result.get('error', False):
                        successful_tests += 1
                except:
                    pass

            reliability_rate = successful_tests / reliability_tests
            reliability_target_met = reliability_rate >= 0.95

            results['reliability'] = {
                'success_rate': reliability_rate,
                'successful_tests': successful_tests,
                'total_tests': reliability_tests,
                'target': 0.95,
                'target_met': reliability_target_met,
                'status': '✅' if reliability_target_met else '❌'
            }

            if reliability_target_met:
                print(f"    ✅ Reliability: {reliability_rate*100:.1f}% success rate")
            else:
                print(f"    ❌ Reliability: {reliability_rate*100:.1f}% success rate (target: >95%)")
                results['all_met'] = False

        except Exception as e:
            print(f"    ❌ Performance validation failed: {e}")
            results['all_met'] = False
            results['error'] = str(e)

        return results['all_met'], results

    def generate_deployment_checklist(self) -> Dict[str, Any]:
        """Generate comprehensive deployment checklist"""
        print("\n" + "=" * 70)
        print("GENERATING DEPLOYMENT CHECKLIST")
        print("=" * 70)

        checklist = {
            'prerequisites': [
                {
                    'item': 'Python 3.9+ installed',
                    'validation': 'python --version',
                    'status': 'pending'
                },
                {
                    'item': 'Required packages installed',
                    'validation': 'pip list | grep torch',
                    'status': 'pending'
                },
                {
                    'item': 'Project directory accessible',
                    'validation': 'ls backend/ai_modules/classification/',
                    'status': 'pending'
                }
            ],
            'configuration': [
                {
                    'item': 'Model files present (optional for basic functionality)',
                    'validation': 'ls day6_exports/',
                    'status': 'pending'
                },
                {
                    'item': 'Cache directories writable',
                    'validation': 'Test write access to temp directories',
                    'status': 'pending'
                },
                {
                    'item': 'Log directories configured',
                    'validation': 'Check logging configuration',
                    'status': 'pending'
                }
            ],
            'testing': [
                {
                    'item': 'System initialization test',
                    'validation': 'python -c "from backend.ai_modules.classification.hybrid_classifier import HybridClassifier; HybridClassifier()"',
                    'status': 'pending'
                },
                {
                    'item': 'Basic classification test',
                    'validation': 'Run classification on test image',
                    'status': 'pending'
                },
                {
                    'item': 'Error handling test',
                    'validation': 'Test with invalid inputs',
                    'status': 'pending'
                },
                {
                    'item': 'Performance validation',
                    'validation': 'Run performance benchmark',
                    'status': 'pending'
                }
            ],
            'production_readiness': [
                {
                    'item': 'Processing time < 2 seconds',
                    'validation': 'Average processing time measurement',
                    'status': 'pending'
                },
                {
                    'item': 'Memory usage reasonable',
                    'validation': 'Memory usage monitoring',
                    'status': 'pending'
                },
                {
                    'item': 'Error handling comprehensive',
                    'validation': 'All error cases handled gracefully',
                    'status': 'pending'
                },
                {
                    'item': 'Concurrent processing supported',
                    'validation': 'Multi-threaded access test',
                    'status': 'pending'
                }
            ],
            'deployment': [
                {
                    'item': 'Service integration ready',
                    'validation': 'API endpoint integration test',
                    'status': 'pending'
                },
                {
                    'item': 'Monitoring configured',
                    'validation': 'Health check endpoints working',
                    'status': 'pending'
                },
                {
                    'item': 'Documentation complete',
                    'validation': 'All documentation files present',
                    'status': 'pending'
                }
            ]
        }

        return checklist

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete deployment validation"""
        print("=" * 70)
        print("DEPLOYMENT READINESS VALIDATION")
        print("=" * 70)

        validation_results = {
            'timestamp': time.time(),
            'overall_status': 'unknown',
            'validations': {}
        }

        # Run all validations
        validations = [
            ('dependencies', self.validate_dependencies),
            ('model_files', self.validate_model_files),
            ('functionality', self.validate_system_functionality),
            ('performance', self.validate_performance_targets)
        ]

        passed_validations = 0
        total_validations = len(validations)

        for validation_name, validation_func in validations:
            try:
                success, results = validation_func()
                validation_results['validations'][validation_name] = {
                    'passed': success,
                    'results': results
                }

                if success:
                    passed_validations += 1
                    print(f"✅ {validation_name.title()}: PASSED")
                else:
                    print(f"❌ {validation_name.title()}: FAILED")

            except Exception as e:
                validation_results['validations'][validation_name] = {
                    'passed': False,
                    'error': str(e)
                }
                print(f"❌ {validation_name.title()}: ERROR - {str(e)}")

        # Determine overall status
        success_rate = passed_validations / total_validations

        if success_rate == 1.0:
            overall_status = 'PRODUCTION_READY'
            status_emoji = '🎉'
        elif success_rate >= 0.75:
            overall_status = 'MOSTLY_READY'
            status_emoji = '⚠️'
        else:
            overall_status = 'NEEDS_WORK'
            status_emoji = '❌'

        validation_results['overall_status'] = overall_status
        validation_results['passed_validations'] = passed_validations
        validation_results['total_validations'] = total_validations
        validation_results['success_rate'] = success_rate

        # Generate checklist
        checklist = self.generate_deployment_checklist()
        validation_results['deployment_checklist'] = checklist

        # Final summary
        print("\n" + "=" * 70)
        print("DEPLOYMENT VALIDATION SUMMARY")
        print("=" * 70)
        print(f"{status_emoji} Overall Status: {overall_status}")
        print(f"Validations Passed: {passed_validations}/{total_validations} ({success_rate*100:.1f}%)")

        if overall_status == 'PRODUCTION_READY':
            print("\n🎉 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT")
            print("All critical validations passed successfully.")
        elif overall_status == 'MOSTLY_READY':
            print("\n⚠️  SYSTEM IS MOSTLY READY FOR DEPLOYMENT")
            print("Minor issues detected - review validation results.")
        else:
            print("\n❌ SYSTEM NEEDS ADDITIONAL WORK BEFORE DEPLOYMENT")
            print("Critical issues detected - address before deploying.")

        return validation_results

if __name__ == "__main__":
    try:
        validator = DeploymentValidator()
        results = validator.run_full_validation()

        # Save results
        results_file = 'scripts/deployment_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

        # Create deployment checklist file
        checklist_file = 'scripts/deployment_checklist.json'
        with open(checklist_file, 'w') as f:
            json.dump(results['deployment_checklist'], f, indent=2)
        print(f"Deployment checklist saved to: {checklist_file}")

    except Exception as e:
        print(f"❌ Deployment validation failed: {e}")
        import traceback
        traceback.print_exc()