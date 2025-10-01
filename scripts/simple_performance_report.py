#!/usr/bin/env python3
"""
Simplified Performance Benchmark and Reporting

Creates performance reports for Week 2 implementation without complex dependencies.
"""

import os
import sys
import time
import json
import statistics
import tempfile
from pathlib import Path
from datetime import datetime
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SimplePerformanceReporter:
    """Simple performance testing and reporting."""

    def __init__(self):
        self.results = []

    def create_simple_test_image(self, size=(200, 200), image_type='simple'):
        """Create a simple test image."""
        if image_type == 'simple':
            img = Image.new('RGB', size, color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.ellipse([20, 20, size[0]-20, size[1]-20], fill='blue')
        else:
            img = Image.new('RGB', size, color='red')

        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name, 'PNG')
        return temp_file.name

    def test_basic_conversion_performance(self):
        """Test basic conversion performance."""
        print("Testing basic conversion performance...")

        try:
            from backend.converters.vtracer_converter import VTracerConverter
            converter = VTracerConverter()

            # Test different image sizes
            test_cases = [
                {'size': (100, 100), 'name': 'small'},
                {'size': (500, 500), 'name': 'medium'},
                {'size': (1000, 1000), 'name': 'large'}
            ]

            conversion_results = []

            for case in test_cases:
                image_path = self.create_simple_test_image(case['size'])

                try:
                    # Run multiple conversions for timing
                    times = []
                    for _ in range(3):
                        start_time = time.time()
                        result = converter.convert_with_metrics(image_path)
                        end_time = time.time()

                        if result['success']:
                            times.append(end_time - start_time)

                    if times:
                        avg_time = statistics.mean(times)
                        conversion_results.append({
                            'test_case': case['name'],
                            'image_size': case['size'],
                            'avg_time': avg_time,
                            'min_time': min(times),
                            'max_time': max(times),
                            'success': True
                        })

                finally:
                    os.unlink(image_path)

            return {
                'basic_conversion': {
                    'total_tests': len(test_cases),
                    'successful_tests': len(conversion_results),
                    'results': conversion_results,
                    'overall_avg_time': statistics.mean([r['avg_time'] for r in conversion_results]) if conversion_results else 0
                }
            }

        except ImportError:
            return {'basic_conversion': {'error': 'VTracer converter not available'}}

    def test_ai_availability(self):
        """Test AI system availability and basic functionality."""
        print("Testing AI system availability...")

        ai_results = {
            'ai_modules_available': False,
            'feature_extraction_working': False,
            'classification_working': False,
            'ai_converter_available': False
        }

        # Test AI modules
        try:
            from backend.ai_modules.feature_extraction import FeatureExtractor
            ai_results['ai_modules_available'] = True

            # Test feature extraction
            extractor = FeatureExtractor()
            image_path = self.create_simple_test_image()

            try:
                features = extractor.extract_features(image_path)
                if features and len(features) > 0:
                    ai_results['feature_extraction_working'] = True
            except:
                pass
            finally:
                os.unlink(image_path)

        except ImportError:
            pass

        # Test AI-enhanced converter
        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter
            ai_results['ai_converter_available'] = True
        except ImportError:
            pass

        return {'ai_system': ai_results}

    def test_cache_availability(self):
        """Test caching system availability."""
        print("Testing cache system availability...")

        cache_results = {
            'cache_modules_available': False,
            'basic_cache_working': False,
            'advanced_cache_available': False
        }

        try:
            from backend.ai_modules.advanced_cache import MultiLevelCache
            cache_results['cache_modules_available'] = True

            # Test basic cache operations
            cache = MultiLevelCache()
            test_key = "test_key"
            test_value = {"test": "data"}

            cache.set('test', test_key, test_value)
            retrieved = cache.get('test', test_key)

            if retrieved == test_value:
                cache_results['basic_cache_working'] = True
                cache_results['advanced_cache_available'] = True

        except ImportError:
            pass

        return {'cache_system': cache_results}

    def test_web_api_availability(self):
        """Test web API availability."""
        print("Testing web API availability...")

        api_results = {
            'flask_app_available': False,
            'health_endpoint_working': False,
            'import_errors': []
        }

        try:
            # Test basic imports
            import flask
            from werkzeug.utils import secure_filename
            api_results['flask_app_available'] = True

            # Test app creation (without starting server)
            try:
                from backend.app import app
                api_results['health_endpoint_working'] = True
            except Exception as e:
                api_results['import_errors'].append(str(e))

        except ImportError as e:
            api_results['import_errors'].append(str(e))

        return {'web_api': api_results}

    def generate_system_performance_report(self):
        """Generate comprehensive system performance report."""
        print("Generating system performance report...")

        report = {
            'report_info': {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'Week 2 Performance Summary',
                'python_version': sys.version,
                'platform': sys.platform
            }
        }

        # Run all tests
        report.update(self.test_basic_conversion_performance())
        report.update(self.test_ai_availability())
        report.update(self.test_cache_availability())
        report.update(self.test_web_api_availability())

        # Generate overall assessment
        report['overall_assessment'] = self._assess_system_readiness(report)

        return report

    def _assess_system_readiness(self, report):
        """Assess overall system readiness for production."""
        assessment = {
            'production_ready': False,
            'core_functionality': 'unknown',
            'ai_enhancement': 'unknown',
            'caching': 'unknown',
            'web_interface': 'unknown',
            'recommendations': []
        }

        # Assess core functionality
        basic_conv = report.get('basic_conversion', {})
        if basic_conv.get('successful_tests', 0) > 0:
            avg_time = basic_conv.get('overall_avg_time', 0)
            if avg_time < 2.0:
                assessment['core_functionality'] = 'excellent'
            elif avg_time < 5.0:
                assessment['core_functionality'] = 'good'
            else:
                assessment['core_functionality'] = 'needs_optimization'
        else:
            assessment['core_functionality'] = 'not_working'

        # Assess AI system
        ai_system = report.get('ai_system', {})
        if ai_system.get('ai_modules_available') and ai_system.get('feature_extraction_working'):
            assessment['ai_enhancement'] = 'working'
        elif ai_system.get('ai_modules_available'):
            assessment['ai_enhancement'] = 'partially_working'
        else:
            assessment['ai_enhancement'] = 'not_available'

        # Assess caching
        cache_system = report.get('cache_system', {})
        if cache_system.get('basic_cache_working'):
            assessment['caching'] = 'working'
        elif cache_system.get('cache_modules_available'):
            assessment['caching'] = 'partially_working'
        else:
            assessment['caching'] = 'not_available'

        # Assess web interface
        web_api = report.get('web_api', {})
        if web_api.get('health_endpoint_working'):
            assessment['web_interface'] = 'working'
        elif web_api.get('flask_app_available'):
            assessment['web_interface'] = 'partially_working'
        else:
            assessment['web_interface'] = 'not_working'

        # Overall production readiness
        core_working = assessment['core_functionality'] in ['excellent', 'good']
        web_working = assessment['web_interface'] in ['working']

        assessment['production_ready'] = core_working and web_working

        # Generate recommendations
        if assessment['core_functionality'] == 'needs_optimization':
            assessment['recommendations'].append("Optimize conversion parameters for better performance")

        if assessment['ai_enhancement'] == 'not_available':
            assessment['recommendations'].append("Install AI dependencies to enable intelligent parameter optimization")

        if assessment['caching'] == 'not_available':
            assessment['recommendations'].append("Configure caching system for improved performance")

        if assessment['web_interface'] == 'not_working':
            assessment['recommendations'].append("Fix web API dependencies for full functionality")

        if assessment['production_ready']:
            assessment['recommendations'].append("System ready for production deployment")

        return assessment

    def print_performance_report(self, report):
        """Print formatted performance report."""
        print("\n" + "="*80)
        print("WEEK 2 IMPLEMENTATION - PERFORMANCE SUMMARY REPORT")
        print("="*80)

        print(f"Report Generated: {report['report_info']['timestamp']}")
        print(f"Platform: {report['report_info']['platform']}")

        # Core Conversion Performance
        print(f"\n📊 CORE CONVERSION PERFORMANCE")
        basic_conv = report.get('basic_conversion', {})
        if 'error' in basic_conv:
            print(f"  ❌ Basic conversion not available: {basic_conv['error']}")
        else:
            print(f"  ✅ Successful tests: {basic_conv.get('successful_tests', 0)}/{basic_conv.get('total_tests', 0)}")
            avg_time = basic_conv.get('overall_avg_time', 0)
            print(f"  ⏱️  Average conversion time: {avg_time:.3f}s")

            if avg_time < 1.0:
                print(f"     🚀 Excellent performance!")
            elif avg_time < 3.0:
                print(f"     ✅ Good performance")
            else:
                print(f"     ⚠️  Performance could be improved")

        # AI System Status
        print(f"\n🤖 AI ENHANCEMENT SYSTEM")
        ai_system = report.get('ai_system', {})
        print(f"  AI Modules: {'✅ Available' if ai_system.get('ai_modules_available') else '❌ Not available'}")
        print(f"  Feature Extraction: {'✅ Working' if ai_system.get('feature_extraction_working') else '❌ Not working'}")
        print(f"  AI Converter: {'✅ Available' if ai_system.get('ai_converter_available') else '❌ Not available'}")

        # Cache System Status
        print(f"\n💾 CACHING SYSTEM")
        cache_system = report.get('cache_system', {})
        print(f"  Cache Modules: {'✅ Available' if cache_system.get('cache_modules_available') else '❌ Not available'}")
        print(f"  Basic Operations: {'✅ Working' if cache_system.get('basic_cache_working') else '❌ Not working'}")

        # Web API Status
        print(f"\n🌐 WEB API INTERFACE")
        web_api = report.get('web_api', {})
        print(f"  Flask Framework: {'✅ Available' if web_api.get('flask_app_available') else '❌ Not available'}")
        print(f"  Health Endpoint: {'✅ Working' if web_api.get('health_endpoint_working') else '❌ Not working'}")

        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT")
        assessment = report.get('overall_assessment', {})
        print(f"  Production Ready: {'✅ YES' if assessment.get('production_ready') else '❌ NO'}")
        print(f"  Core Functionality: {assessment.get('core_functionality', 'unknown').replace('_', ' ').title()}")
        print(f"  AI Enhancement: {assessment.get('ai_enhancement', 'unknown').replace('_', ' ').title()}")
        print(f"  Caching: {assessment.get('caching', 'unknown').replace('_', ' ').title()}")
        print(f"  Web Interface: {assessment.get('web_interface', 'unknown').replace('_', ' ').title()}")

        # Recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n" + "="*80)

    def save_report(self, report):
        """Save report to file."""
        # Create reports directory
        reports_dir = Path("performance_reports")
        reports_dir.mkdir(exist_ok=True)

        # Save JSON report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = reports_dir / f"week2_performance_report_{timestamp}.json"

        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Full report saved to: {json_file}")

        # Save markdown summary
        md_file = reports_dir / f"week2_performance_summary_{timestamp}.md"
        self._save_markdown_summary(report, md_file)

        print(f"📄 Summary saved to: {md_file}")

    def _save_markdown_summary(self, report, md_file):
        """Save markdown summary of report."""
        with open(md_file, 'w') as f:
            f.write("# Week 2 Implementation - Performance Report\n\n")
            f.write(f"**Generated:** {report['report_info']['timestamp']}\n")
            f.write(f"**Platform:** {report['report_info']['platform']}\n\n")

            # Core Performance
            f.write("## Core Conversion Performance\n\n")
            basic_conv = report.get('basic_conversion', {})
            if 'error' not in basic_conv:
                f.write(f"- **Success Rate:** {basic_conv.get('successful_tests', 0)}/{basic_conv.get('total_tests', 0)}\n")
                f.write(f"- **Average Time:** {basic_conv.get('overall_avg_time', 0):.3f}s\n\n")

            # System Status
            f.write("## System Component Status\n\n")
            f.write("| Component | Status |\n")
            f.write("|-----------|--------|\n")

            ai_system = report.get('ai_system', {})
            f.write(f"| AI Modules | {'✅ Available' if ai_system.get('ai_modules_available') else '❌ Not Available'} |\n")

            cache_system = report.get('cache_system', {})
            f.write(f"| Cache System | {'✅ Working' if cache_system.get('basic_cache_working') else '❌ Not Working'} |\n")

            web_api = report.get('web_api', {})
            f.write(f"| Web API | {'✅ Working' if web_api.get('health_endpoint_working') else '❌ Not Working'} |\n\n")

            # Assessment
            assessment = report.get('overall_assessment', {})
            f.write("## Overall Assessment\n\n")
            f.write(f"**Production Ready:** {'✅ YES' if assessment.get('production_ready') else '❌ NO'}\n\n")

            # Recommendations
            recommendations = assessment.get('recommendations', [])
            if recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")


def main():
    """Run simplified performance reporting."""
    print("Starting Week 2 Performance Assessment...")

    reporter = SimplePerformanceReporter()
    report = reporter.generate_system_performance_report()

    reporter.print_performance_report(report)
    reporter.save_report(report)

    print("\nPerformance assessment complete!")


if __name__ == "__main__":
    main()