#!/usr/bin/env python3
"""
Automated Coverage Reporting Script

Generate and analyze coverage reports with detailed insights and recommendations.
Provides comprehensive coverage analysis for the SVG-AI project.

Usage:
    python scripts/coverage_report.py [options]

Options:
    --run-tests         Run tests before generating coverage report
    --html             Generate HTML coverage report
    --json             Generate JSON coverage report
    --xml              Generate XML coverage report
    --detailed         Generate detailed analysis
    --threshold VALUE  Set custom coverage threshold (default: 80%)
    --output DIR       Set output directory for reports
    --exclude PATTERN  Exclude files matching pattern
    --verbose          Enable verbose output
    --help             Show this help message
"""

import argparse
import json
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET


class CoverageAnalyzer:
    """Analyze coverage data and generate insights"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.coverage_dir = self.project_root / "coverage_html_report"
        self.coverage_json = self.project_root / "coverage.json"
        self.coverage_xml = self.project_root / "coverage.xml"

    def run_tests_with_coverage(self, verbose: bool = False) -> bool:
        """Run tests with coverage collection"""
        print("🧪 Running tests with coverage collection...")

        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=backend",
            "--cov-config=.coveragerc",
            "--cov-report=json",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term-missing"
        ]

        if verbose:
            cmd.append("-v")
        else:
            cmd.extend(["-q", "--disable-warnings"])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                print("✅ Tests completed successfully")
                return True
            else:
                print(f"❌ Tests failed with return code {result.returncode}")
                if verbose:
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("⏰ Tests timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False

    def load_coverage_data(self) -> Optional[Dict[str, Any]]:
        """Load coverage data from JSON report"""
        try:
            with open(self.coverage_json, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Coverage JSON file not found: {self.coverage_json}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing coverage JSON: {e}")
            return None

    def analyze_coverage_by_module(self, coverage_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Analyze coverage by module/package"""
        files = coverage_data.get('files', {})
        modules = {}

        for file_path, file_data in files.items():
            # Extract module name from file path
            path_parts = Path(file_path).parts
            if 'backend' in path_parts:
                backend_idx = path_parts.index('backend')
                if len(path_parts) > backend_idx + 1:
                    module = path_parts[backend_idx + 1]
                else:
                    module = 'backend_root'
            else:
                module = 'unknown'

            if module not in modules:
                modules[module] = {
                    'files': [],
                    'total_statements': 0,
                    'covered_statements': 0,
                    'missing_lines': 0
                }

            summary = file_data.get('summary', {})
            modules[module]['files'].append(file_path)
            modules[module]['total_statements'] += summary.get('num_statements', 0)
            modules[module]['covered_statements'] += summary.get('covered_lines', 0)
            modules[module]['missing_lines'] += summary.get('missing_lines', 0)

        # Calculate percentages
        for module, data in modules.items():
            total = data['total_statements']
            covered = data['covered_statements']
            data['coverage_percent'] = (covered / total * 100) if total > 0 else 0
            data['file_count'] = len(data['files'])

        return modules

    def identify_coverage_gaps(self, coverage_data: Dict[str, Any], threshold: float = 80.0) -> List[Dict[str, Any]]:
        """Identify files with coverage below threshold"""
        gaps = []
        files = coverage_data.get('files', {})

        for file_path, file_data in files.items():
            summary = file_data.get('summary', {})
            coverage_percent = summary.get('percent_covered', 0)

            if coverage_percent < threshold:
                gaps.append({
                    'file': file_path,
                    'coverage': coverage_percent,
                    'missing_lines': summary.get('missing_lines', 0),
                    'total_lines': summary.get('num_statements', 0),
                    'gap': threshold - coverage_percent
                })

        # Sort by gap size (largest gaps first)
        gaps.sort(key=lambda x: x['gap'], reverse=True)
        return gaps

    def generate_recommendations(self, modules: Dict[str, Dict[str, float]],
                               gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate coverage improvement recommendations"""
        recommendations = []

        # Module-level recommendations
        low_coverage_modules = [
            (name, data) for name, data in modules.items()
            if data['coverage_percent'] < 60
        ]

        if low_coverage_modules:
            recommendations.append("🎯 **High Priority Modules** (< 60% coverage):")
            for name, data in low_coverage_modules:
                recommendations.append(
                    f"   • {name}: {data['coverage_percent']:.1f}% "
                    f"({data['file_count']} files, {data['missing_lines']} missing lines)"
                )

        # File-level recommendations
        if gaps:
            recommendations.append("\n📁 **Top Coverage Gaps** (files needing attention):")
            for i, gap in enumerate(gaps[:5]):  # Top 5 gaps
                recommendations.append(
                    f"   {i+1}. {Path(gap['file']).name}: {gap['coverage']:.1f}% "
                    f"(need {gap['gap']:.1f}% more, {gap['missing_lines']} lines)"
                )

        # Testing strategy recommendations
        recommendations.append("\n🧪 **Testing Strategy Recommendations**:")

        total_gaps = len(gaps)
        if total_gaps > 20:
            recommendations.append("   • Focus on unit tests for low-hanging fruit (simple functions)")
            recommendations.append("   • Implement mock-based testing for external dependencies")
        elif total_gaps > 10:
            recommendations.append("   • Add integration tests for main workflows")
            recommendations.append("   • Enhance error handling test coverage")
        else:
            recommendations.append("   • Add edge case testing for existing functions")
            recommendations.append("   • Implement property-based testing for complex logic")

        return recommendations

    def generate_summary_report(self, coverage_data: Dict[str, Any],
                              modules: Dict[str, Dict[str, float]],
                              gaps: List[Dict[str, Any]],
                              threshold: float) -> str:
        """Generate a comprehensive summary report"""
        meta = coverage_data.get('meta', {})
        totals = coverage_data.get('totals', {})

        # Overall stats
        overall_coverage = totals.get('percent_covered', 0)
        total_files = len(coverage_data.get('files', {}))
        total_statements = totals.get('num_statements', 0)
        covered_statements = totals.get('covered_lines', 0)
        missing_statements = totals.get('missing_lines', 0)

        # Status determination
        if overall_coverage >= threshold:
            status = "🟢 PASSING"
            status_color = "green"
        elif overall_coverage >= threshold - 10:
            status = "🟡 WARNING"
            status_color = "yellow"
        else:
            status = "🔴 FAILING"
            status_color = "red"

        report = f"""
{'='*80}
🎯 SVG-AI Coverage Report
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Coverage Tool: {meta.get('version', 'Unknown')}
Report Status: {status}

📊 OVERALL COVERAGE SUMMARY
{'─'*40}
Overall Coverage:     {overall_coverage:.1f}% ({status})
Coverage Threshold:   {threshold:.1f}%
Total Files:          {total_files}
Total Statements:     {total_statements:,}
Covered Statements:   {covered_statements:,}
Missing Statements:   {missing_statements:,}

📁 MODULE BREAKDOWN
{'─'*40}"""

        # Module breakdown
        sorted_modules = sorted(modules.items(), key=lambda x: x[1]['coverage_percent'])
        for name, data in sorted_modules:
            status_icon = "🟢" if data['coverage_percent'] >= threshold else "🟡" if data['coverage_percent'] >= 50 else "🔴"
            report += f"\n{status_icon} {name:<20} {data['coverage_percent']:>6.1f}% ({data['file_count']} files)"

        # Coverage gaps
        if gaps:
            report += f"\n\n🚨 COVERAGE GAPS ({len(gaps)} files below {threshold}%)\n{'─'*40}"
            for gap in gaps[:10]:  # Show top 10 gaps
                file_name = Path(gap['file']).name
                report += f"\n• {file_name:<30} {gap['coverage']:>6.1f}% (need {gap['gap']:>5.1f}% more)"

        return report

    def generate_detailed_analysis(self, coverage_data: Dict[str, Any]) -> str:
        """Generate detailed coverage analysis"""
        files = coverage_data.get('files', {})
        analysis = "\n📋 DETAILED FILE ANALYSIS\n" + "="*50 + "\n"

        # Group files by coverage ranges
        coverage_ranges = {
            "excellent": [],    # 95-100%
            "good": [],         # 80-94%
            "fair": [],         # 60-79%
            "poor": [],         # 40-59%
            "critical": []      # 0-39%
        }

        for file_path, file_data in files.items():
            summary = file_data.get('summary', {})
            coverage = summary.get('percent_covered', 0)

            if coverage >= 95:
                coverage_ranges["excellent"].append((file_path, coverage))
            elif coverage >= 80:
                coverage_ranges["good"].append((file_path, coverage))
            elif coverage >= 60:
                coverage_ranges["fair"].append((file_path, coverage))
            elif coverage >= 40:
                coverage_ranges["poor"].append((file_path, coverage))
            else:
                coverage_ranges["critical"].append((file_path, coverage))

        # Report by range
        range_info = {
            "excellent": ("🟢 EXCELLENT", "95-100%"),
            "good": ("🟢 GOOD", "80-94%"),
            "fair": ("🟡 FAIR", "60-79%"),
            "poor": ("🟠 POOR", "40-59%"),
            "critical": ("🔴 CRITICAL", "0-39%")
        }

        for range_name, files_list in coverage_ranges.items():
            if files_list:
                icon, range_desc = range_info[range_name]
                analysis += f"\n{icon} Coverage {range_desc} - {len(files_list)} files:\n"

                for file_path, coverage in sorted(files_list, key=lambda x: x[1], reverse=True):
                    file_name = Path(file_path).name
                    analysis += f"  • {file_name:<40} {coverage:>6.1f}%\n"

        return analysis

    def export_coverage_badge_data(self, coverage_data: Dict[str, Any], output_dir: Path):
        """Export data for coverage badges (shields.io format)"""
        totals = coverage_data.get('totals', {})
        coverage_percent = totals.get('percent_covered', 0)

        # Determine badge color
        if coverage_percent >= 80:
            color = "brightgreen"
        elif coverage_percent >= 60:
            color = "yellow"
        else:
            color = "red"

        badge_data = {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{coverage_percent:.1f}%",
            "color": color
        }

        badge_file = output_dir / "coverage-badge.json"
        with open(badge_file, 'w') as f:
            json.dump(badge_data, f, indent=2)

        print(f"📊 Badge data exported to: {badge_file}")

    def generate_html_enhancement(self, output_dir: Path):
        """Enhance HTML report with custom styling and navigation"""
        html_file = output_dir / "index.html"

        if not html_file.exists():
            print("❌ HTML coverage report not found")
            return

        # Add custom CSS and JavaScript enhancements
        custom_css = """
        <style>
        .coverage-enhancement {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        .coverage-stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
        }
        .quick-nav {
            position: fixed;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border-radius: 6px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        </style>
        """

        enhancement_html = f"""
        {custom_css}
        <div class="coverage-enhancement">
            <h2>🎯 SVG-AI Coverage Report</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <div class="coverage-stats">
                <div class="stat-box">
                    <h3>Overall</h3>
                    <p>Coverage Status</p>
                </div>
                <div class="stat-box">
                    <h3>Modules</h3>
                    <p>Breakdown</p>
                </div>
                <div class="stat-box">
                    <h3>Trends</h3>
                    <p>Analysis</p>
                </div>
            </div>
        </div>
        """

        try:
            with open(html_file, 'r') as f:
                content = f.read()

            # Insert enhancement after <body> tag
            enhanced_content = content.replace(
                '<body>',
                f'<body>{enhancement_html}'
            )

            with open(html_file, 'w') as f:
                f.write(enhanced_content)

            print(f"✨ Enhanced HTML report: {html_file}")

        except Exception as e:
            print(f"⚠️ Could not enhance HTML report: {e}")


def main():
    """Main entry point for coverage reporting"""
    parser = argparse.ArgumentParser(
        description="Generate and analyze coverage reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--run-tests', action='store_true',
                       help='Run tests before generating report')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--json', action='store_true',
                       help='Generate JSON coverage report')
    parser.add_argument('--xml', action='store_true',
                       help='Generate XML coverage report')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed analysis')
    parser.add_argument('--threshold', type=float, default=80.0,
                       help='Coverage threshold percentage (default: 80)')
    parser.add_argument('--output', type=str,
                       help='Output directory for reports')
    parser.add_argument('--exclude', type=str, action='append',
                       help='Exclude files matching pattern')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--badge', action='store_true',
                       help='Generate coverage badge data')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CoverageAnalyzer()

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = analyzer.coverage_dir

    print(f"🚀 SVG-AI Coverage Analysis Tool")
    print(f"Output directory: {output_dir}")
    print(f"Coverage threshold: {args.threshold}%")

    # Run tests if requested
    if args.run_tests:
        success = analyzer.run_tests_with_coverage(verbose=args.verbose)
        if not success:
            print("❌ Test execution failed. Continuing with existing coverage data...")

    # Load coverage data
    coverage_data = analyzer.load_coverage_data()
    if not coverage_data:
        print("❌ No coverage data available. Run tests first with --run-tests")
        return 1

    # Analyze coverage
    print("\n📊 Analyzing coverage data...")
    modules = analyzer.analyze_coverage_by_module(coverage_data)
    gaps = analyzer.identify_coverage_gaps(coverage_data, args.threshold)
    recommendations = analyzer.generate_recommendations(modules, gaps)

    # Generate reports
    summary_report = analyzer.generate_summary_report(
        coverage_data, modules, gaps, args.threshold
    )

    print(summary_report)

    if recommendations:
        print("\n💡 RECOMMENDATIONS")
        print("─" * 40)
        for rec in recommendations:
            print(rec)

    if args.detailed:
        detailed_analysis = analyzer.generate_detailed_analysis(coverage_data)
        print(detailed_analysis)

    # Generate badge data
    if args.badge:
        analyzer.export_coverage_badge_data(coverage_data, output_dir)

    # Enhance HTML report
    if args.html and analyzer.coverage_dir.exists():
        analyzer.generate_html_enhancement(analyzer.coverage_dir)

    # Save summary report to file
    report_file = output_dir / f"coverage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(summary_report)
        if recommendations:
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("─" * 40 + "\n")
            for rec in recommendations:
                f.write(rec + "\n")
        if args.detailed:
            f.write(analyzer.generate_detailed_analysis(coverage_data))

    print(f"\n📄 Summary report saved to: {report_file}")

    # Determine exit code based on coverage threshold
    overall_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
    if overall_coverage >= args.threshold:
        print(f"\n✅ Coverage goal achieved: {overall_coverage:.1f}% >= {args.threshold}%")
        return 0
    else:
        print(f"\n❌ Coverage below threshold: {overall_coverage:.1f}% < {args.threshold}%")
        return 1


if __name__ == "__main__":
    sys.exit(main())