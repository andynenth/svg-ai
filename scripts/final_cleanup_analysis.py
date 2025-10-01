#!/usr/bin/env python3
"""
Final Cleanup Analysis for Day 13
Analyzes remaining files and creates merge plan to reduce from ~40 to ~15 files
"""

from pathlib import Path
import ast
from typing import Dict, List, Set
from collections import defaultdict
import json
import os


class FinalCleanupAnalyzer:
    def __init__(self):
        self.essential_files = set()
        self.merge_candidates = defaultdict(list)
        self.refactor_targets = []

    def analyze_remaining_files(self) -> Dict:
        """Analyze the ~40 remaining files"""

        # Essential entry points (must keep)
        self.essential_files = {
            'backend/app.py',                               # Main app
            'backend/api/ai_endpoints.py',                  # API routes
            'backend/converters/ai_enhanced_converter.py',  # Main converter
            'web_server.py'                                 # Web interface
        }

        # Core AI modules (must keep but can consolidate)
        ai_modules = list(Path('backend/ai_modules').rglob('*.py'))

        analysis = {
            'current_count': len(ai_modules),
            'essential': [],
            'can_merge': [],
            'can_remove': [],
            'needs_refactor': []
        }

        for module in ai_modules:
            if '__pycache__' in str(module):
                continue

            try:
                content = module.read_text()
                lines = content.count('\n')

                # Count actual functionality
                tree = ast.parse(content)
                functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

                module_info = {
                    'path': str(module),
                    'lines': lines,
                    'functions': len(functions),
                    'classes': len(classes)
                }

                # Categorize
                if lines < 50 and len(functions) < 3:
                    analysis['can_merge'].append(module_info)
                    self.merge_candidates[module.parent.name].append(module)
                elif 'test' in module.name or 'demo' in module.name:
                    analysis['can_remove'].append(module_info)
                elif lines > 500:
                    analysis['needs_refactor'].append(module_info)
                else:
                    analysis['essential'].append(module_info)
                    self.essential_files.add(str(module))
            except Exception as e:
                print(f"Error analyzing {module}: {e}")

        return analysis

    def identify_merge_opportunities(self) -> List[Dict]:
        """Identify files that can be merged"""
        merge_plan = []

        # Group small related files
        for directory, files in self.merge_candidates.items():
            if len(files) > 1:
                total_lines = 0
                for f in files:
                    try:
                        total_lines += len(f.read_text().split('\n'))
                    except:
                        continue

                if total_lines < 500:  # Can merge if combined size reasonable
                    merge_plan.append({
                        'target': f"{directory}_unified.py",
                        'sources': [str(f) for f in files],
                        'estimated_lines': total_lines,
                        'reduction': len(files) - 1
                    })

        return merge_plan

    def plan_final_structure(self) -> Dict:
        """Plan the final ~15 file structure"""

        final_structure = {
            'backend/': {
                'app.py': 'Main FastAPI application',
                'api/': {
                    'ai_endpoints.py': 'All API endpoints'
                },
                'converters/': {
                    'ai_enhanced_converter.py': 'Unified converter'
                },
                'ai_modules/': {
                    'classification.py': 'Logo classification (merged)',
                    'optimization.py': 'Parameter optimization (merged)',
                    'quality.py': 'Quality metrics (merged)',
                    'pipeline.py': 'Unified pipeline',
                    'utils.py': 'All utilities (merged)'
                }
            },
            'scripts/': {
                'train_models.py': 'Unified training script',
                'benchmark.py': 'Performance benchmarking',
                'validate.py': 'Validation script'
            },
            'tests/': {
                'test_integration.py': 'Integration tests',
                'test_models.py': 'Model tests',
                'test_api.py': 'API tests'
            }
        }

        return final_structure

    def generate_detailed_report(self) -> Dict:
        """Generate comprehensive analysis report"""

        analysis = self.analyze_remaining_files()
        merge_opportunities = self.identify_merge_opportunities()
        final_structure = self.plan_final_structure()

        # Count current files by type
        all_py_files = list(Path('backend').rglob('*.py'))
        current_count = len([f for f in all_py_files if '__pycache__' not in str(f)])

        report = {
            'timestamp': '2025-09-30',
            'current_file_count': current_count,
            'target_file_count': 15,
            'reduction_target': current_count - 15,
            'analysis': analysis,
            'merge_opportunities': merge_opportunities,
            'final_structure': final_structure,
            'recommendations': []
        }

        # Generate recommendations
        if len(merge_opportunities) > 0:
            report['recommendations'].append(f"Merge {len(merge_opportunities)} sets of related files")

        if len(analysis['can_remove']) > 0:
            report['recommendations'].append(f"Remove {len(analysis['can_remove'])} test/demo files")

        if len(analysis['needs_refactor']) > 0:
            report['recommendations'].append(f"Refactor {len(analysis['needs_refactor'])} large files")

        return report


def create_merge_plan():
    """Create detailed plan for merging files"""

    merge_operations = [
        {
            'name': 'Merge Classification Modules',
            'target': 'backend/ai_modules/classification.py',
            'sources': [
                'backend/ai_modules/classification/statistical_classifier.py',
                'backend/ai_modules/classification/logo_classifier.py',
                'backend/ai_modules/classification/feature_extractor.py',
                'backend/ai_modules/classification/efficientnet_classifier.py',
                'backend/ai_modules/classification/hybrid_classifier.py'
            ],
            'strategy': 'Combine into single ClassificationModule class'
        },
        {
            'name': 'Merge Optimization Modules',
            'target': 'backend/ai_modules/optimization.py',
            'sources': [
                'backend/ai_modules/optimization/learned_optimizer.py',
                'backend/ai_modules/optimization/parameter_tuner.py',
                'backend/ai_modules/optimization/online_learner.py',
                'backend/ai_modules/optimization/unified_parameter_formulas.py',
                'backend/ai_modules/optimization/learned_correlations.py',
                'backend/ai_modules/optimization/correlation_rollout.py'
            ],
            'strategy': 'Create OptimizationEngine with all methods'
        },
        {
            'name': 'Merge Quality Modules',
            'target': 'backend/ai_modules/quality.py',
            'sources': [
                'backend/ai_modules/quality/enhanced_metrics.py',
                'backend/ai_modules/quality/quality_tracker.py',
                'backend/ai_modules/quality/ab_testing.py'
            ],
            'strategy': 'Unified QualitySystem class'
        },
        {
            'name': 'Merge Utilities',
            'target': 'backend/ai_modules/utils.py',
            'sources': [
                'backend/ai_modules/utils/cache_manager.py',
                'backend/ai_modules/utils/parallel_processor.py',
                'backend/ai_modules/utils/lazy_loader.py',
                'backend/ai_modules/utils/request_queue.py'
            ],
            'strategy': 'Organize as utility submodules'
        }
    ]

    return merge_operations


def main():
    """Run final cleanup analysis"""
    print("🔍 Final Cleanup Analysis - Day 13")
    print("=" * 50)

    analyzer = FinalCleanupAnalyzer()

    # Generate analysis report
    report = analyzer.generate_detailed_report()

    print(f"Current file count: {report['current_file_count']}")
    print(f"Target file count: {report['target_file_count']}")
    print(f"Files to reduce: {report['reduction_target']}")
    print()

    # Show analysis results
    analysis = report['analysis']
    print("📊 Analysis Results:")
    print(f"  Essential files: {len(analysis['essential'])}")
    print(f"  Can merge: {len(analysis['can_merge'])}")
    print(f"  Can remove: {len(analysis['can_remove'])}")
    print(f"  Needs refactor: {len(analysis['needs_refactor'])}")
    print()

    # Show merge opportunities
    print("🔗 Merge Opportunities:")
    for opportunity in report['merge_opportunities']:
        print(f"  {opportunity['target']}: {opportunity['reduction']} files → 1 file")
    print()

    # Show recommendations
    print("💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    print()

    # Save detailed report
    with open('final_cleanup_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("📄 Detailed report saved to: final_cleanup_analysis_report.json")

    # Generate merge plan
    merge_plan = create_merge_plan()

    print("\n📋 Merge Plan:")
    for operation in merge_plan:
        print(f"  {operation['name']}")
        print(f"    Target: {operation['target']}")
        print(f"    Sources: {len(operation['sources'])} files")
        print(f"    Strategy: {operation['strategy']}")
        print()

    # Save merge plan
    with open('merge_plan.json', 'w') as f:
        json.dump(merge_plan, f, indent=2)

    print("📋 Merge plan saved to: merge_plan.json")

    return report


if __name__ == "__main__":
    main()