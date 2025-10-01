#!/usr/bin/env python3
"""
Fix Test Imports for 80% Coverage Target
Systematically updates all test files to use consolidated module structure
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


class TestImportFixer:
    """Fix import statements in test files to match consolidated module structure"""

    def __init__(self):
        # Mapping of old import patterns to new consolidated imports
        self.import_mappings = {
            # Classification module mappings
            'from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor':
                'from backend.ai_modules.classification import ClassificationModule',
            'from backend.ai_modules.classification.logo_classifier import LogoClassifier':
                'from backend.ai_modules.classification import ClassificationModule',
            'from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier':
                'from backend.ai_modules.classification import ClassificationModule',
            'from backend.ai_modules.classification.efficientnet_classifier_fixed import EfficientNetClassifierFixed':
                'from backend.ai_modules.classification import ClassificationModule',

            # Optimization module mappings
            'from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.feature_mapping_optimizer import FeatureMappingOptimizer':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.adaptive_optimizer import AdaptiveOptimizer':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.intelligent_router import IntelligentRouter':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.regression_optimizer import RegressionBasedOptimizer':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.vtracer_environment import VTracerEnvironment':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.learned_correlations import LearnedCorrelations':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.correlation_formulas import CorrelationFormulas':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.parameter_bounds import VTracerParameterBounds':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.quality_metrics import OptimizationQualityMetrics':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.optimization_logger import OptimizationLogger':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.validator import ParameterValidator':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.validation_pipeline import ValidationPipeline':
                'from backend.ai_modules.optimization import OptimizationEngine',
            'from backend.ai_modules.optimization.vtracer_test import VTracerTestHarness':
                'from backend.ai_modules.optimization import OptimizationEngine',

            # Quality module mappings
            'from backend.ai_modules.quality.enhanced_metrics import EnhancedMetrics':
                'from backend.ai_modules.quality import QualitySystem',
            'from backend.ai_modules.quality.ab_testing import ABTesting':
                'from backend.ai_modules.quality import QualitySystem',

            # Utils module mappings
            'from backend.ai_modules.utils.performance_monitor import PerformanceMonitor':
                'from backend.ai_modules.utils import UnifiedUtils',
            'from backend.ai_modules.utils.cache_manager import CacheManager':
                'from backend.ai_modules.utils import UnifiedUtils',
            'from backend.ai_modules.utils.parallel_processor import ParallelProcessor':
                'from backend.ai_modules.utils import UnifiedUtils',

            # Converter mappings
            'from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter':
                'from backend.converters.ai_enhanced_converter import AIEnhancedConverter',
        }

        # Class name replacements
        self.class_replacements = {
            'ImageFeatureExtractor': 'ClassificationModule().feature_extractor',
            'LogoClassifier': 'ClassificationModule()',
            'RuleBasedClassifier': 'ClassificationModule()',
            'EfficientNetClassifierFixed': 'ClassificationModule()',
            'FeatureMappingOptimizer': 'OptimizationEngine()',
            'AdaptiveOptimizer': 'OptimizationEngine()',
            'IntelligentRouter': 'OptimizationEngine()',
            'RegressionBasedOptimizer': 'OptimizationEngine()',
            'VTracerEnvironment': 'OptimizationEngine()',
            'LearnedCorrelations': 'OptimizationEngine()',
            'CorrelationFormulas': 'OptimizationEngine()',
            'VTracerParameterBounds': 'OptimizationEngine()',
            'OptimizationQualityMetrics': 'OptimizationEngine()',
            'OptimizationLogger': 'OptimizationEngine()',
            'ParameterValidator': 'OptimizationEngine()',
            'ValidationPipeline': 'OptimizationEngine()',
            'VTracerTestHarness': 'OptimizationEngine()',
            'EnhancedMetrics': 'QualitySystem()',
            'ABTesting': 'QualitySystem()',
            'PerformanceMonitor': 'UnifiedUtils()',
            'CacheManager': 'UnifiedUtils()',
            'ParallelProcessor': 'UnifiedUtils()',
            'AIEnhancedSVGConverter': 'AIEnhancedConverter',
        }

    def find_broken_test_files(self) -> List[Path]:
        """Find all test files with import errors"""
        test_files = list(Path('tests').rglob('*.py'))
        broken_files = []

        for test_file in test_files:
            try:
                content = test_file.read_text()
                # Check for problematic import patterns
                if any(old_import in content for old_import in self.import_mappings.keys()):
                    broken_files.append(test_file)
            except Exception:
                continue

        return broken_files

    def fix_imports_in_file(self, file_path: Path) -> bool:
        """Fix imports in a single test file"""
        try:
            content = file_path.read_text()
            original_content = content

            # Fix import statements
            for old_import, new_import in self.import_mappings.items():
                content = content.replace(old_import, new_import)

            # Fix class instantiations (more conservative approach)
            for old_class, new_class in self.class_replacements.items():
                # Only replace if it's a clear instantiation pattern
                pattern = rf'\b{re.escape(old_class)}\s*\('
                if re.search(pattern, content):
                    content = re.sub(pattern, f'{new_class}(', content)

            if content != original_content:
                # Create backup
                backup_path = file_path.with_suffix('.py.backup')
                backup_path.write_text(original_content)

                # Write fixed content
                file_path.write_text(content)
                print(f"✅ Fixed imports in {file_path}")
                return True
            else:
                print(f"⚪ No changes needed in {file_path}")
                return False

        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
            return False

    def fix_all_test_imports(self) -> Tuple[int, int]:
        """Fix imports in all broken test files"""
        broken_files = self.find_broken_test_files()
        fixed_count = 0

        print(f"📋 Found {len(broken_files)} test files with import issues")

        for test_file in broken_files:
            if self.fix_imports_in_file(test_file):
                fixed_count += 1

        return fixed_count, len(broken_files)

    def cleanup_pycache(self):
        """Clean up __pycache__ directories to force reimport"""
        import shutil

        pycache_dirs = list(Path('.').rglob('__pycache__'))
        for pycache_dir in pycache_dirs:
            try:
                shutil.rmtree(pycache_dir)
                print(f"🧹 Cleaned {pycache_dir}")
            except Exception:
                pass


def main():
    """Main function to fix test imports"""
    print("🔧 Fixing Test Imports for 80% Coverage Target")
    print("=" * 60)

    fixer = TestImportFixer()

    # Clean up caches first
    print("\n🧹 Cleaning __pycache__ directories...")
    fixer.cleanup_pycache()

    # Fix imports
    print("\n🔧 Fixing import statements...")
    fixed_count, total_count = fixer.fix_all_test_imports()

    print(f"\n📊 RESULTS:")
    print(f"Fixed: {fixed_count}/{total_count} files")
    print(f"Backup files created with .backup extension")

    if fixed_count > 0:
        print("\n✅ Import fixes applied successfully!")
        print("💡 Next step: Run coverage analysis to check improvement")
    else:
        print("\n⚠️ No files were fixed - may need manual intervention")


if __name__ == "__main__":
    main()