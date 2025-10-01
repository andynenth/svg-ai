#!/usr/bin/env python3
"""
Final Structure Setup for Day 13
Creates final directory structure and updates configuration files
"""

from pathlib import Path
import os
import shutil


def create_final_structure():
    """Create the final ~15 file structure"""

    # Define final structure
    final_structure = {
        'backend/': {
            'app.py': None,  # Keep existing
            'api/': {
                'ai_endpoints.py': None  # Keep existing
            },
            'converters/': {
                'ai_enhanced_converter.py': None  # Keep existing
            },
            'ai_modules/': {
                'classification.py': 'Merged classification module',
                'optimization.py': 'Merged optimization module',
                'quality.py': 'Merged quality module',
                'pipeline.py': 'Keep existing unified pipeline',
                'utils.py': 'Merged utilities'
            }
        },
        'scripts/': {
            'train_models.py': 'Unified training script',
            'benchmark.py': 'Performance benchmarking',
            'validate.py': 'Validation script'
        },
        'tests/': {
            'test_integration.py': 'All integration tests',
            'test_models.py': 'All model tests',
            'test_api.py': 'All API tests'
        }
    }

    print("📁 Creating final directory structure...")

    # Clean up empty directories
    backend_path = Path('backend')
    for root, dirs, files in os.walk(backend_path, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            if dir_path.name == '__pycache__':
                continue  # Skip pycache directories

            # Check if directory is empty (except for __pycache__)
            contents = [f for f in dir_path.iterdir() if f.name != '__pycache__']
            if not contents:
                try:
                    dir_path.rmdir()
                    print(f"Removed empty directory: {dir_path}")
                except OSError as e:
                    print(f"Could not remove {dir_path}: {e}")

    # Verify structure
    essential_files = [
        'backend/app.py',
        'backend/api/ai_endpoints.py',
        'backend/converters/ai_enhanced_converter.py',
        'backend/ai_modules/classification.py',
        'backend/ai_modules/optimization.py',
        'backend/ai_modules/quality.py',
        'backend/ai_modules/utils.py',
        'scripts/train_models.py'
    ]

    actual_files = []
    for file_path in essential_files:
        if Path(file_path).exists():
            actual_files.append(file_path)

    print(f"\n📊 Structure verification:")
    print(f"Essential files found: {len(actual_files)}")
    print(f"Target: ~15 files")

    for file_path in essential_files:
        exists = "✓" if Path(file_path).exists() else "✗"
        print(f"  {exists} {file_path}")

    # Count all Python files in backend
    all_backend_files = list(Path('backend').rglob('*.py'))
    backend_count = len([f for f in all_backend_files if '__pycache__' not in str(f)])

    print(f"\nTotal backend Python files: {backend_count}")

    if backend_count > 25:
        print("⚠️  Still more files than target!")
        print("Extra files that could be considered for removal:")
        for f in all_backend_files:
            if '__pycache__' not in str(f) and str(f) not in essential_files:
                print(f"  - {f}")

    return backend_count


def update_configurations():
    """Update configuration files for new structure"""

    print("🔧 Updating configuration files...")

    # Update main __init__.py
    backend_init = Path('backend/__init__.py')
    init_content = '''"""AI-enhanced SVG conversion system"""

__version__ = "2.0.0"

# Public API
from .ai_modules.classification import ClassificationModule
from .ai_modules.optimization import OptimizationEngine
from .ai_modules.quality import QualitySystem
from .ai_modules.utils import UnifiedUtils

try:
    from .ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
except ImportError:
    # Fallback if pipeline not available
    UnifiedAIPipeline = None

__all__ = [
    "ClassificationModule",
    "OptimizationEngine",
    "QualitySystem",
    "UnifiedUtils",
    "UnifiedAIPipeline"
]
'''

    backend_init.write_text(init_content)
    print(f"✓ Updated: {backend_init}")

    # Update ai_modules __init__.py
    ai_modules_init = Path('backend/ai_modules/__init__.py')
    ai_modules_content = '''"""AI modules for SVG conversion"""

from .classification import ClassificationModule
from .optimization import OptimizationEngine
from .quality import QualitySystem
from .utils import UnifiedUtils

__all__ = [
    "ClassificationModule",
    "OptimizationEngine",
    "QualitySystem",
    "UnifiedUtils"
]
'''

    ai_modules_init.write_text(ai_modules_content)
    print(f"✓ Updated: {ai_modules_init}")

    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "from backend.ai_modules.classification import ClassificationModule",
        "from backend.ai_modules.optimization import OptimizationEngine",
        "from backend.ai_modules.quality import QualitySystem",
        "from backend.ai_modules.utils import UnifiedUtils"
    ]

    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"✓ {import_stmt}")
        except Exception as e:
            print(f"✗ {import_stmt} - Error: {e}")

    return True


def generate_structure_report():
    """Generate final structure report"""

    print("\n📄 Generating structure report...")

    report = {
        'timestamp': '2025-09-30',
        'final_structure': {},
        'file_counts': {},
        'consolidations': []
    }

    # Count files by type
    backend_files = list(Path('backend').rglob('*.py'))
    backend_files = [f for f in backend_files if '__pycache__' not in str(f)]

    script_files = list(Path('scripts').glob('*.py'))
    test_files = list(Path('tests').glob('*.py')) if Path('tests').exists() else []

    report['file_counts'] = {
        'backend_modules': len(backend_files),
        'scripts': len(script_files),
        'tests': len(test_files),
        'total': len(backend_files) + len(script_files) + len(test_files)
    }

    # Document consolidations
    report['consolidations'] = [
        {
            'target': 'backend/ai_modules/classification.py',
            'consolidated_from': ['classification/', 'feature_extractor.py', 'logo_classifier.py'],
            'description': 'Unified classification with feature extraction and neural networks'
        },
        {
            'target': 'backend/ai_modules/optimization.py',
            'consolidated_from': ['optimization/', 'parameter_formulas.py', 'learned_correlations.py'],
            'description': 'Complete parameter optimization with ML and correlation analysis'
        },
        {
            'target': 'backend/ai_modules/quality.py',
            'consolidated_from': ['quality/', 'enhanced_metrics.py', 'ab_testing.py'],
            'description': 'Quality measurement and A/B testing system'
        },
        {
            'target': 'backend/ai_modules/utils.py',
            'consolidated_from': ['utils/', 'cache_manager.py', 'parallel_processor.py'],
            'description': 'Unified utilities for caching and parallel processing'
        },
        {
            'target': 'scripts/train_models.py',
            'consolidated_from': ['scripts/train_*.py'],
            'description': 'Unified training script for all AI models'
        }
    ]

    # Save report
    import json
    with open('final_structure_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Structure report saved to: final_structure_report.json")

    # Print summary
    print(f"\n📈 Final Structure Summary:")
    print(f"  Backend modules: {report['file_counts']['backend_modules']}")
    print(f"  Scripts: {report['file_counts']['scripts']}")
    print(f"  Tests: {report['file_counts']['tests']}")
    print(f"  Total files: {report['file_counts']['total']}")
    print(f"  Target achieved: {'Yes' if report['file_counts']['total'] <= 20 else 'Partially'}")

    return report


def main():
    """Execute final structure setup"""
    print("🏗️  Final Structure Setup - Day 13")
    print("=" * 50)

    # Create final structure
    file_count = create_final_structure()

    # Update configurations
    update_configurations()

    # Generate report
    report = generate_structure_report()

    print(f"\n🎉 Final structure setup completed!")
    print(f"Total files: {report['file_counts']['total']}")


if __name__ == "__main__":
    main()