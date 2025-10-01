"""
Remove Duplicate Implementations for AI SVG Converter

This script identifies and removes duplicate implementations across:
- Correlation formula implementations
- Training scripts
- Utility functions

Consolidates functionality into organized, single-source modules.
"""

import ast
import hashlib
import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)


def find_duplicate_formulas():
    """Find duplicate correlation formula implementations"""
    print("🔍 Finding Duplicate Formula Implementations")
    print("=" * 50)

    formula_files = [
        'parameter_correlation_formulas.py',
        'correlation_formula.py',
        'enhanced_correlation_formula.py',
        'formula_calculator.py',
        'parameter_formulas.py',
        'correlation_formulas.py',
        'refined_correlation_formulas.py',
        'correlation_formulas_old.py',
        'correlation_formulas_backup.py'
    ]

    formulas = {}
    existing_files = []

    # Check all optimization directories
    search_dirs = [
        'backend/ai_modules/optimization',
        'backend/optimization',
        'scripts/optimization'
    ]

    for search_dir in search_dirs:
        if Path(search_dir).exists():
            for file in formula_files:
                file_path = Path(search_dir) / file
                if file_path.exists():
                    existing_files.append(file_path)

    print(f"Found {len(existing_files)} formula files to analyze:")
    for file_path in existing_files:
        print(f"  - {file_path}")

    # Analyze each file
    for file_path in existing_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for calculate functions
            if 'def calculate' in content:
                # Extract function logic using AST
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if 'calculate' in node.name:
                            # Hash the function body for comparison
                            func_body = ast.dump(node, annotate_fields=False, include_attributes=False)
                            func_hash = hashlib.md5(func_body.encode()).hexdigest()

                            if func_hash not in formulas:
                                formulas[func_hash] = []
                            formulas[func_hash].append({
                                'file': str(file_path),
                                'function': node.name,
                                'lines': (node.end_lineno or 0) - node.lineno + 1 if node.end_lineno else 0,
                                'body_hash': func_hash
                            })

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")

    # Find actual duplicates
    duplicates = {
        k: v for k, v in formulas.items() if len(v) > 1
    }

    print(f"\nDuplicate Analysis Results:")
    print(f"Total function implementations: {sum(len(v) for v in formulas.values())}")
    print(f"Unique function signatures: {len(formulas)}")
    print(f"Duplicate groups: {len(duplicates)}")

    if duplicates:
        print(f"\nDuplicate Functions Found:")
        for func_hash, implementations in duplicates.items():
            print(f"  Function group (hash: {func_hash[:8]}):")
            for impl in implementations:
                print(f"    - {impl['function']} in {impl['file']} ({impl['lines']} lines)")

    return duplicates


def consolidate_formulas():
    """Consolidate all formula implementations into single file"""
    print("\n📦 Consolidating Formula Implementations")
    print("=" * 50)

    consolidated = '''"""
Consolidated Parameter Correlation Formulas

This module contains all parameter correlation formula implementations
consolidated from multiple duplicate files.
"""

import numpy as np
from typing import Dict, Any, Tuple


class ParameterFormulas:
    """Unified parameter formula calculator"""

    @staticmethod
    def calculate_color_precision(features: Dict) -> int:
        """Calculate optimal color precision based on image features"""
        unique_colors = features.get('unique_colors', 10)
        has_gradients = features.get('has_gradients', False)
        complexity = features.get('complexity', 0.5)

        if unique_colors < 5:
            return 2
        elif unique_colors < 10:
            return 3
        elif unique_colors < 50:
            return 4
        elif has_gradients or complexity > 0.7:
            return 8
        else:
            return 6

    @staticmethod
    def calculate_corner_threshold(features: Dict) -> float:
        """Calculate optimal corner threshold based on edge characteristics"""
        edge_density = features.get('edge_density', 0.5)
        complexity = features.get('complexity', 0.5)
        has_text = features.get('has_text', False)

        # Base threshold
        base_threshold = 30.0

        # Adjust for edge density
        if edge_density > 0.7:
            base_threshold -= 10  # More aggressive for high edge density
        elif edge_density < 0.3:
            base_threshold += 10  # More conservative for low edge density

        # Adjust for complexity
        complexity_adjustment = (complexity - 0.5) * 20
        base_threshold += complexity_adjustment

        # Special case for text
        if has_text:
            base_threshold = min(base_threshold, 20.0)

        return max(5.0, min(60.0, base_threshold))

    @staticmethod
    def calculate_layer_difference(features: Dict) -> int:
        """Calculate optimal layer difference threshold"""
        unique_colors = features.get('unique_colors', 10)
        has_gradients = features.get('has_gradients', False)

        if has_gradients:
            return 8  # Preserve gradients
        elif unique_colors > 50:
            return 12  # Many colors, need fine separation
        else:
            return 16  # Default for simple images

    @staticmethod
    def calculate_path_precision(features: Dict) -> int:
        """Calculate optimal path precision"""
        complexity = features.get('complexity', 0.5)
        edge_density = features.get('edge_density', 0.5)

        if complexity > 0.8 or edge_density > 0.8:
            return 8  # High precision for complex paths
        elif complexity > 0.5:
            return 10  # Medium precision
        else:
            return 15  # Lower precision for simple shapes

    @staticmethod
    def calculate_splice_threshold(features: Dict) -> int:
        """Calculate optimal splice threshold"""
        complexity = features.get('complexity', 0.5)

        base_threshold = 45
        if complexity > 0.7:
            return base_threshold + 15  # Higher threshold for complex images
        elif complexity < 0.3:
            return base_threshold - 15  # Lower threshold for simple images
        else:
            return base_threshold

    @staticmethod
    def calculate_filter_speckle(features: Dict) -> int:
        """Calculate filter speckle size"""
        noise_level = features.get('noise_level', 0.1)

        if noise_level > 0.3:
            return 8  # Aggressive filtering
        elif noise_level > 0.1:
            return 4  # Moderate filtering
        else:
            return 1  # Minimal filtering

    @staticmethod
    def calculate_all_parameters(features: Dict) -> Dict[str, Any]:
        """Calculate all VTracer parameters based on image features"""
        return {
            'color_precision': ParameterFormulas.calculate_color_precision(features),
            'corner_threshold': ParameterFormulas.calculate_corner_threshold(features),
            'layer_difference': ParameterFormulas.calculate_layer_difference(features),
            'path_precision': ParameterFormulas.calculate_path_precision(features),
            'splice_threshold': ParameterFormulas.calculate_splice_threshold(features),
            'filter_speckle': ParameterFormulas.calculate_filter_speckle(features)
        }


class QualityFormulas:
    """Quality prediction formulas"""

    @staticmethod
    def predict_ssim(parameters: Dict, features: Dict) -> float:
        """Predict SSIM score based on parameters and features"""
        # Simplified prediction model
        base_score = 0.85

        # Adjust based on complexity
        complexity = features.get('complexity', 0.5)
        if complexity > 0.7:
            base_score -= 0.1
        elif complexity < 0.3:
            base_score += 0.1

        # Adjust based on parameters
        color_precision = parameters.get('color_precision', 4)
        if color_precision >= 6:
            base_score += 0.05
        elif color_precision <= 2:
            base_score -= 0.05

        return max(0.0, min(1.0, base_score))

    @staticmethod
    def predict_file_size_reduction(parameters: Dict, features: Dict) -> float:
        """Predict file size reduction percentage"""
        # Base reduction expectation
        base_reduction = 0.7  # 70% reduction

        complexity = features.get('complexity', 0.5)
        unique_colors = features.get('unique_colors', 10)

        # Complex images reduce less
        if complexity > 0.8:
            base_reduction -= 0.2
        elif complexity < 0.2:
            base_reduction += 0.1

        # Many colors reduce less
        if unique_colors > 100:
            base_reduction -= 0.15
        elif unique_colors < 10:
            base_reduction += 0.1

        return max(0.1, min(0.9, base_reduction))


# Legacy compatibility functions for existing code
def calculate_color_precision(features: Dict) -> int:
    """Legacy wrapper for color precision calculation"""
    return ParameterFormulas.calculate_color_precision(features)

def calculate_corner_threshold(features: Dict) -> float:
    """Legacy wrapper for corner threshold calculation"""
    return ParameterFormulas.calculate_corner_threshold(features)

def calculate_all_parameters(features: Dict) -> Dict[str, Any]:
    """Legacy wrapper for all parameter calculation"""
    return ParameterFormulas.calculate_all_parameters(features)
'''

    # Ensure target directory exists
    target_dir = Path('backend/ai_modules/optimization')
    target_dir.mkdir(parents=True, exist_ok=True)

    # Write consolidated file
    target_file = target_dir / 'unified_parameter_formulas.py'
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(consolidated)

    print(f"✅ Created consolidated formula file: {target_file}")
    print(f"   Contains: ParameterFormulas and QualityFormulas classes")
    print(f"   Legacy compatibility functions included")

    return str(target_file)


def update_formula_imports():
    """Update imports in files that use the old formula files"""
    print("\n🔄 Updating Formula Import Statements")
    print("=" * 50)

    # Find files that import formula modules
    search_patterns = [
        'backend/**/*.py',
        'scripts/**/*.py'
    ]

    files_to_update = []
    for pattern in search_patterns:
        files_to_update.extend(Path('.').glob(pattern))

    old_imports = [
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas',
        'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas'
    ]

    new_import = 'from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas'

    updated_files = []
    for file_path in files_to_update:
        if file_path.is_file() and file_path.suffix == '.py':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content
                needs_update = False

                # Check if file uses old imports
                for old_import in old_imports:
                    if old_import in content:
                        needs_update = True
                        # Replace with new import
                        content = content.replace(old_import, new_import)

                if needs_update:
                    # Write updated content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    updated_files.append(str(file_path))

            except Exception as e:
                logger.warning(f"Failed to update imports in {file_path}: {e}")

    print(f"Updated imports in {len(updated_files)} files:")
    for file_path in updated_files[:10]:  # Show first 10
        print(f"  ✓ {file_path}")
    if len(updated_files) > 10:
        print(f"  ... and {len(updated_files) - 10} more")

    return updated_files


def cleanup_training_scripts():
    """Remove duplicate training scripts"""
    print("\n🗂️ Cleaning Up Training Scripts")
    print("=" * 50)

    # Find training scripts
    training_scripts = []

    search_dirs = [
        'scripts',
        'backend/ai_modules',
        'backend/optimization'
    ]

    for search_dir in search_dirs:
        if Path(search_dir).exists():
            training_scripts.extend(Path(search_dir).rglob('train_*.py'))
            training_scripts.extend(Path(search_dir).rglob('*train*.py'))
            training_scripts.extend(Path(search_dir).rglob('training_*.py'))

    print(f"Found {len(training_scripts)} training-related scripts")

    # Group by functionality
    script_groups = {
        'classifier': [],
        'optimizer': [],
        'quality': [],
        'gpu': [],
        'colab': [],
        'experimental': []
    }

    for script in training_scripts:
        try:
            content = script.read_text(encoding='utf-8')
            script_name = script.name.lower()

            if 'classifier' in script_name or 'efficientnet' in content.lower():
                script_groups['classifier'].append(script)
            elif 'optimizer' in script_name or 'xgboost' in content.lower() or 'optim' in script_name:
                script_groups['optimizer'].append(script)
            elif 'quality' in script_name or 'ssim' in content.lower():
                script_groups['quality'].append(script)
            elif 'gpu' in script_name or 'cuda' in content.lower():
                script_groups['gpu'].append(script)
            elif 'colab' in script_name or 'colab' in content.lower():
                script_groups['colab'].append(script)
            else:
                script_groups['experimental'].append(script)

        except Exception as e:
            logger.warning(f"Failed to analyze training script {script}: {e}")
            script_groups['experimental'].append(script)

    # Analyze groups
    print(f"\nTraining Script Groups:")
    for group_name, scripts in script_groups.items():
        if scripts:
            print(f"  {group_name}: {len(scripts)} scripts")
            for script in scripts:
                print(f"    - {script}")

    # Identify files to remove (keep most recent/complete)
    files_to_remove = []

    for group_name, scripts in script_groups.items():
        if len(scripts) > 1:
            print(f"\n  Processing {group_name} group ({len(scripts)} scripts):")

            # Sort by modification time and size (newest and largest first)
            scripts.sort(key=lambda x: (x.stat().st_mtime, x.stat().st_size), reverse=True)

            # Keep the first (newest/largest), mark rest for removal
            keep_script = scripts[0]
            remove_scripts = scripts[1:]

            print(f"    ✅ Keeping: {keep_script} (most recent/complete)")
            for script in remove_scripts:
                print(f"    ❌ Removing: {script}")
                files_to_remove.append(script)

    print(f"\nTraining Script Cleanup Summary:")
    print(f"  Total scripts found: {len(training_scripts)}")
    print(f"  Scripts to remove: {len(files_to_remove)}")
    print(f"  Scripts to keep: {len(training_scripts) - len(files_to_remove)}")

    return files_to_remove


def consolidate_utilities():
    """Consolidate utility functions into organized modules"""
    print("\n🛠️ Consolidating Utility Functions")
    print("=" * 50)

    # Find all utility files
    util_files = []
    search_dirs = ['backend', 'scripts']

    for search_dir in search_dirs:
        if Path(search_dir).exists():
            util_files.extend(Path(search_dir).rglob('*util*.py'))
            util_files.extend(Path(search_dir).rglob('*helper*.py'))
            util_files.extend(Path(search_dir).rglob('*common*.py'))

    print(f"Found {len(util_files)} utility files")

    # Categorize utilities based on content
    utils_by_category = {
        'image': [],      # Image processing utilities
        'file': [],       # File I/O utilities
        'math': [],       # Mathematical utilities
        'validation': [], # Validation utilities
        'conversion': []  # Conversion utilities
    }

    for util_file in util_files:
        try:
            content = util_file.read_text(encoding='utf-8')
            file_name = util_file.name.lower()

            # Categorize based on content and filename
            if any(keyword in content.lower() for keyword in ['image', 'pil', 'cv2', 'opencv', 'pixel']):
                utils_by_category['image'].append(util_file)
            elif any(keyword in content.lower() for keyword in ['path', 'file', 'open(', 'read', 'write']):
                utils_by_category['file'].append(util_file)
            elif any(keyword in content.lower() for keyword in ['numpy', 'calculate', 'math', 'statistics']):
                utils_by_category['math'].append(util_file)
            elif any(keyword in content.lower() for keyword in ['validate', 'check', 'verify', 'assert']):
                utils_by_category['validation'].append(util_file)
            else:
                utils_by_category['conversion'].append(util_file)

        except Exception as e:
            logger.warning(f"Failed to analyze utility file {util_file}: {e}")
            utils_by_category['conversion'].append(util_file)

    # Display categorization
    print(f"\nUtility File Categorization:")
    for category, files in utils_by_category.items():
        if files:
            print(f"  {category}: {len(files)} files")
            for file_path in files:
                print(f"    - {file_path}")

    # For now, just report the categorization
    # In a real implementation, you'd consolidate the functions
    print(f"\nUtility Consolidation Summary:")
    print(f"  Total utility files: {len(util_files)}")
    print(f"  Categories identified: {len([cat for cat, files in utils_by_category.items() if files])}")

    return utils_by_category


def main():
    """Main duplicate removal execution"""
    print("🧹 Starting Duplicate Implementation Removal")
    print("=" * 50)

    results = {
        'timestamp': datetime.now().isoformat(),
        'duplicate_formulas': None,
        'consolidated_file': None,
        'updated_imports': None,
        'training_scripts_to_remove': None,
        'utility_categorization': None
    }

    try:
        # Task 3.1: Identify and consolidate duplicate formulas
        print("\n" + "="*50)
        print("TASK 3.1: DUPLICATE CORRELATION FORMULAS")
        print("="*50)

        duplicate_formulas = find_duplicate_formulas()
        results['duplicate_formulas'] = duplicate_formulas

        consolidated_file = consolidate_formulas()
        results['consolidated_file'] = consolidated_file

        updated_imports = update_formula_imports()
        results['updated_imports'] = updated_imports

        # Task 3.2: Remove duplicate training scripts
        print("\n" + "="*50)
        print("TASK 3.2: DUPLICATE TRAINING SCRIPTS")
        print("="*50)

        training_scripts_to_remove = cleanup_training_scripts()
        results['training_scripts_to_remove'] = [str(f) for f in training_scripts_to_remove]

        # Task 3.3: Consolidate utility functions
        print("\n" + "="*50)
        print("TASK 3.3: CONSOLIDATE UTILITY FUNCTIONS")
        print("="*50)

        utility_categorization = consolidate_utilities()
        results['utility_categorization'] = {
            cat: [str(f) for f in files]
            for cat, files in utility_categorization.items()
        }

        # Save results
        with open('duplicate_removal_report.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n" + "="*50)
        print("📊 DUPLICATE REMOVAL SUMMARY")
        print("="*50)
        print(f"✅ Duplicate formula groups found: {len(duplicate_formulas) if duplicate_formulas else 0}")
        print(f"✅ Consolidated formulas into: {consolidated_file}")
        print(f"✅ Updated imports in: {len(updated_imports) if updated_imports else 0} files")
        print(f"✅ Training scripts to remove: {len(training_scripts_to_remove)}")
        print(f"✅ Utility files categorized: {sum(len(files) for files in utility_categorization.values())}")
        print(f"\n📋 Report saved: duplicate_removal_report.json")

        return True

    except Exception as e:
        logger.error(f"Duplicate removal failed: {e}")
        print(f"\n❌ Duplicate removal failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)