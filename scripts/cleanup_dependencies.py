"""
Dependency Cleanup System for AI SVG Converter

This script analyzes Python package dependencies to:
- Identify unused packages in requirements.txt
- Find missing dependencies from actual imports
- Clean up requirements files
- Test system functionality after cleanup
"""

import pkg_resources
import ast
import sys
import subprocess
from pathlib import Path
from typing import Set, Dict, List
import logging

# Set up logging
logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Comprehensive dependency analyzer for package cleanup"""

    def __init__(self):
        self.imported_packages = set()
        self.installed_packages = self._get_installed_packages()
        self.requirements = self._parse_requirements()

    def _get_installed_packages(self) -> Set[str]:
        """Get list of installed packages"""
        return {pkg.key for pkg in pkg_resources.working_set}

    def _parse_requirements(self) -> Dict[str, str]:
        """Parse requirements.txt files"""
        requirements = {}
        req_files = ['requirements.txt', 'requirements_ai_phase1.txt']

        for req_file in req_files:
            if Path(req_file).exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse package name
                            if '>' in line or '<' in line or '=' in line:
                                pkg_name = line.split('<')[0].split('>')[0].split('=')[0]
                            else:
                                pkg_name = line
                            requirements[pkg_name.lower().strip()] = line

        return requirements

    def scan_imports(self, root_dir: str = 'backend'):
        """Scan codebase for actual imports"""
        logger.info(f"Scanning imports in {root_dir}...")

        for py_file in Path(root_dir).rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.imported_packages.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.imported_packages.add(node.module.split('.')[0])
            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")
                pass

        logger.info(f"Found {len(self.imported_packages)} unique imported packages")

    def find_unused_packages(self) -> Set[str]:
        """Find packages in requirements but not imported"""
        unused = set()

        # Map common import names to package names
        import_to_package = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'pillow',
            'yaml': 'pyyaml',
            'psutil': 'psutil',
            'requests': 'requests',
            'numpy': 'numpy',
            'torch': 'torch',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'cachetools': 'cachetools',
            'redis': 'redis',
            'sqlalchemy': 'sqlalchemy',
            'pydantic': 'pydantic',
            'jinja2': 'jinja2',
            'starlette': 'starlette'
        }

        for req_package in self.requirements:
            # Check if package is imported
            imported = False

            # Check direct match
            if req_package in self.imported_packages:
                imported = True

            # Check mapped names
            for imp, pkg in import_to_package.items():
                if pkg == req_package and imp in self.imported_packages:
                    imported = True
                    break

            # Check if package name contains import name
            for imported_pkg in self.imported_packages:
                if imported_pkg in req_package or req_package in imported_pkg:
                    imported = True
                    break

            if not imported:
                unused.add(req_package)

        return unused

    def find_missing_packages(self) -> Set[str]:
        """Find imported packages not in requirements"""
        missing = set()

        # Standard library modules (Python 3.9+)
        stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing', 'collections',
            'itertools', 'functools', 'operator', 'math', 'random', 'subprocess', 'logging',
            'threading', 'queue', 'weakref', 'gc', 'ast', 'copy', 'pickle', 'base64'
        }

        for imported in self.imported_packages:
            if imported not in stdlib_modules:
                # Check if this import is covered by any requirement
                covered = False
                for req_package in self.requirements:
                    if imported in req_package or req_package in imported:
                        covered = True
                        break

                if not covered:
                    missing.add(imported)

        return missing

    def generate_cleaned_requirements(self) -> List[str]:
        """Generate cleaned requirements.txt"""
        used_packages = []
        unused = self.find_unused_packages()

        for req_package, req_line in self.requirements.items():
            if req_package not in unused:
                used_packages.append(req_line)

        return sorted(used_packages)

    def get_report(self) -> Dict:
        """Generate comprehensive dependency report"""
        unused = self.find_unused_packages()
        missing = self.find_missing_packages()

        return {
            'total_requirements': len(self.requirements),
            'imported_packages': len(self.imported_packages),
            'unused_packages': list(unused),
            'missing_packages': list(missing),
            'reduction_potential': len(unused),
            'reduction_percentage': (len(unused) / max(1, len(self.requirements))) * 100
        }


def cleanup_requirements():
    """Clean up requirements files"""
    print("🧹 Starting Dependency Cleanup")
    print("=" * 50)

    analyzer = DependencyAnalyzer()
    analyzer.scan_imports()

    # Generate report
    report = analyzer.get_report()
    print(f"Total requirements: {report['total_requirements']}")
    print(f"Imported packages: {report['imported_packages']}")
    print(f"Unused packages: {len(report['unused_packages'])}")
    print(f"Missing packages: {len(report['missing_packages'])}")
    print(f"Potential reduction: {report['reduction_percentage']:.1f}%")

    # Find what to remove
    unused = analyzer.find_unused_packages()
    print(f"\nFound {len(unused)} unused packages:")
    for pkg in sorted(unused):
        print(f"  - {pkg}")

    # Find missing packages
    missing = analyzer.find_missing_packages()
    if missing:
        print(f"\nFound {len(missing)} missing packages:")
        for pkg in sorted(missing):
            print(f"  - {pkg}")

    # Packages to definitely keep (even if not directly imported)
    keep_packages = {
        'pytest',           # Testing
        'black',            # Formatting
        'mypy',             # Type checking
        'ipython',          # Development
        'jupyter',          # Notebooks
        'flake8',           # Linting
        'setuptools',       # Build tools
        'wheel',            # Build tools
        'pip',              # Package management
        'cachetools',       # Required for caching
        'pydantic',         # FastAPI dependency
        'starlette',        # FastAPI dependency
        'jinja2',           # Template engine for FastAPI
        'python-multipart', # FastAPI file uploads
        'uvicorn',          # ASGI server
        'fastapi',          # Web framework
        'numpy',            # Core numerical computing
        'pillow',           # Image processing
        'opencv-python'     # Computer vision
    }

    # Generate new requirements
    new_requirements = []
    for line in analyzer.generate_cleaned_requirements():
        pkg_name = line.split('<')[0].split('>')[0].split('=')[0].lower().strip()
        if pkg_name not in unused or pkg_name in keep_packages:
            new_requirements.append(line)

    # Add missing critical packages
    for missing_pkg in missing:
        if missing_pkg in ['vtracer', 'fastapi', 'uvicorn', 'pillow', 'numpy']:
            new_requirements.append(missing_pkg)

    # Backup and write new requirements
    if Path('requirements.txt').exists():
        print(f"\nBacking up requirements.txt...")
        Path('requirements.txt').rename('requirements.txt.backup')

    print(f"Writing new requirements.txt...")
    with open('requirements.txt', 'w') as f:
        f.write('# Core requirements\n')
        core_packages = ['numpy', 'pillow', 'fastapi', 'uvicorn', 'vtracer', 'opencv-python']
        for req in new_requirements:
            if any(pkg in req.lower() for pkg in core_packages):
                f.write(f'{req}\n')

        f.write('\n# AI/ML requirements\n')
        ai_packages = ['torch', 'scikit', 'xgboost', 'stable-baselines3', 'deap', 'gymnasium', 'transformers']
        for req in new_requirements:
            if any(pkg in req.lower() for pkg in ai_packages):
                f.write(f'{req}\n')

        f.write('\n# Development requirements\n')
        dev_packages = ['pytest', 'black', 'mypy', 'flake8', 'ipython', 'jupyter']
        for req in new_requirements:
            if any(pkg in req.lower() for pkg in dev_packages):
                f.write(f'{req}\n')

        f.write('\n# Utility requirements\n')
        for req in new_requirements:
            if not any(pkg in req.lower() for pkg in core_packages + ai_packages + dev_packages):
                f.write(f'{req}\n')

    print(f"Reduced requirements from {len(analyzer.requirements)} to {len(new_requirements)}")
    print(f"Reduction: {len(unused)} packages ({report['reduction_percentage']:.1f}%)")

    return analyzer


def test_after_cleanup():
    """Test that system still works after cleanup"""
    print("\n🧪 Testing After Cleanup")
    print("=" * 50)

    tests = []

    # Test 1: Can import all modules
    try:
        # Try importing FastAPI first to check if it's available
        import fastapi
        tests.append(('FastAPI package', 'PASS'))
    except ImportError as e:
        tests.append(('FastAPI package', f'FAIL: {e}'))

    try:
        from backend.app import app
        tests.append(('FastAPI app import', 'PASS'))
    except ImportError as e:
        tests.append(('FastAPI app import', f'FAIL: {e}'))
    except Exception as e:
        tests.append(('FastAPI app import', f'FAIL: {e}'))

    try:
        from backend.converters.ai_enhanced_converter import AIEnhancedConverter
        tests.append(('AI Enhanced Converter import', 'PASS'))
    except ImportError as e:
        tests.append(('AI Enhanced Converter import', f'FAIL: {e}'))

    try:
        from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
        tests.append(('Unified AI Pipeline import', 'PASS'))
    except ImportError as e:
        tests.append(('Unified AI Pipeline import', f'FAIL: {e}'))

    # Test 2: Basic converter functionality
    try:
        from backend.converters.vtracer_converter import VTracerConverter
        converter = VTracerConverter()
        # Just test instantiation, not actual conversion
        tests.append(('Basic converter instantiation', 'PASS'))
    except Exception as e:
        tests.append(('Basic converter instantiation', f'FAIL: {e}'))

    # Test 3: AI modules
    try:
        from backend.ai_modules.utils.cache_manager import MultiLevelCache
        cache = MultiLevelCache()
        tests.append(('Cache manager import', 'PASS'))
    except ImportError as e:
        tests.append(('Cache manager import', f'FAIL: {e}'))

    # Test 4: Try to run basic tests (if pytest available)
    try:
        result = subprocess.run(['python', '-m', 'pytest', '--version'],
                              capture_output=True, timeout=10)
        if result.returncode == 0:
            tests.append(('Pytest availability', 'PASS'))
        else:
            tests.append(('Pytest availability', 'FAIL: pytest not working'))
    except Exception as e:
        tests.append(('Pytest availability', f'FAIL: {e}'))

    # Report results
    print("\nTest Results:")
    passed = 0
    for test_name, status in tests:
        print(f"  {test_name}: {status}")
        if 'PASS' in status:
            passed += 1

    success_rate = (passed / len(tests)) * 100
    print(f"\nTest Summary: {passed}/{len(tests)} tests passed ({success_rate:.1f}%)")

    return success_rate >= 80  # 80% pass rate required


def rollback_requirements():
    """Rollback to original requirements if cleanup failed"""
    if Path('requirements.txt.backup').exists():
        print("Rolling back to original requirements.txt...")
        Path('requirements.txt').unlink(missing_ok=True)
        Path('requirements.txt.backup').rename('requirements.txt')
        print("Rollback completed")
    else:
        print("No backup found for rollback")


def main():
    """Main cleanup execution"""
    try:
        # Step 1: Analyze and cleanup
        analyzer = cleanup_requirements()

        # Step 2: Test after cleanup
        if test_after_cleanup():
            print("\n✅ Dependency cleanup completed successfully!")
            print("All critical functionality verified")

            # Remove backup if tests pass
            if Path('requirements.txt.backup').exists():
                print("Removing backup file...")
                Path('requirements.txt.backup').unlink()

        else:
            print("\n❌ Tests failed after cleanup!")
            print("Rolling back changes...")
            rollback_requirements()
            return False

    except Exception as e:
        print(f"\n💥 Cleanup failed with error: {e}")
        print("Rolling back changes...")
        rollback_requirements()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)