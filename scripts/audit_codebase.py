"""
File Usage Audit and Dependency Analysis for Code Cleanup

This script analyzes the codebase to identify:
- File dependencies and usage patterns
- Unused files with no incoming references
- Duplicate function implementations
- Circular dependencies
- Removal candidates for cleanup
"""

import ast
import os
import sys
import json
from pathlib import Path
from typing import Dict, Set, List, Optional
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Dependency graph analysis will be limited.")


class CodebaseAuditor:
    """Comprehensive codebase analyzer for file dependencies and cleanup"""

    def __init__(self, root_dir: str = 'backend'):
        self.root_dir = Path(root_dir)
        self.files = {}
        self.imports = {}
        if NETWORKX_AVAILABLE:
            self.usage_graph = nx.DiGraph()
        else:
            self.usage_graph = {}
        self.unused_files = set()
        self.duplicate_functions = {}

    def scan_codebase(self):
        """Scan all Python files in codebase"""
        print(f"Scanning codebase in {self.root_dir}...")

        for file_path in self.root_dir.rglob('*.py'):
            if '__pycache__' not in str(file_path):
                print(f"  Analyzing: {file_path}")
                self.files[str(file_path)] = self._analyze_file(file_path)

        print(f"Analyzed {len(self.files)} Python files")

    def _analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            analysis = {
                'path': str(file_path),
                'size': len(content),
                'lines': content.count('\n'),
                'imports': self._extract_imports(tree),
                'functions': self._extract_functions(tree),
                'classes': self._extract_classes(tree),
                'global_vars': self._extract_globals(tree),
                'docstring': ast.get_docstring(tree),
                'last_modified': os.path.getmtime(file_path)
            }

            return analysis
        except Exception as e:
            return {'path': str(file_path), 'error': str(e)}

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        """Extract all imports from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def _extract_functions(self, tree: ast.Module) -> List[Dict]:
        """Extract function definitions"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'lines': node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                    'docstring': ast.get_docstring(node)
                })
        return functions

    def _extract_classes(self, tree: ast.Module) -> List[Dict]:
        """Extract class definitions"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    'bases': [self._get_name(base) for base in node.bases]
                })
        return classes

    def _extract_globals(self, tree: ast.Module) -> List[str]:
        """Extract global variable names"""
        globals_vars = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        globals_vars.append(target.id)
        return globals_vars

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)

    def build_dependency_graph(self):
        """Build graph of file dependencies"""
        print("Building dependency graph...")

        for file_path, analysis in self.files.items():
            if NETWORKX_AVAILABLE:
                self.usage_graph.add_node(file_path)
            else:
                self.usage_graph[file_path] = []

            if 'imports' in analysis and not 'error' in analysis:
                for imp in analysis['imports']:
                    # Try to resolve import to file
                    imported_file = self._resolve_import(imp)
                    if imported_file and imported_file in self.files:
                        if NETWORKX_AVAILABLE:
                            self.usage_graph.add_edge(file_path, imported_file)
                        else:
                            self.usage_graph[file_path].append(imported_file)

    def _resolve_import(self, import_name: str) -> Optional[str]:
        """Try to resolve an import to a file path"""
        # Simple resolution - try to map import to file
        parts = import_name.split('.')

        # Try different combinations
        for i in range(len(parts)):
            potential_path = '/'.join(parts[:i+1]) + '.py'
            for file_path in self.files:
                if file_path.endswith(potential_path):
                    return file_path

        return None

    def find_unused_files(self) -> Set[str]:
        """Find files with no incoming dependencies"""
        # Entry points that should never be marked as unused
        entry_points = {
            'backend/app.py',
            'backend/api/ai_endpoints.py',
            'web_server.py',
            # Add any additional entry points
        }

        unused = set()

        if NETWORKX_AVAILABLE:
            for file_path in self.files:
                if file_path not in entry_points:
                    in_degree = self.usage_graph.in_degree(file_path)
                    if in_degree == 0:
                        unused.add(file_path)
        else:
            # Simple analysis without networkx
            referenced_files = set()
            for file_path, deps in self.usage_graph.items():
                referenced_files.update(deps)

            for file_path in self.files:
                if file_path not in entry_points and file_path not in referenced_files:
                    unused.add(file_path)

        return unused

    def find_duplicate_code(self) -> Dict:
        """Find duplicate functions across files"""
        function_signatures = {}

        for file_path, analysis in self.files.items():
            if 'functions' in analysis and not 'error' in analysis:
                for func in analysis['functions']:
                    signature = f"{func['name']}({','.join(func['args'])})"
                    if signature not in function_signatures:
                        function_signatures[signature] = []
                    function_signatures[signature].append(file_path)

        # Find duplicates
        duplicates = {
            sig: files
            for sig, files in function_signatures.items()
            if len(files) > 1
        }

        return duplicates

    def generate_report(self) -> Dict:
        """Generate comprehensive audit report"""
        self.scan_codebase()
        self.build_dependency_graph()

        circular_deps = []
        if NETWORKX_AVAILABLE:
            try:
                circular_deps = list(nx.simple_cycles(self.usage_graph))
            except:
                circular_deps = []

        report = {
            'total_files': len(self.files),
            'total_lines': sum(f.get('lines', 0) for f in self.files.values() if 'error' not in f),
            'unused_files': list(self.find_unused_files()),
            'duplicate_functions': self.find_duplicate_code(),
            'largest_files': self._get_largest_files(10),
            'least_used_files': self._get_least_used_files(10),
            'circular_dependencies': circular_deps,
            'recommendations': self._generate_recommendations(),
            'error_files': [f['path'] for f in self.files.values() if 'error' in f]
        }

        return report

    def _get_largest_files(self, n: int) -> List[Dict]:
        """Get n largest files by line count"""
        sorted_files = sorted(
            [(path, data) for path, data in self.files.items() if 'error' not in data],
            key=lambda x: x[1].get('lines', 0),
            reverse=True
        )
        return [
            {'path': path, 'lines': data.get('lines', 0)}
            for path, data in sorted_files[:n]
        ]

    def _get_least_used_files(self, n: int) -> List[Dict]:
        """Get n least used files (approximation)"""
        # Simple heuristic: files with few imports or small size
        scored_files = []
        for path, data in self.files.items():
            if 'error' not in data:
                score = len(data.get('imports', [])) + data.get('lines', 0) / 100
                scored_files.append({'path': path, 'score': score, 'lines': data.get('lines', 0)})

        scored_files.sort(key=lambda x: x['score'])
        return scored_files[:n]

    def _generate_recommendations(self) -> List[str]:
        """Generate cleanup recommendations"""
        recommendations = []

        # Check for unused files
        unused = self.find_unused_files()
        if unused:
            recommendations.append(f"Remove {len(unused)} unused files")

        # Check for duplicates
        duplicates = self.find_duplicate_code()
        if duplicates:
            recommendations.append(f"Consolidate {len(duplicates)} duplicate functions")

        # Check for circular dependencies
        if NETWORKX_AVAILABLE:
            try:
                cycles = list(nx.simple_cycles(self.usage_graph))
                if cycles:
                    recommendations.append(f"Resolve {len(cycles)} circular dependencies")
            except:
                pass

        # Check for large files
        large_files = [f for f in self._get_largest_files(5) if f['lines'] > 500]
        if large_files:
            recommendations.append(f"Consider refactoring {len(large_files)} large files (>500 lines)")

        return recommendations


def analyze_optimization_files():
    """Specifically analyze optimization-related files"""
    print("\n=== Analyzing Optimization Files ===")

    auditor = CodebaseAuditor()

    # Target directories with optimization files
    optimization_dirs = [
        'backend/optimization',
        'backend/ai_modules/optimization',
        'scripts/optimization',
        'legacy/optimization'
    ]

    optimization_files = []
    for dir_path in optimization_dirs:
        if Path(dir_path).exists():
            for file in Path(dir_path).rglob('*.py'):
                optimization_files.append(file)

    # Also check for optimization files scattered elsewhere
    for py_file in Path('backend').rglob('*.py'):
        if 'optim' in py_file.name.lower() or 'param' in py_file.name.lower():
            if py_file not in optimization_files:
                optimization_files.append(py_file)

    print(f"Found {len(optimization_files)} optimization files")

    # Categorize files
    categories = {
        'genetic_algorithms': [],
        'reinforcement_learning': [],
        'parameter_tuning': [],
        'quality_metrics': [],
        'batch_processing': [],
        'correlation_formulas': [],
        'experimental': [],
        'utilities': []
    }

    for file in optimization_files:
        try:
            content = file.read_text(encoding='utf-8')
            file_name = file.name.lower()

            if 'genetic' in file_name or 'GA' in content or 'genetic' in content:
                categories['genetic_algorithms'].append(file)
            elif 'reinforcement' in file_name or 'RL' in content or 'reinforcement' in content:
                categories['reinforcement_learning'].append(file)
            elif 'param' in file_name or 'tune' in file_name or 'parameter' in content:
                categories['parameter_tuning'].append(file)
            elif 'quality' in file_name or 'metric' in file_name or 'ssim' in content.lower():
                categories['quality_metrics'].append(file)
            elif 'batch' in file_name or 'parallel' in file_name or 'concurrent' in content:
                categories['batch_processing'].append(file)
            elif 'correlation' in file_name or 'formula' in file_name or 'correlation' in content:
                categories['correlation_formulas'].append(file)
            elif 'test' in file_name or 'experiment' in file_name or 'demo' in file_name:
                categories['experimental'].append(file)
            else:
                categories['utilities'].append(file)
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
            categories['utilities'].append(file)

    # Print categorization results
    print("\nOptimization File Categories:")
    for category, files in categories.items():
        if files:
            print(f"  {category}: {len(files)} files")
            for f in files[:3]:  # Show first 3 files
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")

    return categories


def identify_removal_candidates(auditor: CodebaseAuditor) -> List[Dict]:
    """Identify files safe to remove"""
    print("\n=== Identifying Removal Candidates ===")

    candidates = []

    # Priority 1: Completely unused files
    unused = auditor.find_unused_files()
    print(f"Found {len(unused)} unused files")
    for file in unused:
        candidates.append({
            'file': file,
            'priority': 1,
            'reason': 'No imports or references found',
            'risk': 'low'
        })

    # Priority 2: Duplicate implementations
    duplicates = auditor.find_duplicate_code()
    print(f"Found {len(duplicates)} duplicate function signatures")
    for func, files in duplicates.items():
        for file in files[1:]:  # Keep first, remove rest
            candidates.append({
                'file': file,
                'priority': 2,
                'reason': f'Duplicate of {files[0]}',
                'risk': 'low'
            })

    # Priority 3: Old/experimental files
    experimental_count = 0
    for file_path, data in auditor.files.items():
        if ('experiment' in file_path.lower() or
            'old' in file_path.lower() or
            'backup' in file_path.lower() or
            'test_' in Path(file_path).name.lower() or
            '_test.py' in file_path.lower()):
            candidates.append({
                'file': file_path,
                'priority': 3,
                'reason': 'Experimental, test, or backup file',
                'risk': 'low'
            })
            experimental_count += 1

    print(f"Found {experimental_count} experimental/backup files")

    # Priority 4: Superseded implementations (check if they exist)
    superseded = [
        ('correlation_formula_v1.py', 'correlation_formula_v2.py'),
        ('optimizer_basic.py', 'optimizer_advanced.py'),
        ('quality_simple.py', 'quality_enhanced.py')
    ]

    superseded_count = 0
    for old, new in superseded:
        # Find files matching the pattern
        for file_path in auditor.files.keys():
            if old in file_path:
                # Check if the new version exists
                new_exists = any(new in fp for fp in auditor.files.keys())
                if new_exists:
                    candidates.append({
                        'file': file_path,
                        'priority': 4,
                        'reason': f'Superseded by {new}',
                        'risk': 'medium'
                    })
                    superseded_count += 1

    print(f"Found {superseded_count} superseded implementations")

    # Sort by priority
    candidates.sort(key=lambda x: (x['priority'], x['file']))

    print(f"\nTotal removal candidates: {len(candidates)}")
    for priority in [1, 2, 3, 4]:
        count = len([c for c in candidates if c['priority'] == priority])
        if count > 0:
            print(f"  Priority {priority}: {count} files")

    return candidates


def main():
    """Main execution function"""
    print("🔍 Starting Codebase Audit for Cleanup")
    print("=" * 50)

    # Create auditor and generate main report
    auditor = CodebaseAuditor()
    report = auditor.generate_report()

    # Save main report
    with open('audit_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Analyze optimization files specifically
    optimization_categories = analyze_optimization_files()

    # Identify removal candidates
    candidates = identify_removal_candidates(auditor)

    # Save removal candidates
    with open('removal_candidates.json', 'w') as f:
        json.dump(candidates, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 AUDIT SUMMARY")
    print("=" * 50)
    print(f"Total files analyzed: {report['total_files']}")
    print(f"Total lines of code: {report['total_lines']:,}")
    print(f"Unused files: {len(report['unused_files'])}")
    print(f"Duplicate functions: {len(report['duplicate_functions'])}")
    print(f"Files with errors: {len(report['error_files'])}")
    print(f"Removal candidates: {len(candidates)}")

    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    if report['error_files']:
        print(f"\nFiles with parsing errors:")
        for error_file in report['error_files'][:5]:
            print(f"  - {error_file}")

    print(f"\nReports saved:")
    print(f"  - audit_report.json")
    print(f"  - removal_candidates.json")

    print("\n✅ Audit completed successfully!")


if __name__ == "__main__":
    main()