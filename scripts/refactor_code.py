#!/usr/bin/env python3
"""
Code Refactoring Script for Day 13
Standardizes naming, improves organization, and adds documentation
"""

import re
import ast
from pathlib import Path
from typing import List, Dict


class CodeRefactorer:
    def __init__(self):
        self.naming_rules = {
            'classes': 'PascalCase',
            'functions': 'snake_case',
            'constants': 'UPPER_SNAKE_CASE',
            'private': '_leading_underscore'
        }

    def standardize_names(self, file_path: Path):
        """Apply standard naming conventions"""

        content = file_path.read_text()

        # Fix class names
        content = re.sub(
            r'class\s+([a-z][a-zA-Z0-9_]*)',
            lambda m: f"class {self._to_pascal_case(m.group(1))}",
            content
        )

        # Fix constant names
        content = re.sub(
            r'^([A-Z][a-z][a-zA-Z0-9_]*)\s*=',
            lambda m: f"{m.group(1).upper()} =",
            content,
            flags=re.MULTILINE
        )

        # Fix function names to snake_case
        content = re.sub(
            r'def\s+([A-Z][a-zA-Z0-9]*)',
            lambda m: f"def {self._to_snake_case(m.group(1))}",
            content
        )

        file_path.write_text(content)

    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase"""
        parts = name.split('_')
        return ''.join(word.capitalize() for word in parts)

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def add_type_hints(self, file_path: Path):
        """Add type hints where missing"""

        content = file_path.read_text()

        # Add return type hints
        content = re.sub(
            r'def\s+(\w+)\((.*?)\)(\s*):',
            lambda m: self._add_return_type(m),
            content
        )

        file_path.write_text(content)

    def _add_return_type(self, match):
        """Add return type hint to function"""
        func_name = match.group(1)
        params = match.group(2)

        # Heuristic for return types
        if func_name.startswith('is_') or func_name.startswith('has_'):
            return f"def {func_name}({params}) -> bool:"
        elif func_name.startswith('get_') or func_name.startswith('calculate_'):
            return f"def {func_name}({params}) -> Dict:"
        elif func_name == '__init__':
            return f"def {func_name}({params}) -> None:"
        else:
            return match.group(0)  # Keep original

    def reorganize_code(self, file_path: Path):
        """Improve code organization within files"""

        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            # Separate components
            imports = []
            constants = []
            classes = []
            functions = []
            main_block = []

            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
                elif isinstance(node, ast.Assign):
                    # Check if constant (UPPER_CASE)
                    if any(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                        constants.append(node)
                    else:
                        main_block.append(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node)
                else:
                    main_block.append(node)

            # Reorganize in standard order
            reorganized = []

            # 1. Module docstring
            if ast.get_docstring(tree):
                reorganized.append(tree.body[0])

            # 2. Imports (sorted)
            reorganized.extend(sorted(imports, key=lambda x: ast.unparse(x)))

            # 3. Constants
            reorganized.extend(constants)

            # 4. Classes
            reorganized.extend(classes)

            # 5. Functions
            reorganized.extend(functions)

            # 6. Main block
            reorganized.extend(main_block)

            # Write reorganized code
            new_content = ast.unparse(ast.Module(body=reorganized, type_ignores=[]))
            file_path.write_text(new_content)

        except Exception as e:
            print(f"Error reorganizing {file_path}: {e}")

    def add_docstrings(self, file_path: Path):
        """Add docstrings to all public functions and classes"""

        try:
            content = file_path.read_text()
            tree = ast.parse(content)

            modified = False

            for node in ast.walk(tree):
                # Add docstrings to classes
                if isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        # Generate docstring
                        docstring = f'"""{node.name} class for AI processing"""'
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        modified = True

                # Add docstrings to functions
                elif isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node) and not node.name.startswith('_'):
                        # Generate docstring based on function name
                        docstring = f'"""{node.name.replace("_", " ").title()}"""'
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                        modified = True

            if modified:
                file_path.write_text(ast.unparse(tree))

        except Exception as e:
            print(f"Error adding docstrings to {file_path}: {e}")

    def refactor_file(self, file_path: Path):
        """Apply all refactoring to a single file"""
        print(f"Refactoring: {file_path}")

        # Apply naming conventions
        self.standardize_names(file_path)

        # Add type hints
        self.add_type_hints(file_path)

        # Reorganize code
        self.reorganize_code(file_path)

        # Add docstrings
        self.add_docstrings(file_path)

    def refactor_all_files(self):
        """Refactor all AI module files"""

        # Get all merged module files
        target_files = [
            Path('backend/ai_modules/classification.py'),
            Path('backend/ai_modules/optimization.py'),
            Path('backend/ai_modules/quality.py'),
            Path('backend/ai_modules/utils.py')
        ]

        for file_path in target_files:
            if file_path.exists():
                try:
                    self.refactor_file(file_path)
                    print(f"✓ Refactored: {file_path}")
                except Exception as e:
                    print(f"✗ Failed to refactor {file_path}: {e}")
            else:
                print(f"⚠ File not found: {file_path}")

        print("\n🎉 Code refactoring completed!")


def main():
    """Run code refactoring"""
    print("🔧 Code Refactoring - Day 13")
    print("=" * 40)

    refactorer = CodeRefactorer()
    refactorer.refactor_all_files()


if __name__ == "__main__":
    main()