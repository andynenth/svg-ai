#!/usr/bin/env python3
"""Fix syntax errors created by automated import replacement"""

import os
import re
from pathlib import Path

class SyntaxErrorFixer:
    def __init__(self):
        self.patterns_to_fix = [
            # Fix double instantiation patterns
            (r'OptimizationEngine\(\)\(\)', 'OptimizationEngine()'),
            (r'ClassificationModule\(\)\(\)', 'ClassificationModule()'),

            # Fix any other double instantiation patterns that might exist
            (r'(\w+Module\(\))\(\)', r'\1'),
            (r'(\w+Engine\(\))\(\)', r'\1'),
        ]

        self.files_fixed = []
        self.total_fixes = 0

    def fix_file(self, file_path):
        """Fix syntax errors in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes_in_file = 0

            # Apply each pattern fix
            for pattern, replacement in self.patterns_to_fix:
                matches = re.findall(pattern, content)
                if matches:
                    content = re.sub(pattern, replacement, content)
                    fixes_in_file += len(matches)
                    print(f"  Fixed {len(matches)} instances of '{pattern}' -> '{replacement}'")

            # Only write if changes were made
            if fixes_in_file > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.files_fixed.append(file_path)
                self.total_fixes += fixes_in_file
                print(f"✅ Fixed {fixes_in_file} syntax errors in {file_path}")
                return True
            else:
                print(f"  No syntax errors found in {file_path}")
                return False

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False

    def fix_all_test_files(self):
        """Fix syntax errors in all test files"""
        print("🔧 Fixing syntax errors in test files...")

        test_dir = Path('tests')
        if not test_dir.exists():
            print("❌ Tests directory not found")
            return False

        # Find all Python test files
        test_files = list(test_dir.rglob('*.py'))

        print(f"Found {len(test_files)} test files to check")

        for test_file in test_files:
            print(f"\nChecking {test_file}...")
            self.fix_file(test_file)

        return True

    def generate_summary(self):
        """Generate summary of fixes applied"""
        print("\n" + "="*60)
        print("SYNTAX ERROR FIXING SUMMARY")
        print("="*60)
        print(f"Total files processed: {len(self.files_fixed)}")
        print(f"Total syntax errors fixed: {self.total_fixes}")

        if self.files_fixed:
            print("\nFiles with fixes applied:")
            for file_path in self.files_fixed:
                print(f"  - {file_path}")

        print("\n✅ Syntax error fixing completed!")

def main():
    fixer = SyntaxErrorFixer()

    if fixer.fix_all_test_files():
        fixer.generate_summary()
        return True
    else:
        print("❌ Failed to fix syntax errors")
        return False

if __name__ == '__main__':
    main()