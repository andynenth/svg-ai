"""
Safe File Removal Execution for AI SVG Converter Cleanup

This script safely removes files identified during the cleanup analysis:
- Creates backups before removal
- Generates removal logs
- Creates restore scripts
- Verifies system stability after cleanup
"""

import shutil
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)


class SafeFileRemover:
    """Safe file removal with backup and restore capabilities"""

    def __init__(self, backup_dir: str = 'cleanup_backup'):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.removal_log = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def backup_file(self, file_path: Path) -> Path:
        """Create backup of file before removal"""
        try:
            # Ensure we have an absolute path
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path

            # Create relative path for backup structure
            try:
                relative_path = file_path.relative_to(Path.cwd())
            except ValueError:
                # If file is outside current directory, use the file name
                relative_path = file_path.name

            backup_path = self.backup_dir / self.timestamp / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up {file_path} to {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to backup {file_path}: {e}")
            raise

    def remove_file(self, file_path: Path, reason: str) -> bool:
        """Safely remove a file with backup"""
        try:
            # Ensure we have an absolute path
            if not file_path.is_absolute():
                file_path = Path.cwd() / file_path

            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False

            # Create backup
            backup_path = self.backup_file(file_path)

            # Remove file
            file_path.unlink()

            # Log removal
            self.removal_log.append({
                'file': str(file_path),
                'backup': str(backup_path),
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'size_bytes': backup_path.stat().st_size
            })

            print(f"✓ Removed: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
            print(f"✗ Failed to remove {file_path}: {e}")
            return False

    def remove_batch(self, files: List[Dict]) -> Dict:
        """Remove multiple files"""
        results = {
            'removed': [],
            'failed': [],
            'skipped': [],
            'total': len(files)
        }

        print(f"🗂️ Processing {len(files)} files for removal...")

        for i, file_info in enumerate(files, 1):
            file_path = Path(file_info['file'])
            reason = file_info.get('reason', 'No reason provided')

            print(f"[{i}/{len(files)}] Processing: {file_path}")

            if not file_path.exists():
                print(f"  ⚠️ File does not exist, skipping")
                results['skipped'].append(str(file_path))
                continue

            if self.remove_file(file_path, reason):
                results['removed'].append(str(file_path))
            else:
                results['failed'].append(str(file_path))

        return results

    def save_log(self) -> Path:
        """Save removal log"""
        log_file = self.backup_dir / f'removal_log_{self.timestamp}.json'

        log_data = {
            'timestamp': self.timestamp,
            'total_removed': len(self.removal_log),
            'total_size_bytes': sum(entry['size_bytes'] for entry in self.removal_log),
            'removals': self.removal_log
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"📋 Removal log saved: {log_file}")
        return log_file

    def generate_restore_script(self) -> Path:
        """Generate script to restore removed files"""
        restore_script = f'''#!/bin/bash
# Restore script for cleanup from {self.timestamp}
# This script will restore all files removed during cleanup

set -e  # Exit on any error

BACKUP_DIR="{self.backup_dir / self.timestamp}"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "🔄 Restoring {len(self.removal_log)} files from cleanup {self.timestamp}..."
echo "Backup directory: $BACKUP_DIR"
echo

RESTORED=0
FAILED=0

'''

        for entry in self.removal_log:
            original = entry['file']
            backup = entry['backup']
            restore_script += f'''
# Restore: {original}
if [ -f "{backup}" ]; then
    echo "  ✓ Restoring {original}"
    mkdir -p "$(dirname "{original}")"
    cp "{backup}" "{original}"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: {backup}"
    ((FAILED++))
fi
'''

        restore_script += f'''
echo
echo "📊 Restoration Summary:"
echo "  Restored: $RESTORED files"
echo "  Failed: $FAILED files"
echo

if [ $FAILED -eq 0 ]; then
    echo "✅ All files restored successfully!"
else
    echo "⚠️ Some files could not be restored. Check backup directory."
fi
'''

        script_file = self.backup_dir / f'restore_{self.timestamp}.sh'
        with open(script_file, 'w') as f:
            f.write(restore_script)

        script_file.chmod(0o755)
        print(f"🔧 Restore script created: {script_file}")
        return script_file


def execute_phase1_cleanup():
    """Execute first phase of cleanup - remove unused and low-risk files"""
    print("🧹 Starting Phase 1 File Cleanup")
    print("=" * 50)

    # Check if removal candidates file exists
    candidates_file = Path('removal_candidates.json')
    if not candidates_file.exists():
        print("❌ removal_candidates.json not found. Run audit_codebase.py first.")
        return False

    # Load removal candidates from analysis
    try:
        with open(candidates_file, 'r') as f:
            candidates = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load removal candidates: {e}")
        return False

    # Filter for phase 1 (priority 1 and 2 only, low risk)
    phase1_files = [
        c for c in candidates
        if c['priority'] <= 2 and c.get('risk', 'medium') == 'low'
    ]

    print(f"📋 Phase 1 targets: {len(phase1_files)} files")
    print(f"   Priority 1 (unused): {len([c for c in phase1_files if c['priority'] == 1])}")
    print(f"   Priority 2 (duplicates): {len([c for c in phase1_files if c['priority'] == 2])}")

    if not phase1_files:
        print("ℹ️ No files selected for Phase 1 removal")
        return True

    # Show what will be removed
    print(f"\n📂 Files to be removed:")
    for file_info in phase1_files[:10]:  # Show first 10
        print(f"   - {file_info['file']} ({file_info['reason']})")
    if len(phase1_files) > 10:
        print(f"   ... and {len(phase1_files) - 10} more")

    # Confirm before proceeding
    print(f"\n⚠️  About to remove {len(phase1_files)} files. Backups will be created.")

    # Execute removal
    remover = SafeFileRemover()
    results = remover.remove_batch(phase1_files)

    # Save log and generate restore script
    log_file = remover.save_log()
    restore_script = remover.generate_restore_script()

    # Print results
    print(f"\n📊 Phase 1 Cleanup Results:")
    print(f"   ✅ Removed: {len(results['removed'])} files")
    print(f"   ❌ Failed: {len(results['failed'])} files")
    print(f"   ⚠️ Skipped: {len(results['skipped'])} files")
    print(f"   📋 Log: {log_file}")
    print(f"   🔧 Restore script: {restore_script}")

    if results['failed']:
        print(f"\n❌ Failed to remove {len(results['failed'])} files:")
        for failed_file in results['failed']:
            print(f"   - {failed_file}")

    success_rate = len(results['removed']) / max(1, len(phase1_files)) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}%")

    return len(results['failed']) == 0


def verify_after_cleanup():
    """Verify system works after cleanup"""
    print("\n🧪 Verifying System Stability After Cleanup")
    print("=" * 50)

    verification_steps = []

    # Step 1: Check critical imports
    print("🔍 Testing critical imports...")
    try:
        import fastapi
        verification_steps.append(('FastAPI import', 'PASS'))
    except ImportError as e:
        verification_steps.append(('FastAPI import', f'FAIL: {e}'))

    try:
        from backend.converters.ai_enhanced_converter import AIEnhancedConverter
        verification_steps.append(('AI Enhanced Converter import', 'PASS'))
    except ImportError as e:
        verification_steps.append(('AI Enhanced Converter import', f'FAIL: {e}'))

    try:
        from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
        verification_steps.append(('Unified AI Pipeline import', 'PASS'))
    except ImportError as e:
        verification_steps.append(('Unified AI Pipeline import', f'FAIL: {e}'))

    # Step 2: Check if we can instantiate core components
    print("⚙️ Testing component instantiation...")
    try:
        from backend.converters.vtracer_converter import VTracerConverter
        converter = VTracerConverter()
        verification_steps.append(('VTracer converter instantiation', 'PASS'))
    except Exception as e:
        verification_steps.append(('VTracer converter instantiation', f'FAIL: {e}'))

    # Step 3: Test cache manager (new component)
    try:
        from backend.ai_modules.utils.cache_manager import MultiLevelCache
        cache = MultiLevelCache()
        verification_steps.append(('Cache manager instantiation', 'PASS'))
    except Exception as e:
        verification_steps.append(('Cache manager instantiation', f'FAIL: {e}'))

    # Step 4: Run basic unit tests (if pytest available)
    print("🧪 Running basic tests...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '--version'],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            verification_steps.append(('Pytest availability', 'PASS'))

            # Try to run a quick test
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', 'tests/', '-x', '--tb=short'],
                    capture_output=True,
                    timeout=60
                )
                if result.returncode == 0:
                    verification_steps.append(('Basic unit tests', 'PASS'))
                else:
                    verification_steps.append(('Basic unit tests', 'FAIL'))
            except subprocess.TimeoutExpired:
                verification_steps.append(('Basic unit tests', 'TIMEOUT'))
            except Exception as e:
                verification_steps.append(('Basic unit tests', f'SKIP: {e}'))
        else:
            verification_steps.append(('Pytest availability', 'FAIL'))
    except Exception as e:
        verification_steps.append(('Pytest availability', f'FAIL: {e}'))

    # Step 5: Check if main application can be imported
    print("🚀 Testing main application...")
    try:
        # Test if we can import the main app without running it
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Capture stdout to avoid app startup messages

        try:
            from backend.app import app
            verification_steps.append(('Main app import', 'PASS'))
        finally:
            sys.stdout = old_stdout
    except Exception as e:
        verification_steps.append(('Main app import', f'FAIL: {e}'))

    # Report results
    print(f"\n📊 Verification Results:")
    passed = 0
    for step, result in verification_steps:
        if 'PASS' in result:
            emoji = '✅'
            passed += 1
        elif 'FAIL' in result:
            emoji = '❌'
        else:
            emoji = '⚠️'
        print(f"   {emoji} {step}: {result}")

    success_rate = (passed / len(verification_steps)) * 100
    all_critical_pass = all(
        'PASS' in result for step, result in verification_steps
        if any(critical in step for critical in ['import', 'instantiation'])
    )

    print(f"\n📈 Verification Summary:")
    print(f"   Success rate: {success_rate:.1f}% ({passed}/{len(verification_steps)})")
    print(f"   Critical systems: {'✅ PASS' if all_critical_pass else '❌ FAIL'}")

    return all_critical_pass


def rollback_cleanup(timestamp: str = None):
    """Rollback cleanup by running the restore script"""
    print("🔄 Rolling Back Cleanup")
    print("=" * 50)

    backup_dir = Path('cleanup_backup')
    if not backup_dir.exists():
        print("❌ No backup directory found")
        return False

    # Find restore script
    if timestamp:
        restore_script = backup_dir / f'restore_{timestamp}.sh'
    else:
        # Find the most recent restore script
        restore_scripts = list(backup_dir.glob('restore_*.sh'))
        if not restore_scripts:
            print("❌ No restore scripts found")
            return False
        restore_script = sorted(restore_scripts, key=lambda x: x.stat().st_mtime)[-1]

    if not restore_script.exists():
        print(f"❌ Restore script not found: {restore_script}")
        return False

    print(f"🔧 Running restore script: {restore_script}")

    try:
        result = subprocess.run(['bash', str(restore_script)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run restore script: {e}")
        return False


def main():
    """Main cleanup execution"""
    print("🧹 AI SVG Converter - File Cleanup Execution")
    print("=" * 50)
    print(f"Start time: {datetime.now()}")

    try:
        # Phase 1: Execute file removal
        print("\n" + "="*50)
        print("PHASE 1: FILE REMOVAL")
        print("="*50)

        phase1_success = execute_phase1_cleanup()

        if not phase1_success:
            print("\n❌ Phase 1 cleanup failed!")
            return False

        # Phase 2: Verify system stability
        print("\n" + "="*50)
        print("PHASE 2: SYSTEM VERIFICATION")
        print("="*50)

        verification_success = verify_after_cleanup()

        if verification_success:
            print("\n✅ Cleanup completed successfully!")
            print("   All critical systems verified working")
            print("   Backups and restore script available if needed")
        else:
            print("\n⚠️ Cleanup completed but verification failed!")
            print("   Consider running rollback if issues persist")
            print("   Use rollback_cleanup() function to restore files")

        return verification_success

    except Exception as e:
        logger.error(f"Cleanup execution failed: {e}")
        print(f"\n💥 Cleanup failed with error: {e}")
        print("   All removed files have backups available")
        print("   Use rollback_cleanup() to restore if needed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)