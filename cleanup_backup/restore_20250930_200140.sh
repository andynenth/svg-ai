#!/bin/bash
# Restore script for cleanup from 20250930_200140
# This script will restore all files removed during cleanup

set -e  # Exit on any error

BACKUP_DIR="cleanup_backup/20250930_200140"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "🔄 Restoring 0 files from cleanup 20250930_200140..."
echo "Backup directory: $BACKUP_DIR"
echo

RESTORED=0
FAILED=0


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
