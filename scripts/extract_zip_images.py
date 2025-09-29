#!/usr/bin/env python3
"""
ZIP Extraction Script for SVG-AI Project
Extracts all ZIP files from zip-images/ to data/raw_logos/ and removes ZIP files after extraction.
"""

import os
import zipfile
import shutil
from pathlib import Path
import sys
from typing import List, Tuple

def verify_directories() -> bool:
    """Verify that source and target directories exist."""
    zip_dir = Path("zip-images")
    target_dir = Path("data/raw_logos")

    if not zip_dir.exists():
        print(f"❌ Source directory '{zip_dir}' does not exist")
        return False

    if not target_dir.exists():
        print(f"❌ Target directory '{target_dir}' does not exist")
        return False

    print(f"✅ Source directory: {zip_dir} (exists)")
    print(f"✅ Target directory: {target_dir} (exists)")
    return True

def get_zip_files() -> List[Path]:
    """Get list of all ZIP files in zip-images directory."""
    zip_dir = Path("zip-images")
    zip_files = list(zip_dir.glob("*.zip"))
    print(f"📁 Found {len(zip_files)} ZIP files to extract")
    return zip_files

def extract_zip_file(zip_path: Path, target_dir: Path) -> Tuple[bool, int]:
    """
    Extract a single ZIP file to target directory.
    Returns (success, file_count)
    """
    try:
        print(f"📦 Extracting: {zip_path.name}")

        # Check if ZIP file is valid
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test ZIP integrity
            zip_ref.testzip()

            # Get list of files to extract
            file_list = zip_ref.namelist()
            file_count = len([f for f in file_list if not f.endswith('/')])

            # Extract all files (overwrite existing)
            zip_ref.extractall(target_dir)

            print(f"   ✅ Extracted {file_count} files")
            return True, file_count

    except zipfile.BadZipFile:
        print(f"   ❌ Invalid ZIP file: {zip_path.name}")
        return False, 0
    except Exception as e:
        print(f"   ❌ Extraction failed: {zip_path.name} - {str(e)}")
        return False, 0

def remove_zip_file(zip_path: Path) -> bool:
    """Remove ZIP file after successful extraction."""
    try:
        zip_path.unlink()
        print(f"   🗑️  Removed: {zip_path.name}")
        return True
    except Exception as e:
        print(f"   ⚠️  Failed to remove {zip_path.name}: {str(e)}")
        return False

def main():
    """Main extraction workflow."""
    print("🚀 Starting ZIP extraction process...")
    print("=" * 50)

    # Verify directories exist
    if not verify_directories():
        sys.exit(1)

    # Get list of ZIP files
    zip_files = get_zip_files()
    if not zip_files:
        print("ℹ️  No ZIP files found to extract")
        return

    print("=" * 50)

    # Process each ZIP file
    total_files_extracted = 0
    successful_extractions = 0
    removed_zips = 0

    target_dir = Path("data/raw_logos")

    for zip_path in zip_files:
        success, file_count = extract_zip_file(zip_path, target_dir)

        if success:
            successful_extractions += 1
            total_files_extracted += file_count

            # Remove ZIP file after successful extraction
            if remove_zip_file(zip_path):
                removed_zips += 1

        print()  # Add spacing between files

    # Summary
    print("=" * 50)
    print("📊 EXTRACTION SUMMARY")
    print(f"   ZIP files processed: {len(zip_files)}")
    print(f"   Successful extractions: {successful_extractions}")
    print(f"   Total files extracted: {total_files_extracted}")
    print(f"   ZIP files removed: {removed_zips}")

    if successful_extractions == len(zip_files) and removed_zips == successful_extractions:
        print("✅ All ZIP files extracted and removed successfully!")
    else:
        print("⚠️  Some operations failed. Check output above for details.")

if __name__ == "__main__":
    main()