
# COLAB DATASET SETUP CODE
# Copy this code into your Colab notebook for automatic dataset preparation

import zipfile
import os
from pathlib import Path
import shutil

def setup_colab_dataset():
    """Automatically setup dataset in Colab after upload"""

    print("🔄 Setting up dataset in Colab...")

    # Create directory structure
    os.makedirs('data/training/classification', exist_ok=True)
    for split in ['train', 'val', 'test']:
        for class_name in ['simple', 'text', 'gradient', 'complex']:
            os.makedirs(f'data/training/classification/{split}/{class_name}', exist_ok=True)

    # Look for uploaded zip file
    zip_files = [f for f in os.listdir('.') if f.endswith('.zip') and 'logo' in f.lower()]

    if not zip_files:
        print("❌ No logo dataset zip file found")
        print("Please upload colab_logo_dataset.zip using files.upload()")
        return False

    zip_file = zip_files[0]
    print(f"📦 Found dataset: {zip_file}")

    # Extract dataset
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        zipf.extractall('.')

    print(f"✅ Extracted {zip_file}")

    # Verify extraction
    raw_logos_path = Path('raw_logos')
    if raw_logos_path.exists():
        png_count = len(list(raw_logos_path.glob('*.png')))
        print(f"✅ Found {png_count} logo images in raw_logos/")
        return True
    else:
        print("❌ raw_logos directory not found after extraction")
        return False

# Run setup
if setup_colab_dataset():
    print("🎯 Dataset ready for intelligent organization!")
else:
    print("❌ Dataset setup failed - please check upload")
