#!/usr/bin/env python3
"""
Intelligent Logo Organizer for ULTRATHINK v2.0
Automatically classifies raw logos into categories using CV2 analysis
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
import random
from collections import defaultdict
import shutil

class IntelligentLogoAnalyzer:
    def __init__(self):
        self.categories = ['simple', 'text', 'gradient', 'complex']
        self.stats = defaultdict(int)

    def analyze_logo_type(self, image_path):
        """Analyze logo and classify into one of 4 categories"""
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return 'complex'  # Fallback

        h, w = img.shape[:2]

        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Feature extraction
        features = self._extract_features(img, gray, hsv)

        # Classification logic
        return self._classify_from_features(features)

    def _extract_features(self, img, gray, hsv):
        """Extract comprehensive features for classification"""
        h, w = img.shape[:2]

        # Color analysis
        unique_colors = len(np.unique(img.reshape(-1, img.shape[-1]), axis=0))
        color_variance = np.var(img, axis=(0,1)).mean()

        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_strength = np.mean(gradient_magnitude)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)

        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Geometric analysis
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0
        else:
            solidity = 0
            contour_area = 0

        # Text detection (simplified)
        text_score = self._detect_text_like_features(gray)

        return {
            'unique_colors': unique_colors,
            'color_variance': color_variance,
            'gradient_strength': gradient_strength,
            'edge_density': edge_density,
            'num_contours': num_contours,
            'solidity': solidity,
            'text_score': text_score,
            'aspect_ratio': w / h
        }

    def _detect_text_like_features(self, gray):
        """Detect text-like features in image"""
        # Horizontal and vertical line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        h_score = np.sum(horizontal_lines > 0) / gray.size
        v_score = np.sum(vertical_lines > 0) / gray.size

        return h_score + v_score

    def _classify_from_features(self, features):
        """Classify logo based on extracted features"""
        # Simple geometric shapes
        if (features['unique_colors'] < 20 and
            features['edge_density'] < 0.1 and
            features['solidity'] > 0.8 and
            features['num_contours'] < 5):
            return 'simple'

        # Text-based logos
        if (features['text_score'] > 0.05 or
            features['aspect_ratio'] > 2.0 or
            features['aspect_ratio'] < 0.5):
            return 'text'

        # Gradient/artistic logos
        if (features['color_variance'] > 2000 or
            features['gradient_strength'] > 100 or
            features['unique_colors'] > 50):
            return 'gradient'

        # Complex logos (default)
        return 'complex'

    def organize_dataset(self, raw_dir, output_dir):
        """Organize raw logos into classified structure"""
        raw_path = Path(raw_dir)
        output_path = Path(output_dir)

        # Create directory structure
        for split in ['train', 'val', 'test']:
            for category in self.categories:
                (output_path / split / category).mkdir(parents=True, exist_ok=True)

        # Get all logo files
        logo_files = list(raw_path.glob('*.png'))
        random.shuffle(logo_files)

        print(f"Organizing {len(logo_files)} logos...")

        # Classify and organize
        classified = defaultdict(list)
        for i, logo_file in enumerate(logo_files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(logo_files)} images...")

            category = self.analyze_logo_type(logo_file)
            classified[category].append(logo_file)
            self.stats[category] += 1

        # Balance dataset and split
        target_per_class = 200  # ~800 total for balanced dataset
        organized_count = 0

        for category in self.categories:
            category_files = classified[category][:target_per_class]

            # Split: 70% train, 20% val, 10% test
            n_train = int(len(category_files) * 0.7)
            n_val = int(len(category_files) * 0.2)

            train_files = category_files[:n_train]
            val_files = category_files[n_train:n_train + n_val]
            test_files = category_files[n_train + n_val:]

            # Copy files
            for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                for file_path in files:
                    dest = output_path / split / category / file_path.name
                    shutil.copy2(file_path, dest)
                    organized_count += 1

        # Generate metadata
        metadata = {
            'total_organized': organized_count,
            'raw_stats': dict(self.stats),
            'structure': {
                split: {
                    category: len(list((output_path / split / category).glob('*.png')))
                    for category in self.categories
                }
                for split in ['train', 'val', 'test']
            }
        }

        with open(output_path / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Dataset organized: {organized_count} images")
        print(f"📊 Raw distribution: {dict(self.stats)}")
        print(f"📁 Structure created in: {output_path}")

        return metadata

def main():
    """Main organization function"""
    organizer = IntelligentLogoAnalyzer()

    # Default paths for Colab
    raw_dir = "/tmp/claude/raw_logos"
    output_dir = "/tmp/claude/data/training/classification"

    # Check if running locally vs Colab
    if not os.path.exists(raw_dir):
        print("⚠️  Running locally - using local paths")
        raw_dir = "/tmp/claude/raw_logos"
        output_dir = "/tmp/claude/data/training/classification"

    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")

    if not os.path.exists(raw_dir):
        print("❌ Raw logos directory not found. Please extract dataset first.")
        return

    metadata = organizer.organize_dataset(raw_dir, output_dir)
    print(f"🎯 Organization complete! Metadata saved.")

    return metadata

if __name__ == "__main__":
    main()