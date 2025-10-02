#!/usr/bin/env python3
"""
Enhanced training script with detailed progress monitoring
"""

import json
import random
from pathlib import Path
from backend.converter import convert_image
import numpy as np
from PIL import Image
import cv2
from collections import Counter
import time
from datetime import datetime, timedelta
import sys

class ProgressTracker:
    """Track and display training progress"""

    def __init__(self, total_items, task_name="Training"):
        self.total = total_items
        self.current = 0
        self.start_time = time.time()
        self.task_name = task_name
        self.last_update = 0

    def update(self, item_name="", extra_info=""):
        """Update progress with detailed information"""
        self.current += 1
        current_time = time.time()

        # Update every second or on completion
        if current_time - self.last_update >= 1.0 or self.current == self.total:
            self.last_update = current_time
            self._display(item_name, extra_info)

    def _display(self, item_name="", extra_info=""):
        """Display progress bar and statistics"""
        # Calculate metrics
        elapsed = time.time() - self.start_time
        progress = self.current / self.total

        # Time estimates
        if self.current > 0:
            avg_time_per_item = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = avg_time_per_item * remaining_items
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "calculating..."

        # Progress bar
        bar_length = 50
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Clear line and print progress
        sys.stdout.write('\r')
        sys.stdout.write(f"{self.task_name}: [{bar}] {self.current}/{self.total} ({progress:.1%}) | ETA: {eta}")

        # Add extra info on new line if provided
        if extra_info and self.current % 10 == 0:
            sys.stdout.write(f"\n  ↳ {extra_info}\n")

        sys.stdout.flush()

    def complete(self, summary=""):
        """Mark completion and show summary"""
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.total if self.total > 0 else 0

        print(f"\n✅ {self.task_name} Complete!")
        print(f"   Total time: {str(timedelta(seconds=int(elapsed)))}")
        print(f"   Average: {avg_time:.2f}s per item")
        if summary:
            print(f"   {summary}")

def analyze_logo_with_progress(image_path):
    """Analyze logo with detailed feature extraction"""
    img = Image.open(image_path).convert('RGB')
    cv_img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    features = {}

    # Color complexity
    colors = img.getcolors(maxcolors=256)
    features['unique_colors'] = len(colors) if colors else 256

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    features['edge_ratio'] = np.sum(edges > 0) / (img.width * img.height)

    # Gradient detection
    gradient = np.gradient(gray.astype(float))
    features['gradient_strength'] = np.mean(np.abs(gradient[0]) + np.abs(gradient[1]))

    # Text detection (horizontal patterns)
    horizontal_projection = np.sum(edges, axis=1)
    features['horizontal_variance'] = np.var(horizontal_projection)

    # Classify
    if features['unique_colors'] < 10 and features['edge_ratio'] < 0.05:
        logo_type = 'simple_geometric'
    elif features['horizontal_variance'] > 10000:
        logo_type = 'text_based'
    elif features['gradient_strength'] > 15:
        logo_type = 'gradient'
    elif features['edge_ratio'] > 0.15:
        logo_type = 'complex'
    else:
        logo_type = 'abstract'

    return logo_type, features

def generate_training_data_with_progress(sample_size=500):
    """Generate training data with detailed progress monitoring"""

    print("=" * 70)
    print("🚀 AI TRAINING DATA GENERATION")
    print("=" * 70)
    print(f"📁 Source: data/raw_logos/ (2,069 real company logos)")
    print(f"🎯 Sample size: {sample_size} logos")
    print(f"⚙️  Parameters tested per logo: 6")
    print(f"📊 Total conversions: {sample_size * 6}")
    print("=" * 70)

    # Get logo paths
    logo_paths = list(Path("data/raw_logos").glob("*.png"))

    if sample_size < len(logo_paths):
        print(f"🎲 Randomly sampling {sample_size} from {len(logo_paths)} logos...")
        logo_paths = random.sample(logo_paths, sample_size)

    # Parameter combinations
    param_combinations = [
        {"color_precision": 3, "corner_threshold": 30, "segment_length": 6.0},
        {"color_precision": 4, "corner_threshold": 40, "segment_length": 5.0},
        {"color_precision": 6, "corner_threshold": 60, "segment_length": 4.0},
        {"color_precision": 7, "corner_threshold": 70, "segment_length": 3.5},
        {"color_precision": 8, "corner_threshold": 80, "segment_length": 3.0},
        {"color_precision": 10, "corner_threshold": 90, "segment_length": 2.5},
    ]

    training_data = []
    logo_type_counts = Counter()
    quality_scores = []

    # Phase 1: Logo Analysis
    print("\n📊 PHASE 1: Analyzing Logos")
    print("-" * 70)

    analysis_progress = ProgressTracker(len(logo_paths), "Analysis")
    logo_analyses = {}

    for image_path in logo_paths:
        try:
            logo_type, features = analyze_logo_with_progress(image_path)
            logo_analyses[str(image_path)] = (logo_type, features)
            logo_type_counts[logo_type] += 1

            analysis_progress.update(
                item_name=image_path.name,
                extra_info=f"Type: {logo_type}, Colors: {features['unique_colors']}"
            )
        except Exception as e:
            print(f"\n⚠️  Error analyzing {image_path.name}: {e}")

    analysis_progress.complete(f"Found {len(logo_type_counts)} logo types")

    # Show distribution
    print("\n📈 Logo Type Distribution:")
    for logo_type, count in logo_type_counts.most_common():
        bar = '█' * int(20 * count / len(logo_paths))
        print(f"  {logo_type:20s}: {bar} {count} ({100*count/len(logo_paths):.1f}%)")

    # Phase 2: Parameter Testing
    print("\n🔬 PHASE 2: Testing Conversion Parameters")
    print("-" * 70)

    total_conversions = len(logo_paths) * len(param_combinations)
    conversion_progress = ProgressTracker(total_conversions, "Conversions")

    best_scores_by_type = {lt: [] for lt in logo_type_counts.keys()}
    conversion_times = []

    for image_path in logo_paths:
        logo_type, features = logo_analyses[str(image_path)]
        best_result = None
        best_ssim = 0

        for params in param_combinations:
            try:
                start_time = time.time()
                result = convert_image(str(image_path), converter_type='vtracer', **params)
                conversion_time = time.time() - start_time
                conversion_times.append(conversion_time)

                ssim = result.get("ssim", 0)
                quality_scores.append(ssim)

                # Track best for this logo
                if ssim > best_ssim:
                    best_ssim = ssim
                    best_result = {
                        "image_path": str(image_path),
                        "logo_type": logo_type,
                        "parameters": params,
                        "quality_score": ssim,
                        "file_size": len(result.get("svg", "")),
                        "conversion_time": conversion_time,
                        "features": features,
                        "is_best": True
                    }

                # Save all results
                training_data.append({
                    "image_path": str(image_path),
                    "logo_type": logo_type,
                    "parameters": params,
                    "quality_score": ssim,
                    "file_size": len(result.get("svg", "")),
                    "conversion_time": conversion_time,
                    "features": features
                })

                conversion_progress.update(
                    item_name=f"{image_path.name}",
                    extra_info=f"SSIM: {ssim:.3f}, Time: {conversion_time:.2f}s"
                )

            except Exception as e:
                print(f"\n⚠️  Conversion error: {e}")
                conversion_progress.update()

        if best_result:
            best_scores_by_type[logo_type].append(best_ssim)

    conversion_progress.complete(f"Average SSIM: {np.mean(quality_scores):.3f}")

    # Phase 3: Save Results
    print("\n💾 PHASE 3: Saving Training Data")
    print("-" * 70)

    output_file = "training_data_real_logos.json"
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Saved {len(training_data)} samples to {output_file}")

    # Final Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING DATA GENERATION SUMMARY")
    print("=" * 70)

    print(f"\n🎯 Overall Statistics:")
    print(f"  • Total samples generated: {len(training_data)}")
    print(f"  • Unique logos processed: {len(logo_paths)}")
    print(f"  • Average SSIM score: {np.mean(quality_scores):.3f}")
    print(f"  • Best SSIM achieved: {np.max(quality_scores):.3f}")
    print(f"  • Worst SSIM: {np.min(quality_scores):.3f}")
    print(f"  • Average conversion time: {np.mean(conversion_times):.2f}s")

    print(f"\n📈 Performance by Logo Type:")
    for logo_type in logo_type_counts.keys():
        if best_scores_by_type[logo_type]:
            avg_score = np.mean(best_scores_by_type[logo_type])
            max_score = np.max(best_scores_by_type[logo_type])
            print(f"  {logo_type:20s}: Avg {avg_score:.3f}, Best {max_score:.3f}")

    print(f"\n🚀 Next Steps:")
    print(f"  1. Train classifier:        python train_classifier.py")
    print(f"  2. Train quality predictor: python train_quality_predictor.py")
    print(f"  3. Train optimizer:         python train_optimizer.py")

    print("\n✨ Training data ready for AI model training!")

    return training_data

if __name__ == "__main__":
    import sys

    # Get sample size from command line
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
    else:
        sample_size = 100  # Smaller default for demo
        print("ℹ️  Using default 100 logos. Pass a number for more: python train_with_progress.py 500")

    # Generate training data with progress monitoring
    generate_training_data_with_progress(sample_size=sample_size)