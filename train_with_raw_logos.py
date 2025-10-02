#!/usr/bin/env python3
"""
Train AI models using the 2,069 real logos in data/raw_logos/
This will produce much better models than synthetic test data!
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

def analyze_logo(image_path):
    """Analyze logo to auto-classify its type"""
    img = Image.open(image_path).convert('RGB')
    cv_img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Feature extraction for classification
    features = {}

    # Color complexity
    colors = img.getcolors(maxcolors=256)
    features['unique_colors'] = len(colors) if colors else 256

    # Edge detection (indicates complexity)
    edges = cv2.Canny(gray, 50, 150)
    features['edge_ratio'] = np.sum(edges > 0) / (img.width * img.height)

    # Detect gradients
    gradient = np.gradient(gray.astype(float))
    features['gradient_strength'] = np.mean(np.abs(gradient[0]) + np.abs(gradient[1]))

    # Detect text-like horizontal patterns
    horizontal_projection = np.sum(edges, axis=1)
    features['horizontal_variance'] = np.var(horizontal_projection)

    # Classify based on features
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

def generate_training_data_from_raw_logos(sample_size=500):
    """
    Generate high-quality training data from real logos

    Args:
        sample_size: How many logos to use (randomly sampled)
    """
    print(f"Generating training data from real logos...")
    print(f"Found 2,069 logos in data/raw_logos/")

    # Get all logo paths
    logo_paths = list(Path("data/raw_logos").glob("*.png"))

    # Random sample for manageable training time
    if sample_size < len(logo_paths):
        logo_paths = random.sample(logo_paths, sample_size)
        print(f"Using random sample of {sample_size} logos")

    training_data = []
    logo_type_counts = Counter()

    # Parameter combinations to test (optimized for real logos)
    param_combinations = [
        # Low precision (fast, simple logos)
        {"color_precision": 3, "corner_threshold": 30, "segment_length": 6.0},
        {"color_precision": 4, "corner_threshold": 40, "segment_length": 5.0},

        # Medium precision (balanced)
        {"color_precision": 6, "corner_threshold": 60, "segment_length": 4.0},
        {"color_precision": 7, "corner_threshold": 70, "segment_length": 3.5},

        # High precision (complex logos)
        {"color_precision": 8, "corner_threshold": 80, "segment_length": 3.0},
        {"color_precision": 10, "corner_threshold": 90, "segment_length": 2.5},
    ]

    print("\nProcessing logos...")
    for i, image_path in enumerate(logo_paths):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(logo_paths)} logos processed")

        try:
            # Auto-classify logo type
            logo_type, features = analyze_logo(image_path)
            logo_type_counts[logo_type] += 1

            # Test different parameters
            best_result = None
            best_ssim = 0

            for params in param_combinations:
                try:
                    start_time = time.time()
                    result = convert_image(str(image_path), converter_type='vtracer', **params)
                    conversion_time = time.time() - start_time

                    ssim = result.get("ssim", 0)

                    # Keep best result
                    if ssim > best_ssim:
                        best_ssim = ssim
                        best_result = {
                            "image_path": str(image_path),
                            "logo_type": logo_type,
                            "parameters": params,
                            "quality_score": ssim,
                            "mse": result.get("mse", 0),
                            "psnr": result.get("psnr", 0),
                            "file_size": len(result.get("svg", "")),
                            "conversion_time": conversion_time,
                            "features": features
                        }

                    # Also save non-optimal results for learning
                    training_data.append({
                        "image_path": str(image_path),
                        "logo_type": logo_type,
                        "parameters": params,
                        "quality_score": ssim,
                        "mse": result.get("mse", 0),
                        "psnr": result.get("psnr", 0),
                        "file_size": len(result.get("svg", "")),
                        "conversion_time": conversion_time,
                        "features": features
                    })

                except Exception as e:
                    print(f"    Error with params {params}: {e}")

            # Add best result with flag
            if best_result:
                best_result["is_best"] = True
                training_data.append(best_result)

        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")

    # Save training data
    output_file = "training_data_real_logos.json"
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"✅ Generated {len(training_data)} training samples")
    print(f"📁 Saved to {output_file}")
    print(f"🏢 From {len(logo_paths)} real company logos")

    print("\nLogo Type Distribution:")
    total = sum(logo_type_counts.values())
    for logo_type, count in logo_type_counts.most_common():
        percentage = 100 * count / total
        print(f"  {logo_type:20s}: {count:4d} ({percentage:.1f}%)")

    # Quality statistics
    ssim_scores = [d['quality_score'] for d in training_data if 'quality_score' in d]
    if ssim_scores:
        print(f"\nQuality Statistics:")
        print(f"  Average SSIM: {np.mean(ssim_scores):.3f}")
        print(f"  Best SSIM:    {np.max(ssim_scores):.3f}")
        print(f"  Worst SSIM:   {np.min(ssim_scores):.3f}")

    print("\n🎯 Next Steps:")
    print("1. Train classifier:        python train_classifier_real.py")
    print("2. Train quality predictor: python train_quality_real.py")
    print("3. Train optimizer:         python train_optimizer_real.py")

    return training_data

if __name__ == "__main__":
    import sys

    # Get sample size from command line or use default
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])
        print(f"Using {sample_size} logos as specified")
    else:
        sample_size = 500
        print(f"Using default {sample_size} logos (pass a number to change)")

    # Generate training data
    generate_training_data_from_raw_logos(sample_size=sample_size)