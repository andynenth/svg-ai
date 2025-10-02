#!/usr/bin/env python3
"""
Generate training data from existing logos for AI model training
"""

import json
from pathlib import Path
from backend.converter import convert_image
import time

def generate_training_data():
    """Generate training data from existing logos"""

    training_data = []
    logo_dirs = Path("data/logos").glob("*")

    print("Generating training data from logos...")

    for logo_dir in logo_dirs:
        if logo_dir.is_dir():
            logo_type = logo_dir.name  # simple_geometric, text_based, etc.
            print(f"\nProcessing {logo_type} logos...")

            for image_path in list(logo_dir.glob("*.png"))[:5]:  # Limit to 5 per category for quick demo
                print(f"  - {image_path.name}")

                # Try different VTracer parameters
                params_to_try = [
                    {"color_precision": 6, "corner_threshold": 60, "segment_length": 4.0},
                    {"color_precision": 4, "corner_threshold": 30, "segment_length": 5.0},
                    {"color_precision": 8, "corner_threshold": 90, "segment_length": 3.0},
                ]

                for params in params_to_try:
                    start_time = time.time()
                    result = convert_image(str(image_path), converter_type='vtracer', **params)
                    conversion_time = time.time() - start_time

                    training_data.append({
                        "image_path": str(image_path),
                        "logo_type": logo_type,
                        "parameters": params,
                        "quality_score": result.get("ssim", 0),
                        "mse": result.get("mse", 0),
                        "psnr": result.get("psnr", 0),
                        "file_size": len(result.get("svg", "")),
                        "conversion_time": conversion_time
                    })

    # Save training data
    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\n✅ Generated {len(training_data)} training samples")
    print(f"📁 Saved to training_data.json")

    # Show summary
    print("\nTraining Data Summary:")
    logo_types = {}
    for item in training_data:
        lt = item['logo_type']
        if lt not in logo_types:
            logo_types[lt] = []
        logo_types[lt].append(item['quality_score'])

    for lt, scores in logo_types.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"  {lt}: {len(scores)} samples, avg SSIM: {avg_score:.3f}")

if __name__ == "__main__":
    generate_training_data()