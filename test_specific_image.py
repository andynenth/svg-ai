#!/usr/bin/env python3
"""Test AI conversion with specific image"""

import requests
import json
from pathlib import Path
import sys

def test_image(image_path):
    base_url = "http://localhost:8001"

    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"📤 Uploading {image_path}...")

    # Upload the image
    with open(image_path, 'rb') as f:
        filename = Path(image_path).name
        files = {'file': (filename, f, 'image/png')}
        response = requests.post(f"{base_url}/api/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return

    upload_data = response.json()
    file_id = upload_data.get('file_id')
    print(f"✅ Uploaded. File ID: {file_id}")

    # Test all 3 tiers
    for tier in [1, 2, 3]:
        print(f"\n🤖 Testing Tier {tier} conversion...")

        ai_payload = {
            "file_id": file_id,
            "tier": tier,
            "target_quality": 0.9,
            "time_budget": 30.0
        }

        response = requests.post(f"{base_url}/api/convert-ai", json=ai_payload)

        if response.status_code == 200:
            result = response.json()

            if result.get('success'):
                # Save the SVG
                svg_content = result.get('svg')
                output_file = f"{Path(image_path).stem}_tier{tier}.svg"

                with open(output_file, 'w') as f:
                    f.write(svg_content)

                print(f"  ✅ Success!")
                print(f"  📄 Saved to: {output_file}")
                print(f"  ⏱️  Time: {result.get('processing_time', 0):.3f}s")

                # Show parameters used
                params = result.get('ai_metadata', {}).get('conversion', {}).get('parameters_used', {})
                if params:
                    print(f"  🔧 Parameters: color_precision={params.get('color_precision')}, "
                          f"corner_threshold={params.get('corner_threshold')}, "
                          f"max_iterations={params.get('max_iterations')}")

                # Show features detected
                features = result.get('ai_metadata', {}).get('conversion', {}).get('features', {})
                if features:
                    print(f"  📊 Features: complexity={features.get('complexity_score', 0):.2f}, "
                          f"colors={features.get('unique_colors', 0):.2f}, "
                          f"gradient={features.get('gradient_strength', 0):.2f}")
            else:
                print(f"  ❌ Conversion failed")
        else:
            print(f"  ❌ HTTP {response.status_code}: {response.text[:100]}")

    print(f"\n✨ Done! Check the generated SVG files:")
    print(f"   - {Path(image_path).stem}_tier1.svg (Simple)")
    print(f"   - {Path(image_path).stem}_tier2.svg (Medium)")
    print(f"   - {Path(image_path).stem}_tier3.svg (Complex)")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_logos/62088.png"
    test_image(image_path)