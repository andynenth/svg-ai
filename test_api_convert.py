#!/usr/bin/env python3
"""
Test API conversion of the triangle logo
"""

import requests
import json

# Server URL
BASE_URL = "http://localhost:8001"

# Upload the file
with open("data/raw_logos/103944.png", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/api/upload", files=files)

if response.status_code != 200:
    print(f"Upload failed: {response.text}")
    exit(1)

upload_result = response.json()
file_id = upload_result["file_id"]
print(f"✅ Uploaded: {file_id}")

# Convert using standard endpoint with AI classification
convert_data = {
    "file_id": file_id,
    "converter_type": "vtracer",
    "optimize_logo": True  # This triggers AI classification
}

response = requests.post(f"{BASE_URL}/api/convert", json=convert_data)

if response.status_code == 200:
    result = response.json()

    print(f"\n✅ Conversion successful!")
    print(f"   Converter: {result.get('converter_used', 'unknown')}")

    if 'classification' in result:
        print(f"   Logo type: {result['classification'].get('logo_type', 'unknown')}")
        print(f"   Confidence: {result['classification'].get('confidence', 0):.2%}")

    if 'parameters_used' in result:
        params = result['parameters_used']
        print(f"   Parameters:")
        print(f"     - Color precision: {params.get('color_precision', 'default')}")
        print(f"     - Corner threshold: {params.get('corner_threshold', 'default')}")

    if 'metrics' in result:
        metrics = result['metrics']
        print(f"   Quality metrics:")
        print(f"     - SSIM: {metrics.get('ssim', 0):.3f}")
        print(f"     - File size: {metrics.get('file_size_kb', 0):.1f} KB")

    # Save SVG
    if 'svg' in result:
        with open("triangle_converted.svg", "w") as f:
            f.write(result['svg'])
        print(f"\n💾 Saved to: triangle_converted.svg")

        # Show first few lines of SVG
        print(f"\n📄 SVG Preview:")
        svg_lines = result['svg'].split('\n')[:5]
        for line in svg_lines:
            print(f"   {line}")
else:
    print(f"❌ Conversion failed: {response.status_code}")
    print(f"   {response.text[:500]}")