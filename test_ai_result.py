#!/usr/bin/env python3
"""Test script to show AI conversion result"""

import requests
import json
from pathlib import Path

def test_and_save_result():
    base_url = "http://localhost:8001"

    # Step 1: Upload a test image
    test_image = "data/test/gradient_logo.png"

    print(f"📤 Uploading {test_image}...")
    with open(test_image, 'rb') as f:
        files = {'file': ('gradient_logo.png', f, 'image/png')}
        response = requests.post(f"{base_url}/api/upload", files=files)

    upload_data = response.json()
    file_id = upload_data['file_id']
    print(f"✅ File ID: {file_id}")

    # Step 2: Convert with AI
    print("\n🤖 Converting with AI...")
    ai_payload = {
        "file_id": file_id,
        "tier": 2,
        "target_quality": 0.9
    }

    response = requests.post(f"{base_url}/api/convert-ai", json=ai_payload)
    result = response.json()

    if result.get('success'):
        # Extract the SVG content
        svg_content = result.get('svg')

        # Save to file
        output_file = "gradient_logo_ai_converted.svg"
        with open(output_file, 'w') as f:
            f.write(svg_content)

        print(f"✅ Conversion successful!")
        print(f"📄 SVG saved to: {output_file}")
        print(f"📊 Processing tier used: {result.get('ai_metadata', {}).get('tier_used')}")
        print(f"⏱️ Processing time: {result.get('processing_time', 0):.3f} seconds")

        # Show parameters used
        params = result.get('ai_metadata', {}).get('conversion', {}).get('parameters_used', {})
        if params:
            print(f"\n🔧 VTracer parameters used:")
            for key, value in params.items():
                print(f"   - {key}: {value}")

        print(f"\n📁 You can open '{output_file}' in a browser to see the result")
        print(f"   or compare with original: {test_image}")

        # Also show the raw SVG content
        print(f"\n📝 SVG Content (first 500 chars):")
        print(svg_content[:500])

    else:
        print(f"❌ Conversion failed: {result}")

if __name__ == "__main__":
    test_and_save_result()