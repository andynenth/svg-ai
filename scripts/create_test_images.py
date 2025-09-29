#!/usr/bin/env python3
"""
Create Test Images for E2E Testing
Generates basic test images for each logo type category
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_simple_geometric_logo(size=(200, 200)):
    """Create a simple geometric logo"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    # Draw a simple blue circle
    center = (size[0]//2, size[1]//2)
    radius = min(size)//3

    draw.ellipse([
        center[0] - radius, center[1] - radius,
        center[0] + radius, center[1] + radius
    ], fill='blue', outline='darkblue', width=3)

    return img

def create_text_based_logo(size=(200, 200)):
    """Create a text-based logo"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fallback to default
    try:
        font = ImageFont.truetype("Arial.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        except:
            font = ImageFont.load_default()

    # Draw text
    text = "LOGO"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    draw.text((x, y), text, fill='black', font=font)

    # Add underline
    draw.rectangle([x, y + text_height + 5, x + text_width, y + text_height + 10], fill='red')

    return img

def create_gradient_logo(size=(200, 200)):
    """Create a gradient logo"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    # Create gradient effect
    for y in range(size[1]):
        # Create gradient from blue to red
        ratio = y / size[1]
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        g = 50

        color = (r, g, b)
        draw.line([(0, y), (size[0], y)], fill=color)

    # Add a white circle in center
    center = (size[0]//2, size[1]//2)
    radius = min(size)//4

    draw.ellipse([
        center[0] - radius, center[1] - radius,
        center[0] + radius, center[1] + radius
    ], fill='white', outline='gray', width=2)

    return img

def create_complex_logo(size=(200, 200)):
    """Create a complex logo with multiple elements"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(size[1]):
        ratio = y / size[1]
        gray_value = int(240 - 40 * ratio)
        color = (gray_value, gray_value, gray_value)
        draw.line([(0, y), (size[0], y)], fill=color)

    # Multiple shapes
    # Rectangle
    draw.rectangle([20, 20, 80, 80], fill='blue', outline='darkblue', width=2)

    # Circle
    draw.ellipse([120, 20, 180, 80], fill='red', outline='darkred', width=2)

    # Triangle (polygon)
    triangle_points = [(100, 120), (60, 180), (140, 180)]
    draw.polygon(triangle_points, fill='green', outline='darkgreen')

    # Some text
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    draw.text((50, 90), "COMPLEX", fill='black', font=font)

    return img

def main():
    """Generate all test images"""
    # Ensure test directory exists
    test_dir = Path('data/test')
    test_dir.mkdir(parents=True, exist_ok=True)

    # Generate test images
    test_images = {
        'simple_geometric_logo.png': create_simple_geometric_logo(),
        'text_based_logo.png': create_text_based_logo(),
        'gradient_logo.png': create_gradient_logo(),
        'complex_logo.png': create_complex_logo()
    }

    print("Creating test images for E2E testing...")

    for filename, img in test_images.items():
        filepath = test_dir / filename
        img.save(filepath, 'PNG')
        print(f"✅ Created: {filepath}")

    print(f"\nTest images created in {test_dir}")
    print("Ready for E2E testing!")

if __name__ == "__main__":
    from pathlib import Path
    main()