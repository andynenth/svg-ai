#!/usr/bin/env python3
"""
Download test logos from free sources for testing.
Using placeholder logos that are free to use.
"""

import os
import requests
from pathlib import Path
from PIL import Image
import io


def create_test_logos():
    """Create simple test logos for different categories."""

    # Create directories
    categories = ['simple', 'text', 'gradient', 'complex', 'abstract']
    base_dir = Path('data/logos')

    for category in categories:
        (base_dir / category).mkdir(parents=True, exist_ok=True)

    print("Creating test logos...")

    # Simple geometric logos
    simple_dir = base_dir / 'simple'

    # Create circle logo
    img = Image.new('RGB', (200, 200), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill='red')
    img.save(simple_dir / 'circle.png')
    print(f"  ✓ Created: simple/circle.png")

    # Create square logo
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill='blue')
    img.save(simple_dir / 'square.png')
    print(f"  ✓ Created: simple/square.png")

    # Create triangle logo
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 50), (50, 150), (150, 150)], fill='green')
    img.save(simple_dir / 'triangle.png')
    print(f"  ✓ Created: simple/triangle.png")

    # Create diamond logo
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    draw.polygon([(100, 30), (170, 100), (100, 170), (30, 100)], fill='purple')
    img.save(simple_dir / 'diamond.png')
    print(f"  ✓ Created: simple/diamond.png")

    # Create star-like logo
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    # Simple cross/plus shape
    draw.rectangle([90, 50, 110, 150], fill='orange')
    draw.rectangle([50, 90, 150, 110], fill='orange')
    img.save(simple_dir / 'cross.png')
    print(f"  ✓ Created: simple/cross.png")

    # Text-based logos
    text_dir = base_dir / 'text'
    from PIL import ImageFont

    # Try to use a system font
    try:
        # This should work on macOS
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    # Create text logos
    texts = ['LOGO', 'BRAND', 'CORP', 'TECH', 'DESIGN']
    colors = ['black', 'navy', 'darkred', 'darkgreen', 'purple']

    for text, color in zip(texts, colors):
        img = Image.new('RGB', (200, 100), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((100, 50), text, font=font, anchor="mm", fill=color)
        img.save(text_dir / f'{text.lower()}.png')
        print(f"  ✓ Created: text/{text.lower()}.png")

    # Gradient logos (simulated with multiple colors)
    gradient_dir = base_dir / 'gradient'

    # Create gradient circles
    for i, base_color in enumerate(['red', 'blue', 'green']):
        img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(img)

        # Draw concentric circles to simulate gradient
        if base_color == 'red':
            colors = [(255, 200, 200), (255, 150, 150), (255, 100, 100), (255, 50, 50), (255, 0, 0)]
        elif base_color == 'blue':
            colors = [(200, 200, 255), (150, 150, 255), (100, 100, 255), (50, 50, 255), (0, 0, 255)]
        else:
            colors = [(200, 255, 200), (150, 255, 150), (100, 255, 100), (50, 255, 50), (0, 255, 0)]

        for j, color in enumerate(colors):
            size = 150 - j * 25
            draw.ellipse([100-size//2, 100-size//2, 100+size//2, 100+size//2], fill=color)

        img.save(gradient_dir / f'gradient_{base_color}.png')
        print(f"  ✓ Created: gradient/gradient_{base_color}.png")

    # Complex logos (combinations)
    complex_dir = base_dir / 'complex'

    # Create logo with multiple shapes
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 90, 90], fill='red')
    draw.rectangle([110, 30, 170, 90], fill='blue')
    draw.polygon([(60, 110), (30, 170), (90, 170)], fill='green')
    draw.polygon([(140, 110), (110, 170), (170, 170)], fill='yellow')
    img.save(complex_dir / 'shapes_combo.png')
    print(f"  ✓ Created: complex/shapes_combo.png")

    # Abstract logos
    abstract_dir = base_dir / 'abstract'

    # Create wave-like pattern
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    for i in range(5):
        y = 40 + i * 30
        draw.arc([20, y-20, 180, y+20], 0, 180, fill='blue', width=5)
    img.save(abstract_dir / 'waves.png')
    print(f"  ✓ Created: abstract/waves.png")

    # Create spiral-like pattern (simplified as circles)
    img = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(img)
    for i in range(5):
        size = 180 - i * 30
        offset = i * 15
        draw.ellipse([offset, offset, offset+size, offset+size], outline='purple', width=3)
    img.save(abstract_dir / 'spiral.png')
    print(f"  ✓ Created: abstract/spiral.png")

    # Count total logos
    total = sum(len(list((base_dir / cat).glob('*.png'))) for cat in categories)
    print(f"\n✅ Created {total} test logos in data/logos/")

    return total


def download_from_logoipsum():
    """Download sample logos from logoipsum (if available)."""
    base_url = "https://logoipsum.com/logo/logo-{}.svg"

    # Note: logoipsum provides SVG files, we'd need to convert them to PNG
    # For now, we'll skip this and use our generated logos
    pass


def create_dataset_info():
    """Create a dataset info file."""
    info = """# Test Logo Dataset

## Categories

### Simple (5 logos)
- Basic geometric shapes
- Single color
- Clean edges

### Text (5 logos)
- Text-based logos
- Typography focused
- Single color

### Gradient (3 logos)
- Simulated gradients
- Multiple shades
- Radial patterns

### Complex (1 logo)
- Multiple shapes
- Multiple colors
- Combined elements

### Abstract (2 logos)
- Artistic patterns
- Curves and lines
- Non-geometric

## Usage
These logos are generated for testing purposes only.
"""

    with open('data/logos/README.md', 'w') as f:
        f.write(info)

    print("📝 Created dataset README")


if __name__ == "__main__":
    print("🎨 Creating test logo dataset...")
    print("-" * 40)

    # Create test logos
    total = create_test_logos()

    # Create dataset info
    create_dataset_info()

    print("-" * 40)
    print(f"✅ Dataset ready with {total} logos!")
    print("\nYou can now test conversion with:")
    print("  python convert.py data/logos/simple/circle.png")