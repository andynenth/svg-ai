# scripts/create_test_images.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def create_test_images():
    """Create diverse test images for comprehensive testing"""

    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create simple geometric logo
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Draw a simple circle with border
    draw.ellipse([50, 50, 150, 150], fill='blue', outline='black', width=3)
    img.save(test_dir / "simple_geometric.png")
    print("✅ Created simple_geometric.png")

    # Create text-based logo
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)

    # Try to use default font
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # Draw text
    text = "LOGO"
    if font:
        draw.text((50, 30), text, fill='black', font=font)
    else:
        draw.text((50, 30), text, fill='black')

    img.save(test_dir / "text_based.png")
    print("✅ Created text_based.png")

    # Create gradient logo
    img = Image.new('RGB', (200, 200), color='white')
    # Create gradient effect
    for y in range(200):
        for x in range(200):
            # Simple gradient from blue to red
            r = int(255 * (x / 200))
            b = int(255 * (1 - x / 200))
            img.putpixel((x, y), (r, 0, b))

    img.save(test_dir / "gradient_logo.png")
    print("✅ Created gradient_logo.png")

    # Create complex design
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)

    # Multiple shapes and colors
    draw.rectangle([20, 20, 80, 80], fill='red', outline='black')
    draw.ellipse([100, 20, 180, 100], fill='green', outline='blue', width=2)
    draw.polygon([(100, 120), (150, 180), (50, 180)], fill='yellow', outline='purple')

    img.save(test_dir / "complex_design.png")
    print("✅ Created complex_design.png")

    # Create corrupted image (small truncated file)
    with open(test_dir / "corrupted_image.png", 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00')  # Incomplete PNG header

    print("✅ Created corrupted_image.png")

    print(f"📁 All test images created in {test_dir}/")

if __name__ == "__main__":
    create_test_images()
