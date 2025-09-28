#!/usr/bin/env python3
"""
Create test image fixtures for integration tests.

Generates sample images of different types and complexities for testing
the conversion pipeline across various scenarios.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os


class TestImageGenerator:
    """Generator for test images across different categories."""

    def __init__(self, output_dir: str):
        """Initialize with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_simple_geometric_shapes(self):
        """Create simple geometric shape images."""
        shapes_dir = self.output_dir / "simple_geometric"
        shapes_dir.mkdir(exist_ok=True)

        # Red circle on white background
        img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 150, 150], fill='red')
        img.save(shapes_dir / "red_circle.png")

        # Blue square
        img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill='blue')
        img.save(shapes_dir / "blue_square.png")

        # Green triangle
        img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(img)
        draw.polygon([(100, 50), (50, 150), (150, 150)], fill='green')
        img.save(shapes_dir / "green_triangle.png")

        # Multi-colored geometric pattern
        img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill='red')
        draw.rectangle([120, 20, 180, 80], fill='blue')
        draw.ellipse([20, 120, 80, 180], fill='green')
        draw.polygon([(120, 120), (150, 180), (180, 150)], fill='yellow')
        img.save(shapes_dir / "multi_shapes.png")

        # Black and white circle (for B&W testing)
        img = Image.new('L', (200, 200), 255)  # White background
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 150, 150], fill=0)  # Black circle
        img.save(shapes_dir / "bw_circle.png")

    def create_complex_colored_images(self):
        """Create complex colored images with gradients and many colors."""
        complex_dir = self.output_dir / "complex_colored"
        complex_dir.mkdir(exist_ok=True)

        # Radial gradient
        img = Image.new('RGB', (200, 200))
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        center = (100, 100)
        max_radius = 100

        for y in range(200):
            for x in range(200):
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                ratio = min(distance / max_radius, 1.0)
                arr[y, x] = [
                    int(255 * (1 - ratio)),  # Red decreases
                    int(255 * ratio * 0.5),  # Green increases slowly
                    int(255 * ratio)         # Blue increases
                ]

        img = Image.fromarray(arr)
        img.save(complex_dir / "radial_gradient.png")

        # Linear gradient with multiple colors
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        for x in range(200):
            ratio = x / 200
            if ratio < 0.33:
                # Red to Yellow
                t = ratio / 0.33
                arr[:, x] = [255, int(255 * t), 0]
            elif ratio < 0.66:
                # Yellow to Green
                t = (ratio - 0.33) / 0.33
                arr[:, x] = [int(255 * (1 - t)), 255, 0]
            else:
                # Green to Blue
                t = (ratio - 0.66) / 0.34
                arr[:, x] = [0, int(255 * (1 - t)), int(255 * t)]

        img = Image.fromarray(arr)
        img.save(complex_dir / "rainbow_gradient.png")

        # Complex pattern with many colors
        img = Image.new('RGB', (200, 200), 'white')
        draw = ImageDraw.Draw(img)

        # Create a complex pattern with many small colored elements
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        for i in range(50):
            x = np.random.randint(0, 180)
            y = np.random.randint(0, 180)
            size = np.random.randint(5, 20)
            color = np.random.choice(colors)
            shape = np.random.choice(['rectangle', 'ellipse'])

            if shape == 'rectangle':
                draw.rectangle([x, y, x + size, y + size], fill=color)
            else:
                draw.ellipse([x, y, x + size, y + size], fill=color)

        img.save(complex_dir / "complex_pattern.png")

        # Photo-realistic gradient (simulation)
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        for y in range(200):
            for x in range(200):
                # Create noise-like pattern
                noise = np.random.randint(-20, 20)
                base_r = int(127 + 100 * np.sin(x * 0.1) + noise)
                base_g = int(127 + 100 * np.cos(y * 0.1) + noise)
                base_b = int(127 + 100 * np.sin((x + y) * 0.05) + noise)

                arr[y, x] = [
                    np.clip(base_r, 0, 255),
                    np.clip(base_g, 0, 255),
                    np.clip(base_b, 0, 255)
                ]

        img = Image.fromarray(arr)
        img.save(complex_dir / "photo_like.png")

    def create_transparent_images(self):
        """Create images with various transparency scenarios."""
        transparent_dir = self.output_dir / "transparent"
        transparent_dir.mkdir(exist_ok=True)

        # Simple icon with transparency
        img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))  # Fully transparent
        draw = ImageDraw.Draw(img)

        # Create a simple cross/plus icon
        draw.rectangle([80, 20, 120, 180], fill=(255, 0, 0, 255))  # Vertical bar
        draw.rectangle([20, 80, 180, 120], fill=(255, 0, 0, 255))  # Horizontal bar

        img.save(transparent_dir / "red_cross_icon.png")

        # Partially transparent overlay
        img = Image.new('RGBA', (200, 200), (255, 255, 255, 255))  # White background
        draw = ImageDraw.Draw(img)

        # Semi-transparent overlapping circles
        draw.ellipse([50, 50, 120, 120], fill=(255, 0, 0, 128))    # Semi-transparent red
        draw.ellipse([80, 80, 150, 150], fill=(0, 0, 255, 128))   # Semi-transparent blue

        img.save(transparent_dir / "overlapping_circles.png")

        # Complex transparency with gradient alpha
        img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        arr = np.array(img)

        # Create gradient alpha effect
        for y in range(200):
            for x in range(200):
                distance = np.sqrt((x - 100)**2 + (y - 100)**2)
                alpha = max(0, 255 - int(distance * 2.5))
                arr[y, x] = [0, 150, 255, alpha]  # Blue with gradient alpha

        img = Image.fromarray(arr)
        img.save(transparent_dir / "gradient_alpha.png")

        # Icon with cutout (common in logos)
        img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Outer shape
        draw.ellipse([20, 20, 180, 180], fill=(0, 128, 255, 255))

        # Inner cutout (transparent)
        draw.ellipse([60, 60, 140, 140], fill=(0, 0, 0, 0))

        # Inner detail
        draw.ellipse([80, 80, 120, 120], fill=(255, 255, 0, 255))

        img.save(transparent_dir / "logo_cutout.png")

    def create_edge_case_images(self):
        """Create edge case images for stress testing."""
        edge_dir = self.output_dir / "edge_cases"
        edge_dir.mkdir(exist_ok=True)

        # 1x1 pixel image
        img = Image.new('RGB', (1, 1), 'red')
        img.save(edge_dir / "1x1_pixel.png")

        # Very small image
        img = Image.new('RGB', (5, 5), 'blue')
        img.save(edge_dir / "5x5_small.png")

        # Extremely tall/narrow image
        img = Image.new('RGB', (10, 500), 'green')
        img.save(edge_dir / "tall_narrow.png")

        # Extremely wide/short image
        img = Image.new('RGB', (500, 10), 'yellow')
        img.save(edge_dir / "wide_short.png")

        # Large image (for memory testing)
        img = Image.new('RGB', (1000, 1000), 'white')
        draw = ImageDraw.Draw(img)
        # Add some content so it's not completely empty
        draw.ellipse([200, 200, 800, 800], fill='red')
        draw.rectangle([100, 100, 900, 200], fill='blue')
        img.save(edge_dir / "large_1000x1000.png")

        # All black image
        img = Image.new('RGB', (200, 200), 'black')
        img.save(edge_dir / "all_black.png")

        # All white image
        img = Image.new('RGB', (200, 200), 'white')
        img.save(edge_dir / "all_white.png")

        # Single color with noise
        img = Image.new('RGB', (200, 200), 'red')
        arr = np.array(img)
        # Add small amount of noise
        noise = np.random.randint(-5, 5, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
        img.save(edge_dir / "red_with_noise.png")

        # High contrast B&W image
        img = Image.new('L', (200, 200), 255)
        draw = ImageDraw.Draw(img)
        # Create checkerboard pattern
        square_size = 20
        for y in range(0, 200, square_size):
            for x in range(0, 200, square_size):
                if (x // square_size + y // square_size) % 2 == 0:
                    draw.rectangle([x, y, x + square_size, y + square_size], fill=0)
        img.save(edge_dir / "checkerboard.png")

    def create_text_based_images(self):
        """Create text-based images (simulating logos with text)."""
        text_dir = self.output_dir / "text_based"
        text_dir.mkdir(exist_ok=True)

        # Simple text on white background
        img = Image.new('RGB', (300, 100), 'white')
        draw = ImageDraw.Draw(img)

        try:
            # Try to use a larger font if available
            font = ImageFont.truetype("Arial.ttf", 36)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
            except:
                font = ImageFont.load_default()

        draw.text((20, 30), "LOGO", fill='black', font=font)
        img.save(text_dir / "simple_logo_text.png")

        # Black text on transparent background
        img = Image.new('RGBA', (300, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "BRAND", fill='black', font=font)
        img.save(text_dir / "black_text_transparent.png")

        # Colored text
        img = Image.new('RGB', (300, 100), 'white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "COLOR", fill='red', font=font)
        img.save(text_dir / "red_text.png")

        # Text with outline effect (simulated)
        img = Image.new('RGB', (300, 100), 'white')
        draw = ImageDraw.Draw(img)

        # Draw outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((20 + dx, 30 + dy), "OUTLINE", fill='black', font=font)

        # Draw main text
        draw.text((20, 30), "OUTLINE", fill='white', font=font)
        img.save(text_dir / "outlined_text.png")

    def create_all_test_images(self):
        """Create all categories of test images."""
        print("Creating test image fixtures...")

        self.create_simple_geometric_shapes()
        print("✓ Simple geometric shapes created")

        self.create_complex_colored_images()
        print("✓ Complex colored images created")

        self.create_transparent_images()
        print("✓ Transparent images created")

        self.create_edge_case_images()
        print("✓ Edge case images created")

        self.create_text_based_images()
        print("✓ Text-based images created")

        print(f"\nAll test fixtures created in: {self.output_dir}")

        # Print summary
        total_images = 0
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir():
                image_count = len(list(category_dir.glob("*.png")))
                print(f"  {category_dir.name}: {image_count} images")
                total_images += image_count

        print(f"\nTotal: {total_images} test images created")


if __name__ == "__main__":
    # Create test images in fixtures directory
    current_dir = Path(__file__).parent
    generator = TestImageGenerator(current_dir / "images")
    generator.create_all_test_images()