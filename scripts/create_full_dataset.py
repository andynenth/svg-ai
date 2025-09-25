#!/usr/bin/env python3
"""
Create a complete 50-logo test dataset for benchmarking.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import colorsys


class LogoGenerator:
    """Generate various types of logos for testing."""

    def __init__(self, base_dir='data/logos'):
        self.base_dir = Path(base_dir)
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#800080', '#FFA500', '#008000', '#000080', '#800000', '#008080',
            '#C0C0C0', '#808080', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'
        ]

    def hex_to_rgb(self, hex_color):
        """Convert hex to RGB."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_simple_geometric_logos(self, count=10):
        """Create simple geometric shape logos."""
        output_dir = self.base_dir / 'simple_geometric'
        output_dir.mkdir(parents=True, exist_ok=True)

        shapes = ['circle', 'square', 'triangle', 'pentagon', 'hexagon',
                  'star', 'diamond', 'oval', 'cross', 'arrow']

        for i in range(count):
            img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            shape = shapes[i % len(shapes)]
            color = self.hex_to_rgb(random.choice(self.colors))

            if shape == 'circle':
                draw.ellipse([64, 64, 192, 192], fill=color)
            elif shape == 'square':
                draw.rectangle([64, 64, 192, 192], fill=color)
            elif shape == 'triangle':
                draw.polygon([(128, 64), (64, 192), (192, 192)], fill=color)
            elif shape == 'pentagon':
                points = []
                for j in range(5):
                    angle = j * 72 - 90
                    x = 128 + 64 * np.cos(np.radians(angle))
                    y = 128 + 64 * np.sin(np.radians(angle))
                    points.append((x, y))
                draw.polygon(points, fill=color)
            elif shape == 'hexagon':
                points = []
                for j in range(6):
                    angle = j * 60
                    x = 128 + 64 * np.cos(np.radians(angle))
                    y = 128 + 64 * np.sin(np.radians(angle))
                    points.append((x, y))
                draw.polygon(points, fill=color)
            elif shape == 'star':
                points = []
                for j in range(10):
                    angle = j * 36 - 90
                    radius = 64 if j % 2 == 0 else 32
                    x = 128 + radius * np.cos(np.radians(angle))
                    y = 128 + radius * np.sin(np.radians(angle))
                    points.append((x, y))
                draw.polygon(points, fill=color)
            elif shape == 'diamond':
                draw.polygon([(128, 64), (192, 128), (128, 192), (64, 128)], fill=color)
            elif shape == 'oval':
                draw.ellipse([80, 64, 176, 192], fill=color)
            elif shape == 'cross':
                draw.rectangle([112, 64, 144, 192], fill=color)
                draw.rectangle([64, 112, 192, 144], fill=color)
            elif shape == 'arrow':
                draw.polygon([(128, 64), (96, 112), (112, 112), (112, 192),
                             (144, 192), (144, 112), (160, 112)], fill=color)

            img.save(output_dir / f'{shape}_{i:02d}.png')

        return count

    def create_text_based_logos(self, count=10):
        """Create text-based logos."""
        output_dir = self.base_dir / 'text_based'
        output_dir.mkdir(parents=True, exist_ok=True)

        words = ['TECH', 'CORP', 'DATA', 'CLOUD', 'AI', 'WEB', 'APP', 'NET', 'SOFT', 'CODE']

        try:
            # Try to use system font on macOS
            font_path = "/System/Library/Fonts/Helvetica.ttc"
            font_large = ImageFont.truetype(font_path, 48)
            font_small = ImageFont.truetype(font_path, 24)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        for i in range(count):
            img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            text = words[i % len(words)]
            color = self.hex_to_rgb(random.choice(self.colors))

            # Get text bbox for centering
            bbox = draw.textbbox((0, 0), text, font=font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = (256 - text_width) // 2
            y = (256 - text_height) // 2

            draw.text((x, y), text, font=font_large, fill=color)

            # Add subtitle for some
            if i % 3 == 0:
                subtitle = "SOLUTIONS"
                bbox = draw.textbbox((0, 0), subtitle, font=font_small)
                sub_width = bbox[2] - bbox[0]
                sub_x = (256 - sub_width) // 2
                draw.text((sub_x, y + text_height + 10), subtitle,
                         font=font_small, fill=color)

            img.save(output_dir / f'text_{text.lower()}_{i:02d}.png')

        return count

    def create_gradient_logos(self, count=10):
        """Create logos with gradients."""
        output_dir = self.base_dir / 'gradients'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            # Create radial gradient effect with concentric circles
            base_color = self.hex_to_rgb(self.colors[i % len(self.colors)])

            for j in range(10):
                factor = 1 - (j * 0.08)
                color = tuple(int(c * factor) for c in base_color)
                size = 192 - j * 16
                offset = (256 - size) // 2
                draw.ellipse([offset, offset, offset + size, offset + size], fill=color)

            img.save(output_dir / f'gradient_radial_{i:02d}.png')

        return count

    def create_complex_logos(self, count=10):
        """Create complex multi-element logos."""
        output_dir = self.base_dir / 'complex'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            # Create combinations of shapes
            num_elements = random.randint(3, 6)

            for j in range(num_elements):
                color = self.hex_to_rgb(random.choice(self.colors))
                shape_type = random.choice(['circle', 'rect', 'triangle'])

                x1 = random.randint(20, 150)
                y1 = random.randint(20, 150)
                x2 = x1 + random.randint(30, 80)
                y2 = y1 + random.randint(30, 80)

                if shape_type == 'circle':
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
                elif shape_type == 'rect':
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    draw.polygon([(x1, y2), (x2, y2), ((x1+x2)//2, y1)], fill=color)

            img.save(output_dir / f'complex_multi_{i:02d}.png')

        return count

    def create_abstract_logos(self, count=10):
        """Create abstract artistic logos."""
        output_dir = self.base_dir / 'abstract'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            pattern = i % 5

            if pattern == 0:  # Spiral
                for j in range(20):
                    angle = j * 20
                    radius = j * 5
                    x = 128 + radius * np.cos(np.radians(angle))
                    y = 128 + radius * np.sin(np.radians(angle))
                    color = self.hex_to_rgb(self.colors[j % len(self.colors)])
                    draw.ellipse([x-10, y-10, x+10, y+10], fill=color)

            elif pattern == 1:  # Waves
                for j in range(5):
                    y = 50 + j * 40
                    color = self.hex_to_rgb(self.colors[j % len(self.colors)])
                    points = []
                    for x in range(0, 257, 8):
                        wave_y = y + 20 * np.sin(x * 0.05)
                        points.append((x, wave_y))
                    for k in range(len(points) - 1):
                        draw.line([points[k], points[k+1]], fill=color, width=3)

            elif pattern == 2:  # Dots pattern
                for x in range(32, 256, 32):
                    for y in range(32, 256, 32):
                        color = self.hex_to_rgb(random.choice(self.colors))
                        radius = random.randint(4, 12)
                        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

            elif pattern == 3:  # Lines pattern
                for j in range(10):
                    x1 = random.randint(0, 256)
                    y1 = random.randint(0, 256)
                    x2 = random.randint(0, 256)
                    y2 = random.randint(0, 256)
                    color = self.hex_to_rgb(self.colors[j % len(self.colors)])
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(2, 5))

            else:  # Bezier-like curves
                for j in range(6):
                    points = []
                    for k in range(4):
                        points.append((random.randint(20, 236), random.randint(20, 236)))
                    color = self.hex_to_rgb(self.colors[j % len(self.colors)])
                    # Approximate bezier with lines
                    for t in range(0, 21):
                        t = t / 20.0
                        x = (1-t)**3 * points[0][0] + 3*(1-t)**2*t * points[1][0] + \
                            3*(1-t)*t**2 * points[2][0] + t**3 * points[3][0]
                        y = (1-t)**3 * points[0][1] + 3*(1-t)**2*t * points[1][1] + \
                            3*(1-t)*t**2 * points[2][1] + t**3 * points[3][1]
                        if t > 0:
                            draw.line([(prev_x, prev_y), (x, y)], fill=color, width=2)
                        prev_x, prev_y = x, y

            img.save(output_dir / f'abstract_{pattern}_{i:02d}.png')

        return count

    def generate_all(self):
        """Generate all 50 logos."""
        total = 0

        print("📊 Creating 50-logo test dataset...")
        print("-" * 40)

        count = self.create_simple_geometric_logos(10)
        print(f"✓ Created {count} simple geometric logos")
        total += count

        count = self.create_text_based_logos(10)
        print(f"✓ Created {count} text-based logos")
        total += count

        count = self.create_gradient_logos(10)
        print(f"✓ Created {count} gradient logos")
        total += count

        count = self.create_complex_logos(10)
        print(f"✓ Created {count} complex logos")
        total += count

        count = self.create_abstract_logos(10)
        print(f"✓ Created {count} abstract logos")
        total += count

        print("-" * 40)
        print(f"✅ Generated {total} logos total!")

        return total


import numpy as np

if __name__ == "__main__":
    generator = LogoGenerator()
    total = generator.generate_all()

    # Create dataset info
    info = f"""# Test Logo Dataset

## Statistics
- Total logos: {total}
- Categories: 5
- Logos per category: 10

## Categories

### 1. Simple Geometric (10 logos)
- Basic shapes: circles, squares, triangles, etc.
- Single solid colors
- Clean edges, no gradients

### 2. Text-Based (10 logos)
- Typography-focused designs
- Company name simulations
- Various font weights

### 3. Gradients (10 logos)
- Radial gradient effects
- Multiple color transitions
- Smooth shading

### 4. Complex (10 logos)
- Multiple overlapping shapes
- Mixed colors
- Compositional designs

### 5. Abstract (10 logos)
- Artistic patterns
- Curves and waves
- Non-geometric designs

## File Format
- Format: PNG with transparency
- Size: 256x256 pixels
- Color depth: RGBA

## Usage
Test dataset for PNG to SVG conversion benchmarking.
"""

    with open('data/logos/README.md', 'w') as f:
        f.write(info)

    print("\n📝 Dataset info saved to data/logos/README.md")