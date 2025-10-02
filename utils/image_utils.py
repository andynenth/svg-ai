"""
Image utility functions for safe loading and processing
"""
from PIL import Image
import io
from typing import Union, Tuple, Optional


def load_image_safe(image_path: str, background_color: Union[str, Tuple[int, int, int]] = 'white') -> Image.Image:
    """
    Safely load an image, handling transparency by compositing onto a background.

    Args:
        image_path: Path to the image file
        background_color: Background color for transparency ('white' or RGB tuple)

    Returns:
        PIL Image in RGB mode with transparency properly handled
    """
    img = Image.open(image_path)

    # Handle RGBA images with transparency
    if img.mode == 'RGBA':
        # Convert background color string to RGB tuple if needed
        if isinstance(background_color, str):
            if background_color == 'white':
                bg_color = (255, 255, 255, 255)
            elif background_color == 'black':
                bg_color = (0, 0, 0, 255)
            else:
                bg_color = (255, 255, 255, 255)  # Default to white
        else:
            # Ensure we have alpha channel
            bg_color = (*background_color, 255)

        # Create background
        background = Image.new('RGBA', img.size, bg_color)

        # Composite image over background
        img = Image.alpha_composite(background, img)

    # Convert to RGB
    return img.convert('RGB')


def load_image_bytes_safe(image_bytes: bytes, background_color: Union[str, Tuple[int, int, int]] = 'white') -> Image.Image:
    """
    Safely load an image from bytes, handling transparency.

    Args:
        image_bytes: Image data as bytes
        background_color: Background color for transparency

    Returns:
        PIL Image in RGB mode with transparency properly handled
    """
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode == 'RGBA':
        if isinstance(background_color, str):
            if background_color == 'white':
                bg_color = (255, 255, 255, 255)
            elif background_color == 'black':
                bg_color = (0, 0, 0, 255)
            else:
                bg_color = (255, 255, 255, 255)
        else:
            bg_color = (*background_color, 255)

        background = Image.new('RGBA', img.size, bg_color)
        img = Image.alpha_composite(background, img)

    return img.convert('RGB')


def has_transparency(image_path: str) -> bool:
    """
    Check if an image has transparency.

    Args:
        image_path: Path to the image file

    Returns:
        True if image has alpha channel or transparency
    """
    img = Image.open(image_path)
    return img.mode in ('RGBA', 'LA', 'P') and (
        img.mode == 'RGBA' or
        (img.mode == 'P' and 'transparency' in img.info) or
        img.mode == 'LA'
    )