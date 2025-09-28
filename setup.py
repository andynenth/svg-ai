#!/usr/bin/env python3
"""
Setup script for SVG-AI package.
"""

from setuptools import setup, find_packages

setup(
    name="svg-ai",
    version="1.0.0",
    description="PNG to SVG converter using VTracer and other vectorization tools",
    author="SVG-AI Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies
        "vtracer",
        "Pillow",
        "numpy",
        "flask",
        "flask-cors",

        # Optional dependencies for additional functionality
        "opencv-python",
        "scipy",
        "scikit-image",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
        ],
        "test": [
            "pytest",
            "pytest-asyncio",
        ],
    },
    entry_points={
        "console_scripts": [
            "svg-ai=backend.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)