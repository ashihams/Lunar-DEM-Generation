"""
NASA Lunar Pipeline - A comprehensive pipeline for processing lunar imagery data.

This package provides classes and utilities for:
- Image preprocessing and enhancement
- Radiometric and geometric correction
- Super resolution processing
- Image stitching
- Parallel processing
- Complete pipeline orchestration
"""

__version__ = "1.0.0"
__author__ = "NASA Lunar Pipeline Team"

# NOTE:
# Keep __init__ lightweight so that external orchestrators (e.g., Airflow) can
# import package metadata without importing heavy, GPU-dependent modules.
# Avoid importing submodules that transitively require torch/opencv at import time.

__all__ = [
    '__version__',
    '__author__'
]