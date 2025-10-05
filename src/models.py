"""
Data models for NASA Lunar Pipeline

This module contains data structures and models used throughout the pipeline
for representing image metadata and configuration.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from .enums import CameraType, DataProductType


@dataclass
class ImageMetadata:
    """Metadata structure for lunar images"""
    product_id: str
    camera_type: CameraType
    product_type: DataProductType
    timestamp: str
    resolution: Tuple[int, int]
    file_path: str
    illumination_angle: Optional[float] = None
    phase_angle: Optional[float] = None 