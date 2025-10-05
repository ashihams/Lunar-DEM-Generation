"""
Enums for NASA Lunar Pipeline

This module contains enumeration classes that define the types of data products
and camera systems used in lunar imagery processing.
"""

from enum import Enum


class DataProductType(Enum):
    """Enumeration of data product types"""
    RAW = "raw"
    CALIBRATION = "calibration"
    DERIVED = "derived"


class CameraType(Enum):
    """Enumeration of camera types"""
    WAC = "wac"  # Wide Angle Camera
    NAC = "nac"  # Narrow Angle Camera 