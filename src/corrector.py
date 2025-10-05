"""
Image Correction Classes for NASA Lunar Pipeline

This module contains classes for radiometric and geometric correction
of lunar imagery data.
"""

import numpy as np
import cv2
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RadiometricCorrector:
    """Handles radiometric correction for raw data"""

    def __init__(self):
        self.dark_current_map = None
        self.flat_field_map = None

    def apply_dark_current_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply dark current correction"""
        if self.dark_current_map is None:
            # Generate synthetic dark current map
            self.dark_current_map = np.random.normal(0, 5, image.shape).astype(np.float32)

        corrected = image.astype(np.float32) - self.dark_current_map
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def apply_flat_field_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply flat field correction"""
        if self.flat_field_map is None:
            # Generate synthetic flat field map
            h, w = image.shape[:2]
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            self.flat_field_map = 1.0 - 0.3 * (distance / max_distance)

        corrected = image.astype(np.float32) / self.flat_field_map
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def apply_radiometric_correction(self, image: np.ndarray) -> np.ndarray:
        """Complete radiometric correction pipeline"""
        image = self.apply_dark_current_correction(image)
        image = self.apply_flat_field_correction(image)
        return image


class GeometricCorrector:
    """Handles geometric correction and calibration"""

    def __init__(self, calibration_params: Dict):
        self.calibration_params = calibration_params

    def undistort_image(self, image: np.ndarray, camera_type: str) -> np.ndarray:
        """Apply lens distortion correction with size validation"""
        h, w = image.shape[:2]

        # Check if image dimensions are within OpenCV limits
        if h >= 32767 or w >= 32767:
            logger.warning(f"Image too large for undistortion ({h}x{w}), resizing first")
            # Calculate scale to fit within limits
            max_dim = 16384  # Safe limit well below 32767
            scale = min(max_dim / h, max_dim / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize image
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Apply undistortion to resized image
            undistorted_resized = self._apply_undistortion(resized_image, camera_type, scale)

            # Resize back to original size if needed
            if scale < 1.0:
                undistorted = cv2.resize(undistorted_resized, (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                undistorted = undistorted_resized
        else:
            undistorted = self._apply_undistortion(image, camera_type, 1.0)

        return undistorted

    def _apply_undistortion(self, image: np.ndarray, camera_type: str, scale: float = 1.0) -> np.ndarray:
        """Apply undistortion with scaled parameters"""
        try:
            params = self.calibration_params[camera_type]
            h, w = image.shape[:2]

            # Scale camera parameters if image was resized
            camera_matrix = np.array([
                [params['fx'] * scale, 0, params['cx'] * scale],
                [0, params['fy'] * scale, params['cy'] * scale],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array([
                params['k1'], params['k2'], params['p1'], params['p2'], 0
            ], dtype=np.float32)

            # Apply undistortion
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
            return undistorted

        except Exception as e:
            logger.warning(f"Undistortion failed: {e}. Returning original image.")
            return image 