"""
Image Processing Classes for NASA Lunar Pipeline

This module contains classes for super resolution processing and image stitching
of lunar imagery data.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import logging
from typing import List

logger = logging.getLogger(__name__)


class SuperResolutionProcessor:
    """GPU-accelerated super resolution for WAC images"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def enhance_resolution(self, image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """Apply super resolution using CUDA"""
        # Convert to tensor and move to GPU
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize input for super resolution
        h, w = image.shape
        low_res = cv2.resize(image, (w // scale_factor, h // scale_factor))

        # Prepare tensor
        input_tensor = self.transform(low_res).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.clamp(output, -1, 1)
            output = (output + 1) / 2  # Denormalize

        # Convert back to numpy
        enhanced = output.squeeze().cpu().numpy()
        enhanced = (enhanced * 255).astype(np.uint8)

        # Resize to target resolution
        enhanced = cv2.resize(enhanced, (w, h))
        return enhanced


class ImageStitcher:
    """High-performance image stitching for NAC images"""

    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    def stitch_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Stitch multiple NAC images together"""
        if len(images) < 2:
            return images[0] if images else None

        # Start with first image
        result = images[0]

        for i in range(1, len(images)):
            result = self._stitch_pair(result, images[i])

        return result

    def _stitch_pair(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Stitch two images together using feature matching"""
        # Convert to grayscale if needed
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return img1  # Return first image if no features found

        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return img1  # Not enough matches

        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return img1

        # Warp and stitch
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Transform corners of first image
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners1_transformed = cv2.perspectiveTransform(corners1, H)

        # Calculate canvas size
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners1_transformed, corners2), axis=0)

        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

        # Translation matrix
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        # Warp first image
        warped1 = cv2.warpPerspective(img1, translation @ H, (x_max - x_min, y_max - y_min))

        # Place second image
        result = warped1.copy()
        result[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2

        return result 