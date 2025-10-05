"""
P‑SFS Inference Scaffold

Minimal placeholder for Photometric Shape-from-Shading deep learning inference.
Given a directory of preprocessed images, runs a PyTorch model to produce
elevation/DEM and confidence maps. This is a scaffold: replace the model
definition and post-processing with your actual implementation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class DummyPSFSNet(nn.Module):
    """Replace with real CNN+Transformer P‑SFS model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output channels: [0] = elevation, [1] = confidence (logit)
        return self.conv(x)


def _load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def run_psfs_inference(
    input_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 4,
) -> Tuple[int, str]:
    """Run P‑SFS inference over all images in input_dir.

    Returns: (processed_count, note)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading P‑SFS model from: {model_path}")
        model = torch.jit.load(model_path, map_location=device)
    else:
        logger.warning("PSFS_MODEL_PATH not set or missing; using DummyPSFSNet scaffold")
        model = DummyPSFSNet()
    model = model.to(device)
    model.eval()

    # Collect images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(Path(input_dir).glob(ext))
    files = [str(f) for f in files]

    if not files:
        logger.warning(f"No images found in {input_dir}")
        return 0, "no_files"

    processed = 0
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch_paths = files[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                g = _load_image_gray(p)
                # Normalize to [0,1]
                t = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0) / 255.0
                imgs.append(t)
            batch = torch.cat(imgs, dim=0).to(device)

            pred = model(batch)  # [B,2,H,W]
            elev = pred[:, 0]
            conf = torch.sigmoid(pred[:, 1])

            for j, p in enumerate(batch_paths):
                e = elev[j].cpu().numpy()
                c = conf[j].cpu().numpy()
                # Scale elevation to 16-bit for demo; real mapping depends on training
                e_scaled = (e - e.min()) / (e.max() - e.min() + 1e-6)
                e_u16 = (e_scaled * 65535.0).astype(np.uint16)
                c_u8 = (c * 255.0).astype(np.uint8)

                out_base = Path(output_dir) / Path(p).with_suffix("").name
                cv2.imwrite(str(out_base) + "_elevation.tif", e_u16)
                cv2.imwrite(str(out_base) + "_confidence.png", c_u8)
                processed += 1

    return processed, "ok"


