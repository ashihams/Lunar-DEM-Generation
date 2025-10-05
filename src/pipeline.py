import os
import json
import numpy as np
import cv2
import torch

from pathlib import Path

from .solar import SolarGeometryProcessor
from .model import SFSDeepNetwork, BayesianUncertaintyEstimator
from .refinement import SubPixelRefinement
from .dem import PoissonSurfaceReconstructor, LOLAAnchorIntegration, GlobalRegistration


class DEMGenerationPipeline:
    def __init__(self, device='cuda'):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.sfs_model = SFSDeepNetwork(in_channels=3).to(self.device)
        self.uncertainty_estimator = BayesianUncertaintyEstimator(self.sfs_model)
        self.solar = SolarGeometryProcessor()
        self.refiner = SubPixelRefinement(device=self.device)
        self.reconstructor = PoissonSurfaceReconstructor()
        self.lola = LOLAAnchorIntegration()
        self.reg = GlobalRegistration()

    def prepare_input(self, image, metadata):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_norm = image.astype(np.float32) / 255.0
        incidence, azimuth = self.solar.calculate_solar_angles(metadata)
        h, w = image.shape[:2]
        incidence = cv2.resize(incidence, (w, h))
        azimuth = cv2.resize(azimuth, (w, h))
        input_array = np.stack([image_norm[:, :, 0], incidence, azimuth], axis=2)
        tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def run(self, image, metadata, output_dir, use_refinement=True):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        x = self.prepare_input(image, metadata)
        with torch.no_grad():
            dzdx, dzdy, conf = self.sfs_model(x)
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(x)
        dzdx_np = dzdx.squeeze().cpu().numpy()
        dzdy_np = dzdy.squeeze().cpu().numpy()
        conf_np = conf.squeeze().cpu().numpy()
        if use_refinement:
            dzdx_np, dzdy_np = self.refiner.refine_slopes(dzdx_np, dzdy_np)
        dem = self.reconstructor.integrate_slopes(dzdx_np, dzdy_np)
        np.save(Path(output_dir) / 'dem.npy', dem)
        np.save(Path(output_dir) / 'confidence.npy', conf_np)
        with open(Path(output_dir) / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        return {
            'dem': dem,
            'confidence': conf_np,
            'slopes': (dzdx_np, dzdy_np)
        }


