import numpy as np


class SolarGeometryProcessor:
    """Prepares solar geometry maps for SFS."""

    def __init__(self):
        self.sun_distance_au = 1.0

    def calculate_solar_angles(self, metadata):
        h, w = metadata['resolution']
        solar_zenith = metadata.get('solar_zenith_angle', 30.0)
        solar_azimuth = metadata.get('solar_azimuth_angle', 180.0)
        incidence_map = np.full((h, w), np.radians(solar_zenith), dtype=np.float32)
        azimuth_map = np.full((h, w), np.radians(solar_azimuth), dtype=np.float32)
        return incidence_map, azimuth_map


