import os
import math
import json
import numpy as np
import rasterio
import xml.etree.ElementTree as ET


def _safe_get_text(elem):
    if elem is None or elem.text is None:
        return None
    text = elem.text.strip()
    return text if text != "" else None


def _find_with_alternates(root, candidates, namespaces):
    for xpath in candidates:
        found = root.find(xpath, namespaces)
        if found is not None and _safe_get_text(found) is not None:
            return found
    return None


def _extract_angles_from_xml(xml_path, img_shape):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Common PDS4 namespaces. If label lacks namespaces, lookups without ns will handle below.
        ns = {
            'pds': 'http://pds.nasa.gov/pds4/pds/v1',
            'disp': 'http://pds.nasa.gov/pds4/disp/v1',
            'img': 'http://pds.nasa.gov/pds4/img/v1'
        }

        # Likely paths for mean incidence/emission angles in PDS4 labels.
        incidence_candidates = [
            './/geometry_statistics/incidence_angle/mean',
            './/img:Geometry/img:Incidence_Angle/img:mean',
            './/pds:geometry_statistics/pds:incidence_angle/pds:mean',
        ]
        emission_candidates = [
            './/geometry_statistics/emission_angle/mean',
            './/img:Geometry/img:Emission_Angle/img:mean',
            './/pds:geometry_statistics/pds:emission_angle/pds:mean',
        ]

        inc_node = _find_with_alternates(root, incidence_candidates, ns) or root.find('.//incidence_angle')
        emi_node = _find_with_alternates(root, emission_candidates, ns) or root.find('.//emission_angle')

        i_deg = float(_safe_get_text(inc_node)) if inc_node is not None else 45.0
        e_deg = float(_safe_get_text(emi_node)) if emi_node is not None else 0.0

        i_map = np.full(img_shape, np.deg2rad(i_deg), dtype=np.float64)
        e_map = np.full(img_shape, np.deg2rad(e_deg), dtype=np.float64)
        return i_map, e_map
    except Exception:
        # Fall back to placeholders; caller may try JSON next
        i_map = np.full(img_shape, np.deg2rad(45.0), dtype=np.float64)
        e_map = np.full(img_shape, np.deg2rad(0.0), dtype=np.float64)
        return i_map, e_map


def _extract_angles_from_json(json_path, img_shape):
    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)

        i_deg = None
        e_deg = None

        # Try common keys
        for key in ('incidence_angle', 'incidence', 'i_deg', 'i'):
            if key in meta:
                i_deg = float(meta[key])
                break
        for key in ('emission_angle', 'emission', 'e_deg', 'e'):
            if key in meta:
                e_deg = float(meta[key])
                break

        i_deg = 45.0 if i_deg is None else i_deg
        e_deg = 0.0 if e_deg is None else e_deg

        i_map = np.full(img_shape, np.deg2rad(i_deg), dtype=np.float64)
        e_map = np.full(img_shape, np.deg2rad(e_deg), dtype=np.float64)
        return i_map, e_map
    except Exception:
        i_map = np.full(img_shape, np.deg2rad(45.0), dtype=np.float64)
        e_map = np.full(img_shape, np.deg2rad(0.0), dtype=np.float64)
        return i_map, e_map


def read_intensity_and_geometry(tif_path, xml_path=None, json_path=None):
    """
    Read intensity from GeoTIFF and derive incidence (i) and emission (e) maps.

    Returns:
        I (np.ndarray float64), i_map (np.ndarray float64, radians), e_map (np.ndarray float64, radians)
        or (None, None, None) on failure to read intensity.
    """
    try:
        with rasterio.open(tif_path) as src:
            intensity_array = src.read(1).astype(np.float64)
            img_shape = intensity_array.shape
    except Exception:
        return None, None, None

    i_map = None
    e_map = None

    if xml_path and os.path.exists(xml_path):
        i_map, e_map = _extract_angles_from_xml(xml_path, img_shape)

    # If XML did not provide, try JSON
    if (i_map is None or e_map is None) and json_path and os.path.exists(json_path):
        i_map, e_map = _extract_angles_from_json(json_path, img_shape)

    # Final safety fallback
    if i_map is None or e_map is None:
        i_map = np.full(img_shape, np.deg2rad(45.0), dtype=np.float64)
        e_map = np.full(img_shape, np.deg2rad(0.0), dtype=np.float64)

    return intensity_array, i_map, e_map


def run_photoclinometry(intensity_array, incidence_map, emission_map):
    """
    Placeholder for photoclinometry model; returns a zero DEM of same shape.
    """
    if intensity_array is None or incidence_map is None or emission_map is None:
        return None
    dem_array = np.zeros_like(intensity_array, dtype=np.float64)
    return dem_array


def demo_run(file_id, data_dir):
    tif_path = os.path.join(data_dir, f"thumb_wac.{file_id}.TIF")
    xml_path = os.path.join(data_dir, f"thumb_wac.{file_id}.XML")
    json_path = os.path.join(data_dir, "Tycho_metadata.json")

    if not os.path.exists(tif_path):
        print(f"Missing TIF: {tif_path}")
        return None

    intensity, i_map, e_map = read_intensity_and_geometry(tif_path, xml_path if os.path.exists(xml_path) else None, json_path if os.path.exists(json_path) else None)
    if intensity is None:
        print("Failed to read intensity")
        return None

    dem = run_photoclinometry(intensity, i_map, e_map)
    return {
        'intensity': intensity,
        'incidence': i_map,
        'emission': e_map,
        'dem': dem,
    }


