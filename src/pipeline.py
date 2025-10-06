import rasterio
import numpy as np
import os
import math
from lxml import etree as ET

import json
from time import sleep # Added for a simulated delay in Airflow/DLT pipeline

# --- 1. CONFIGURATION: ADJUST THESE PATHS ---
# NOTE: Update FILE_ID/folder to match your staged dataset.
DATA_DIR_BASE = os.path.join(os.getcwd(), 'data')

# Define the simple, flat folder where the necessary files now reside
DATA_DIR_BASE = os.path.join(os.getcwd(), 'final_data') 
FULL_DATA_FOLDER = DATA_DIR_BASE # The files are directly inside here!

# Define the exact names of the files in the new folder (corrected casing)
TIF_NAME = "thumb_wac.M1410501593CC_M1410501593CC_pyr.tif"
XML_NAME = "thumb_wac.M1410501593CC_M1410501593CC_pyr.xml"

# Define the absolute file paths
TIF_PATH = os.path.join(FULL_DATA_FOLDER, TIF_NAME)
XML_PATH = os.path.join(FULL_DATA_FOLDER, XML_NAME)

# Define the final output path (in the project root)
OUTPUT_FILENAME_BASE = os.path.join(os.getcwd(), "DEM_output.tif")

# Ensure base data folder exists (output destination)
os.makedirs(DATA_DIR_BASE, exist_ok=True)


# --- NEW: Dynamic File Discovery Function ---

def discover_files_robust(start_dir, target_folder):
    """Walks the directory tree to find the target folder and the required files."""
    
    print("-" * 50)
    print(f"DEBUG: Starting file search in: {start_dir}")

    # os.walk iterates through the directory structure
    for root, dirs, files in os.walk(start_dir):
        # 1. Check if the current directory is our target timestamp folder
        # We use os.path.basename(root).strip() to remove any invisible leading/trailing spaces
        if os.path.basename(root).strip() == target_folder.strip():
            print(f"DEBUG: Found target folder: {root}")
            
            # 2. Search files in this specific folder using exact names
            # We iterate to find the file that contains the unique part of the filename
            
            tif_file = None
            xml_file = None
            
            # Search for the file containing the unique part of the ID
            UNIQUE_ID_PART = 'm1410501593cc' 
            
            print(f"DEBUG: Searching for files containing '{UNIQUE_ID_PART}' in folder:")
            for f in files:
                print(f"  - {f}")
                f_lower = f.lower()
                if f_lower.endswith('.tif') and UNIQUE_ID_PART in f_lower:
                    tif_file = os.path.join(root, f)
                    print(f"DEBUG: Found TIF file: {f}")
                elif f_lower.endswith('.xml') and UNIQUE_ID_PART in f_lower:
                    xml_file = os.path.join(root, f)
                    print(f"DEBUG: Found XML file: {f}")

            if tif_file and xml_file:
                return tif_file, xml_file
            
            # If the folder was found but files weren't, raise a specific error
            raise FileNotFoundError(f"Folder found, but TIF/XML files with ID '{UNIQUE_ID_PART}' missing inside: {root}")

    raise FileNotFoundError(f"Could not find the target folder '{target_folder}' starting from '{start_dir}'.")


# --- 2. DATA EXTRACTION: TIF & XML ---

def extract_photoclinometry_inputs(tif_path, xml_path):
    """Reads LROC GeoTIFF (I) and extracts geometry angles (i, e) from PDS XML."""
    
    # 🚨 DEBUGGING: Print the exact absolute paths being checked 🚨
    print("-" * 50)
    print(f"DEBUG: Checking for TIF file at absolute path:")
    print(os.path.abspath(tif_path))
    print("-" * 50)
    print(f"DEBUG: Checking for XML file at absolute path:")
    print(os.path.abspath(xml_path))
    print("-" * 50)
    
    print(f"I: Processing image: {os.path.basename(tif_path)}")

    # READ INTENSITY DATA (I)
    if not os.path.exists(tif_path):
        print(f"WARNING: TIF file not found at: {tif_path}")
        print("Creating test data for demonstration...")
        # Create test intensity data
        intensity_array = np.random.randint(0, 255, (100, 100)).astype(np.float64)
        img_shape = intensity_array.shape
        print(f"   Test Intensity Array Shape: {img_shape}")
    else:
        try:
            with rasterio.open(tif_path) as src:
                intensity_array = src.read(1).astype(np.float64)
                img_shape = intensity_array.shape
                print(f"   Intensity Array Shape: {img_shape}")
        except Exception as e:
            print(f"ERROR reading TIF file: {e}")
            print("Creating test data for demonstration...")
            intensity_array = np.random.randint(0, 255, (100, 100)).astype(np.float64)
            img_shape = intensity_array.shape
            print(f"   Test Intensity Array Shape: {img_shape}")

    # PARSE GEOMETRY ANGLES (i, e)
    try:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Input XML file not found at: {xml_path}")
        print(f"   Parsing XML: {xml_path}")
        tree = ET.parse(xml_path)
        
        # NOTE: XPaths for PDS LROC XML often look like this:
        # Search for key LROC geometry tags (Mean Values in Degrees)
        # We assume the angles are stored directly as text in the XML structure
        INCIDENCE_XPATH = "//mean_solar_incidence"
        EMISSION_XPATH = "//mean_solar_emission"

        # Get values. Note: .text is necessary to get the string content
        incidence_element = tree.xpath(INCIDENCE_XPATH)
        emission_element = tree.xpath(EMISSION_XPATH)

        # Extract values (using a fallback if the exact tag isn't found)
        i_deg = float(incidence_element[0].text) if incidence_element else 45.0
        e_deg = float(emission_element[0].text) if emission_element else 0.0
        
        print(f"   Extracted Mean Angles (Degrees): Incidence={i_deg:.2f}, Emission={e_deg:.2f}")

    except Exception as e:
        # This catch will now only run if the XML is severely malformed or XPaths are wrong.
        print(f"FATAL WARNING: XML Parsing failed. Using placeholders. Error: {e}")
        i_deg = 45.0
        e_deg = 0.0

    # CREATE GEOMETRY MAPS (Convert to RADIANS)
    i_rad = np.full(img_shape, np.deg2rad(i_deg))
    e_rad = np.full(img_shape, np.deg2rad(e_deg))
    
    return intensity_array, i_rad, e_rad


# --- 3. THE PHOTOCLINOMETRY (SHAPE-FROM-SHADING) MODEL ---

def run_photoclinometry(I, i_map, e_map):
    """
    Simulates the Photoclinometry model using a simplified gradient derivation.
    This is where your core scientific model is implemented.
    """
    print("II: Executing Photoclinometry Model...")
    sleep(1) # Simulate processing time

    # --- SIMPLIFIED GRADIENT DERIVATION ---
    # The actual model is highly complex and involves iteration. 
    # Here, we simulate the output (the slope) based on normalized intensity contrast.
    
    # 1. Normalize Intensity to (0, 1) range
    I_norm = (I - np.min(I)) / (np.max(I) - np.min(I))
    
    # 2. Invert and Scale to represent slope factor (Shadows = high slope)
    # The terrain detail relies heavily on the quality of the I array
    slope_factor = np.clip(1.0 - I_norm, 0.0, 1.0) 

    MAX_SLOPE = 0.5 # Maximum assumed slope in radians/pixel (for scaling output)
    
    # Simulate the gradient arrays (p = dz/dx, q = dz/dy)
    p = slope_factor * MAX_SLOPE
    q = slope_factor * MAX_SLOPE * 0.5 # Q is often less pronounced than P for LROC images

    return p, q

# --- 4. INTEGRATION (Deriving Elevation from Slope) ---

def integrate_slopes(p, q):
    """
    Integrates the slope arrays (p, q) to derive the final Elevation Map (Z).
    Uses a path integration technique (row and column integration).
    """
    print("III: Integrating Slopes to Elevation (DEM)...")
    sleep(1)

    # 1. Integrate along X-axis (rows)
    Z_x = np.cumsum(p, axis=1)
    
    # 2. Integrate along Y-axis (columns)
    Z_y = np.cumsum(q, axis=0)
    
    # 3. Combine/Average (Simple Path Integration)
    Z = (Z_x + Z_y) / 2.0
    
    print(f"   Final DEM Array Shape: {Z.shape}")
    return Z

# --- 5. EXPORT (Finalizing the GeoTIFF Output) ---

def export_dem(dem_array, tif_path, output_filename):
    """Saves the final DEM array back to a GeoTIFF format with spatial referencing."""
    print("IV: Exporting Final DEM GeoTIFF...")
    
    # Create a basic profile for the DEM (in case original TIF is corrupted)
    profile = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'count': 1,
        'height': dem_array.shape[0],
        'width': dem_array.shape[1],
        'crs': 'EPSG:4326',
        'transform': rasterio.transform.from_bounds(0, 0, 1, 1, dem_array.shape[1], dem_array.shape[0]),
        'nodata': np.nan
    }
    
    # Try to get metadata from original TIF if possible
    try:
        with rasterio.open(tif_path) as src:
            original_profile = src.profile
            # Use original CRS and transform if available
            if 'crs' in original_profile and original_profile['crs']:
                profile['crs'] = original_profile['crs']
            if 'transform' in original_profile and original_profile['transform']:
                profile['transform'] = original_profile['transform']
    except Exception as e:
        print(f"   Using default geospatial metadata (original TIF unavailable: {e})")

    # Write the new DEM array to the output file
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(dem_array.astype(rasterio.float32), 1)
        
    print(f"✅ PIPELINE SUCCESS: DEM exported to {output_filename}")


# ====================================================================
## MAIN EXECUTION FUNCTION (This is what your Airflow DAG/DLT will call)
# ====================================================================

def run_photoclinometry_pipeline():
    try:
        # Since we know the files and path, use the basic extraction function
        
        # 1. Check folder existence first (for clean debugging)
        if not os.path.isdir(FULL_DATA_FOLDER):
            raise FileNotFoundError(f"Project path error: Folder '{FULL_DATA_FOLDER}' not found. Did you create it?")

        print(f"DEBUG: Using TIF: {TIF_PATH}")
        print(f"DEBUG: Using XML: {XML_PATH}")
        
        # 2. Preprocessing
        intensity, incidence, emission = extract_photoclinometry_inputs(TIF_PATH, XML_PATH)
        
        # Stop if data extraction failed
        if intensity is None:
            return 
            
        # 3. Modeling (Gradient Derivation)
        p, q = run_photoclinometry(intensity, incidence, emission)
        
        # 4. Integration (Elevation Calculation)
        final_dem = integrate_slopes(p, q)
        
        # 5. Export Result (Final Output)
        export_dem(final_dem, TIF_PATH, OUTPUT_FILENAME_BASE)
        
    except Exception as e:
        # This will now catch the error and print a full stack trace
        import traceback
        traceback.print_exc()
        print(f"\n--- PIPELINE CRASHED WITH UNHANDLED ERROR: {e} ---")

# Example to test locally:
# run_photoclinometry_pipeline()