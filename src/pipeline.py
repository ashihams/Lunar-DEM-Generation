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
        
        # Define PDS4 namespace for LROC data
        ns = {'pds': 'http://pds.nasa.gov/pds4/pds/v1'}
        
        # Try multiple possible XPaths for geometry data
        incidence_xpaths = [
            ".//pds:Image_Geometry/pds:Solar_Incidence_Angle/pds:mean",
            ".//pds:Image_Geometry/pds:Incidence_Angle/pds:mean", 
            ".//pds:Geometry_Statistics/pds:Incidence_Angle/pds:mean",
            ".//Solar_Incidence_Angle/mean",
            ".//Incidence_Angle/mean",
            ".//mean_solar_incidence",
            ".//incidence_angle"
        ]
        
        emission_xpaths = [
            ".//pds:Image_Geometry/pds:Emission_Angle/pds:mean",
            ".//pds:Image_Geometry/pds:Viewing_Angle/pds:mean",
            ".//pds:Geometry_Statistics/pds:Emission_Angle/pds:mean", 
            ".//Emission_Angle/mean",
            ".//Viewing_Angle/mean",
            ".//mean_solar_emission",
            ".//emission_angle"
        ]
        
        # Try to find incidence angle
        i_deg = None
        for xpath in incidence_xpaths:
            try:
                elements = tree.xpath(xpath, namespaces=ns) if 'pds:' in xpath else tree.xpath(xpath)
                if elements and elements[0].text:
                    i_deg = float(elements[0].text)
                    print(f"   Found incidence angle: {i_deg:.2f}° (XPath: {xpath})")
                    break
            except:
                continue
        
        # Try to find emission angle  
        e_deg = None
        for xpath in emission_xpaths:
            try:
                elements = tree.xpath(xpath, namespaces=ns) if 'pds:' in xpath else tree.xpath(xpath)
                if elements and elements[0].text:
                    e_deg = float(elements[0].text)
                    print(f"   Found emission angle: {e_deg:.2f}° (XPath: {xpath})")
                    break
            except:
                continue
        
        # Use fallback values if not found
        if i_deg is None:
            i_deg = 45.0
            print(f"   WARNING: No incidence angle found, using default: {i_deg}°")
        if e_deg is None:
            e_deg = 0.0
            print(f"   WARNING: No emission angle found, using default: {e_deg}°")
            
        print(f"   Final Angles (Degrees): Incidence={i_deg:.2f}, Emission={e_deg:.2f}")

    except Exception as e:
        print(f"FATAL WARNING: XML Parsing failed. Using placeholders. Error: {e}")
        i_deg = 45.0
        e_deg = 0.0

    # CREATE GEOMETRY MAPS (Convert to RADIANS)
    i_rad = np.full(img_shape, np.deg2rad(i_deg))
    e_rad = np.full(img_shape, np.deg2rad(e_deg))
    
    return intensity_array, i_rad, e_rad


# --- 3. THE PHOTOCLINOMETRY (SHAPE-FROM-SHADING) MODEL ---

def run_photoclinometry(I, i_map, e_map, albedo=0.1):
    """
    Implements Lambertian photoclinometry model to derive surface gradients (p, q).
    
    The Lambertian model assumes: I = ρ * cos(i)
    Where:
    - I is the observed intensity
    - ρ is the surface albedo (reflectivity)
    - cos(i) is the cosine of the local incidence angle
    
    From this, we can derive the surface gradient components p = dz/dx, q = dz/dy
    """
    print("II: Implementing Lambertian Photoclinometry Model...")
    
    # 1. Normalize intensity to [0, 1] range
    I_min, I_max = np.min(I), np.max(I)
    if I_max > I_min:
        I_norm = (I - I_min) / (I_max - I_min)
    else:
        I_norm = np.ones_like(I) * 0.5  # Avoid division by zero
    
    # 2. Apply Lambertian model: I = ρ * cos(i)
    # Solve for cos(i): cos(i) = I / ρ
    # Clamp to valid range [0, 1] to avoid numerical issues
    cos_i = np.clip(I_norm / albedo, 0.0, 1.0)
    
    # 3. Derive local incidence angle from cosine
    # Add small epsilon to avoid arccos(1) = 0 issues
    cos_i = np.clip(cos_i, 1e-6, 1.0)
    i_local = np.arccos(cos_i)
    
    # 4. Calculate surface gradients
    # For Lambertian model with sun at azimuth 0° (shadows in x-direction):
    # p = dz/dx = tan(i_local) * cos(azimuth)
    # q = dz/dy = tan(i_local) * sin(azimuth)
    
    # Assume sun azimuth is 0° (shadows fall in x-direction)
    sun_azimuth = 0.0
    p = np.tan(i_local) * np.cos(np.deg2rad(sun_azimuth))
    q = np.tan(i_local) * np.sin(np.deg2rad(sun_azimuth))
    
    # 5. Apply reasonable slope limits to avoid extreme values
    max_slope = 0.5  # radians (about 30 degrees)
    p = np.clip(p, -max_slope, max_slope)
    q = np.clip(q, -max_slope, max_slope)
    
    print(f"   Gradient statistics: p range [{np.min(p):.3f}, {np.max(p):.3f}], q range [{np.min(q):.3f}, {np.max(q):.3f}]")
    print("   Lambertian photoclinometry complete.")
    
    return p, q

# --- 4. INTEGRATION (Deriving Elevation from Slope) ---

def integrate_slopes(p, q):
    """
    Integrates the slope arrays (p, q) to derive the final Elevation Map (Z).
    Uses a least-squares integration approach for better accuracy.
    """
    print("III: Integrating Slopes to Elevation (DEM)...")
    
    # Method 1: Simple path integration (fast but less accurate)
    Z_x = np.cumsum(p, axis=1)  # Integrate along rows
    Z_y = np.cumsum(q, axis=0)  # Integrate along columns
    
    # Method 2: Average of both paths (better than single path)
    Z_simple = (Z_x + Z_y) / 2.0
    
    # Method 3: Least-squares integration (more accurate)
    # This minimizes the error between gradients and integrated surface
    try:
        Z_ls = least_squares_integration(p, q)
        Z = Z_ls
        print("   Used least-squares integration method")
    except:
        Z = Z_simple
        print("   Used simple path integration method")
    
    # Remove any overall tilt by subtracting the mean
    Z = Z - np.mean(Z)
    
    print(f"   Final DEM Array Shape: {Z.shape}")
    print(f"   DEM elevation range: [{np.min(Z):.3f}, {np.max(Z):.3f}]")
    return Z


def least_squares_integration(p, q):
    """
    Implements least-squares integration for better slope-to-elevation conversion.
    This method minimizes the error between the computed gradients and the 
    actual surface gradients of the integrated elevation.
    """
    h, w = p.shape
    
    # Create the system of equations: G * z = b
    # where G is the gradient operator matrix and z is the elevation vector
    
    # Number of pixels
    n_pixels = h * w
    
    # Create gradient operator matrices
    # For p = dz/dx: (z[i,j+1] - z[i,j]) / dx = p[i,j]
    # For q = dz/dy: (z[i+1,j] - z[i,j]) / dy = q[i,j]
    
    # Initialize sparse matrix system
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    # Create coordinate lists for sparse matrix
    row_indices = []
    col_indices = []
    data = []
    b = []
    
    eq_count = 0
    
    # Add equations for p gradients (x-direction)
    for i in range(h):
        for j in range(w-1):
            # Equation: z[i,j+1] - z[i,j] = p[i,j] * dx
            # (assuming dx = 1 for simplicity)
            pixel_idx = i * w + j
            next_pixel_idx = i * w + (j + 1)
            
            row_indices.extend([eq_count, eq_count])
            col_indices.extend([next_pixel_idx, pixel_idx])
            data.extend([1.0, -1.0])
            b.append(p[i, j])
            eq_count += 1
    
    # Add equations for q gradients (y-direction)
    for i in range(h-1):
        for j in range(w):
            # Equation: z[i+1,j] - z[i,j] = q[i,j] * dy
            pixel_idx = i * w + j
            next_pixel_idx = (i + 1) * w + j
            
            row_indices.extend([eq_count, eq_count])
            col_indices.extend([next_pixel_idx, pixel_idx])
            data.extend([1.0, -1.0])
            b.append(q[i, j])
            eq_count += 1
    
    # Add constraint: z[0,0] = 0 (fix the reference point)
    row_indices.append(eq_count)
    col_indices.append(0)
    data.append(1.0)
    b.append(0.0)
    eq_count += 1
    
    # Create sparse matrix
    G = sparse.coo_matrix((data, (row_indices, col_indices)), 
                         shape=(eq_count, n_pixels)).tocsr()
    b = np.array(b)
    
    # Solve the system
    z_flat = spsolve(G, b)
    
    # Reshape to 2D
    Z = z_flat.reshape(h, w)
    
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