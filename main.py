# --- Contents for main.py ---

import os
from src.pipeline import run_photoclinometry_pipeline

if __name__ == "__main__":
    print("--- Starting Lunar DEM Generation Pipeline ---")
    
    # Check if the required 'final_data' folder exists before running
    # This prevents the script from starting if files aren't staged.
    if not os.path.isdir(os.path.join(os.getcwd(), 'final_data')):
        print("\nERROR: Data folder 'final_data' not found.")
        print("Please ensure you completed the manual file movement steps.")
    else:
        # Call the main function we built in src/pipeline.py
        run_photoclinometry_pipeline()