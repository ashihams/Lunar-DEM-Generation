#!/usr/bin/env python3
"""
Enhanced Lunar DEM Generation Pipeline Test Script

This script demonstrates the complete enhanced pipeline with:
1. Advanced Hapke/Lommel-Seeliger models
2. Shadow handling for deep shadow areas
3. DEM quality validation and assessment
4. Multiple model comparison

Usage:
    python test_enhanced_pipeline.py
"""

import os
import sys
import numpy as np
from src.pipeline import (
    run_photoclinometry_pipeline,
    extract_photoclinometry_inputs,
    run_photoclinometry,
    handle_shadows,
    integrate_slopes,
    validate_dem_quality,
    export_dem
)

def test_all_models():
    """Test all available photoclinometry models with enhanced features."""
    
    print("🚀 ENHANCED LUNAR DEM GENERATION PIPELINE TEST")
    print("=" * 60)
    
    # Configuration
    DATA_DIR_BASE = os.path.join(os.getcwd(), 'final_data')
    FULL_DATA_FOLDER = DATA_DIR_BASE
    TIF_NAME = "thumb_wac.M1410501593CC_M1410501593CC_pyr.tif"
    XML_NAME = "thumb_wac.M1410501593CC_M1410501593CC_pyr.xml"
    TIF_PATH = os.path.join(FULL_DATA_FOLDER, TIF_NAME)
    XML_PATH = os.path.join(FULL_DATA_FOLDER, XML_NAME)
    OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📁 Input Data: {TIF_PATH}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print()
    
    # Test models
    models = [
        ("lambertian", "Basic Lambertian Model"),
        ("lommel_seeliger", "Lommel-Seeliger Model (Recommended for Lunar)"),
        ("hapke", "Advanced Hapke Model")
    ]
    
    results = {}
    
    for model_type, model_name in models:
        print(f"\n🔬 TESTING {model_name.upper()}")
        print("-" * 40)
        
        try:
            # Extract input data
            print("1. Extracting input data...")
            intensity, incidence, emission = extract_photoclinometry_inputs(TIF_PATH, XML_PATH)
            
            if intensity is None:
                print(f"❌ Failed to extract data for {model_type}")
                continue
            
            # Run photoclinometry
            print(f"2. Running {model_name}...")
            p, q = run_photoclinometry(intensity, incidence, emission, model_type=model_type)
            
            # Apply shadow handling
            print("3. Applying shadow handling...")
            p, q = handle_shadows(intensity, p, q)
            
            # Integrate slopes
            print("4. Integrating slopes...")
            dem = integrate_slopes(p, q)
            
            # Validate quality
            print("5. Validating DEM quality...")
            quality_score = validate_dem_quality(dem, model_type=model_type)
            
            # Export DEM
            output_filename = os.path.join(OUTPUT_DIR, f"DEM_{model_type}_enhanced.tif")
            print(f"6. Exporting DEM to {output_filename}...")
            export_dem(dem, TIF_PATH, output_filename)
            
            # Store results
            results[model_type] = {
                'quality_score': quality_score,
                'dem_range': np.max(dem) - np.min(dem),
                'dem_std': np.std(dem),
                'output_file': output_filename
            }
            
            print(f"✅ {model_name} completed successfully!")
            print(f"   Quality Score: {quality_score}/100")
            print(f"   DEM Range: {results[model_type]['dem_range']:.3f} meters")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary report
    print("\n📊 ENHANCED PIPELINE SUMMARY REPORT")
    print("=" * 60)
    
    for model_type, model_name in models:
        if model_type in results:
            r = results[model_type]
            print(f"\n{model_name}:")
            print(f"  Quality Score: {r['quality_score']}/100")
            print(f"  DEM Range: {r['dem_range']:.3f} meters")
            print(f"  Standard Deviation: {r['dem_std']:.3f} meters")
            print(f"  Output File: {os.path.basename(r['output_file'])}")
            
            # Quality assessment
            if r['quality_score'] >= 90:
                print(f"  Status: 🏆 EXCELLENT - Ready for scientific analysis")
            elif r['quality_score'] >= 75:
                print(f"  Status: ✅ GOOD - Minor quality issues")
            elif r['quality_score'] >= 60:
                print(f"  Status: ⚠️  FAIR - Consider reprocessing")
            else:
                print(f"  Status: ❌ POOR - Requires improvement")
        else:
            print(f"\n{model_name}: ❌ FAILED")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Best Model: {max(results.keys(), key=lambda k: results[k]['quality_score']) if results else 'None'}")
    print(f"   • For scientific analysis, use the model with highest quality score")
    print(f"   • View results in QGIS for visual validation")
    print(f"   • Compare against known lunar feature measurements")
    
    print(f"\n🎯 ENHANCED PIPELINE TEST COMPLETE!")
    print(f"   Generated {len(results)} high-quality DEMs")
    print(f"   All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_all_models()
