#!/usr/bin/env python3
"""
Main entry point for NASA Lunar Pipeline

This script demonstrates how to use the modular pipeline structure
for processing lunar imagery data.
"""

import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the modular structure
from src.pipeline import LunarDLTPipeline, create_default_config


def main():
    """Main function to run the NASA Lunar Pipeline"""
    
    # Create default configuration if it doesn't exist
    if not os.path.exists('config.json'):
        logger.info("Creating default configuration file...")
        config = create_default_config()
        logger.info("Default configuration created: config.json")
    else:
        logger.info("Using existing configuration file: config.json")

    # Initialize pipeline
    try:
        logger.info("Initializing NASA Lunar Pipeline...")
        pipeline = LunarDLTPipeline('config.json')
        logger.info("Pipeline initialized successfully!")

        # Example usage - process a dataset
        # You can modify these paths as needed
        dataset_path = "./data/input"  # Directory containing lunar images
        output_path = "./data/output"  # Directory for processed images

        # Check if input directory exists
        if not os.path.exists(dataset_path):
            logger.warning(f"Input directory does not exist: {dataset_path}")
            logger.info("Creating example input directory...")
            Path(dataset_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Please place your lunar images in: {dataset_path}")
            logger.info("Then run this script again.")
            return

        # Check if there are any image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.img']:
            image_files.extend(Path(dataset_path).glob(ext))

        if not image_files:
            logger.warning(f"No image files found in {dataset_path}")
            logger.info("Please add some lunar images to the input directory and run again.")
            return

        logger.info(f"Found {len(image_files)} image files to process")
        
        # Run processing
        logger.info(f"Starting processing of lunar images...")
        logger.info(f"Input: {dataset_path}")
        logger.info(f"Output: {output_path}")
        
        results = pipeline.process_dataset(dataset_path, output_path)

        # Display results
        print("\n" + "="*50)
        print("PROCESSING RESULTS:")
        print("="*50)
        print(json.dumps(results, indent=2))

        if results["processed"] > 0:
            print(f"\n✅ Successfully processed {results['processed']} image(s)")
            print(f"📁 Output saved to: {output_path}")
            
            # Display statistics
            if results.get("camera_types"):
                print(f"\n📷 Camera Types:")
                for cam_type, count in results["camera_types"].items():
                    print(f"   - {cam_type.upper()}: {count}")
            
            if results.get("product_types"):
                print(f"\n🔧 Product Types:")
                for prod_type, count in results["product_types"].items():
                    print(f"   - {prod_type.upper()}: {count}")
            
            if results.get("processing_time"):
                print(f"\n⏱️  Total processing time: {results['processing_time']:.2f} seconds")
        else:
            print(f"\n❌ No images were successfully processed")
            print(f"💡 Check the logs above for detailed error information")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main() 