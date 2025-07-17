#!/usr/bin/env python3
"""
Standalone Coordinate Data Extraction Script
===========================================

Use this script to extract satellite data for coordinate datasets
with robust error handling and progress saving.

Usage:
    python extract_coordinate_data.py              # Resume from last progress
    python extract_coordinate_data.py --clean      # Start fresh
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append('src/data')

from robust_data_extractor import RobustDataExtractor

def main():
    print("🛰️  STANDALONE COORDINATE DATA EXTRACTION")
    print("=" * 50)
    
    # Initialize extractor
    extractor = RobustDataExtractor(output_dir="extraction_progress")
    
    # Check for clean flag
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        extractor.clean_progress()
        print("🧹 Starting fresh extraction...")
    else:
        print("📂 Resuming from existing progress (if any)...")
    
    # Extract all datasets
    timeseries, labels, coords, sources = extractor.extract_all_datasets()
    
    # Final summary
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 30)
    print(f"Total samples extracted: {len(labels)}")
    
    if labels:
        # Count by crop type
        from collections import Counter
        crop_counts = Counter(labels)
        for crop, count in crop_counts.items():
            print(f"  {crop}: {count} samples")
    
    print(f"\n✅ Extraction complete!")
    print(f"Data saved in: extraction_progress/")
    print(f"Ready for training!")

if __name__ == "__main__":
    main()