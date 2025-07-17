#!/usr/bin/env python3
"""
Create Ghana Cocoa Samples
===========================

This script processes the Ghana cocoa spatial metrics data to create
coordinate points for satellite data extraction. It extracts centroids
from districts with significant cocoa production.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

def create_ghana_cocoa_samples():
    """Create coordinate samples for Ghana cocoa dataset integration"""
    
    print("🍫 CREATING GHANA COCOA SAMPLES")
    print("=" * 40)
    
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data/raw/spatial-metrics-ghana-cocoa-cocoa_area_district.geojson"
    output_path = project_root / "data/raw/ghana_cocoa_samples.geojson"
    
    # Load Ghana cocoa data
    print("📍 Loading Ghana cocoa district data...")
    ghana_gdf = gpd.read_file(input_path)
    
    print(f"   Total districts: {len(ghana_gdf)}")
    print(f"   CRS: {ghana_gdf.crs}")
    print(f"   Columns: {list(ghana_gdf.columns)}")
    
    # Filter districts with cocoa production
    cocoa_districts = ghana_gdf[ghana_gdf['cocoa_area_hectares_2020'] > 0].copy()
    print(f"   Districts with cocoa: {len(cocoa_districts)}")
    
    # Apply minimum area threshold for quality samples
    min_area_hectares = 100  # Focus on districts with substantial cocoa production
    significant_cocoa = cocoa_districts[cocoa_districts['cocoa_area_hectares_2020'] >= min_area_hectares]
    print(f"   Districts with ≥{min_area_hectares} hectares: {len(significant_cocoa)}")
    
    # Create coordinate points from district centroids
    print("📐 Creating coordinate points from district centroids...")
    
    # Ensure we're in a geographic CRS for coordinates
    if significant_cocoa.crs != 'EPSG:4326':
        significant_cocoa = significant_cocoa.to_crs('EPSG:4326')
    
    # Generate coordinate samples
    coordinate_samples = []
    
    for idx, row in significant_cocoa.iterrows():
        # Get district centroid
        centroid = row.geometry.centroid
        
        # Create sample point
        sample = {
            'geometry': centroid,
            'crop_type': 'cacao',  # Use 'cacao' for consistency with existing Peru data
            'source': 'ghana_cocoa',
            'district_name': row['district_name'],
            'region_name': row['region_name'], 
            'cocoa_area_hectares': row['cocoa_area_hectares_2020'],
            'lat': centroid.y,
            'lon': centroid.x
        }
        
        coordinate_samples.append(sample)
    
    print(f"   Generated {len(coordinate_samples)} coordinate samples")
    
    # Create GeoDataFrame
    samples_gdf = gpd.GeoDataFrame(coordinate_samples, crs='EPSG:4326')
    
    # Sort by cocoa area (largest first) for prioritization
    samples_gdf = samples_gdf.sort_values('cocoa_area_hectares', ascending=False)
    
    # Keep only necessary columns for extraction pipeline
    final_gdf = samples_gdf[['geometry', 'crop_type', 'source']].copy()
    
    # Save as GeoJSON (compatible with existing pipeline)
    print("💾 Saving coordinate samples...")
    final_gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"✅ Saved {len(final_gdf)} Ghana cocoa samples to: {output_path}")
    
    # Show sample statistics
    print(f"\n📊 SAMPLE STATISTICS:")
    print(f"   Total samples: {len(final_gdf)}")
    print(f"   Crop type: cacao")
    print(f"   Source: ghana_cocoa") 
    print(f"   Geographic bounds:")
    bounds = final_gdf.total_bounds
    print(f"      Longitude: {bounds[0]:.4f} to {bounds[2]:.4f}")
    print(f"      Latitude: {bounds[1]:.4f} to {bounds[3]:.4f}")
    
    # Show top districts included
    print(f"\n🏆 TOP 10 DISTRICTS BY COCOA AREA:")
    top_districts = samples_gdf.head(10)
    for i, row in top_districts.iterrows():
        print(f"   {i+1}. {row['district_name']} ({row['region_name']}): {row['cocoa_area_hectares']:,.0f} hectares")
    
    # Show sample coordinates
    print(f"\n📍 SAMPLE COORDINATES (first 5):")
    for i, row in final_gdf.head(5).iterrows():
        geom = row.geometry
        print(f"   {i+1}. lat: {geom.y:.6f}, lon: {geom.x:.6f}")
    
    return output_path

def update_extraction_pipeline():
    """Update the robust_data_extractor.py to include Ghana cocoa samples"""
    
    print("\n🔧 UPDATING EXTRACTION PIPELINE")
    print("=" * 40)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    extractor_path = project_root / "src/data/robust_data_extractor.py"
    
    if not extractor_path.exists():
        print("⚠️  robust_data_extractor.py not found")
        return
    
    # Read the current file
    with open(extractor_path, 'r') as f:
        content = f.read()
    
    # Check if Ghana cocoa is already in the datasets list
    if 'ghana_cocoa_samples.geojson' in content:
        print("✅ Ghana cocoa already in extraction pipeline")
        return
    
    # Find the datasets list and add Ghana cocoa
    datasets_pattern = 'datasets = [\n            ("data/raw/03 cacao_ucayali_v2.json", "cacao", "peru_amazon"),\n            ("data/raw/04 Dataset_Ucayali_Palm_V2.geojson", "oil_palm", "peru_ucayali"),\n            ("data/raw/rubber_grid_samples.geojson", "rubber", "china_xishuangbanna")\n        ]'
    
    updated_datasets = 'datasets = [\n            ("data/raw/03 cacao_ucayali_v2.json", "cacao", "peru_amazon"),\n            ("data/raw/04 Dataset_Ucayali_Palm_V2.geojson", "oil_palm", "peru_ucayali"),\n            ("data/raw/rubber_grid_samples.geojson", "rubber", "china_xishuangbanna"),\n            ("data/raw/ghana_cocoa_samples.geojson", "cacao", "ghana_cocoa")\n        ]'
    
    if datasets_pattern in content:
        updated_content = content.replace(datasets_pattern, updated_datasets)
        
        # Write back the updated content
        with open(extractor_path, 'w') as f:
            f.write(updated_content)
        
        print("✅ Added Ghana cocoa to extraction pipeline")
        print("   File updated: src/data/robust_data_extractor.py")
    else:
        print("⚠️  Could not automatically update extraction pipeline")
        print("   Please manually add this line to the datasets list:")
        print('   ("data/raw/ghana_cocoa_samples.geojson", "cacao", "ghana_cocoa")')

if __name__ == "__main__":
    # Create Ghana cocoa samples
    output_file = create_ghana_cocoa_samples()
    
    # Update extraction pipeline
    update_extraction_pipeline()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Ghana cocoa samples created: {output_file}")
    print(f"   2. Run the extraction pipeline:")
    print(f"      python src/data/robust_data_extractor.py")
    print(f"   3. This will add {len(gpd.read_file(output_file))} Ghana cocoa samples to your dataset")
    print(f"   4. Geographic diversity: Peru Amazon + Ghana West Africa")