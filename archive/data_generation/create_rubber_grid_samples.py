#!/usr/bin/env python3
"""
Create Rubber Grid Samples
===========================

This script creates a grid of sampling points within the rubber study area
and labels them using the raster files. This allows integration with the
existing coordinate-based extraction pipeline.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

def create_rubber_grid_samples():
    """Create grid samples for rubber dataset integration"""
    
    print("🔍 CREATING RUBBER GRID SAMPLES")
    print("=" * 40)
    
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    rubber_dir = project_root / "data/raw/rubber"
    studyarea_path = rubber_dir / "1_StudyArea" / "StudyArea.shp"
    rubber_2018_path = rubber_dir / "2_Rubbertree_20142018" / "Rubbertree_2018.tif"
    output_path = project_root / "data/raw/rubber_grid_samples.geojson"
    
    # Load study area
    print("📍 Loading study area...")
    study_area = gpd.read_file(studyarea_path)
    bounds = study_area.total_bounds  # [minx, miny, maxx, maxy]
    
    print(f"   Study area bounds: {bounds}")
    print(f"   CRS: {study_area.crs}")
    
    # Load rubber raster for labeling
    print("🗺️ Loading rubber raster...")
    with rasterio.open(rubber_2018_path) as src:
        rubber_data = src.read(1)  # Read first band
        rubber_transform = src.transform
        rubber_crs = src.crs
        
        print(f"   Raster shape: {rubber_data.shape}")
        print(f"   Raster CRS: {rubber_crs}")
        print(f"   Unique values: {np.unique(rubber_data)}")
    
    # Create grid points
    print("📐 Creating grid points...")
    
    # Grid spacing (adjust for desired density)
    grid_spacing = 0.01  # degrees (~1km spacing)
    
    # Generate grid coordinates
    x_coords = np.arange(bounds[0], bounds[2], grid_spacing)
    y_coords = np.arange(bounds[1], bounds[3], grid_spacing)
    
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            # Check if point is within study area
            if study_area.contains(point).any():
                grid_points.append({'geometry': point, 'lon': x, 'lat': y})
    
    print(f"   Generated {len(grid_points)} grid points")
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(grid_points, crs=study_area.crs)
    
    # Convert to same CRS as raster for labeling
    if grid_gdf.crs != rubber_crs:
        grid_gdf = grid_gdf.to_crs(rubber_crs)
    
    # Sample raster values at grid points
    print("🏷️ Labeling points with raster values...")
    
    labels = []
    for idx, row in grid_gdf.iterrows():
        # Get raster value at point
        x, y = row.geometry.x, row.geometry.y
        
        # Convert to raster coordinates
        col, row_idx = ~rubber_transform * (x, y)
        col, row_idx = int(col), int(row_idx)
        
        # Check bounds
        if 0 <= row_idx < rubber_data.shape[0] and 0 <= col < rubber_data.shape[1]:
            raster_value = rubber_data[row_idx, col]
            
            # Label based on raster value (adjust based on actual values)
            if raster_value == 1:  # Assuming 1 = rubber
                labels.append('rubber')
            else:
                labels.append('non_rubber')
        else:
            labels.append('non_rubber')  # Default for out-of-bounds
    
    grid_gdf['label'] = labels
    
    # Convert back to geographic coordinates (EPSG:4326)
    grid_gdf = grid_gdf.to_crs('EPSG:4326')
    
    # Filter to only rubber points (optional - or keep both for balanced dataset)
    rubber_points = grid_gdf[grid_gdf['label'] == 'rubber']
    
    print(f"   📊 Label distribution:")
    print(f"      Rubber: {len(rubber_points)}")
    print(f"      Non-rubber: {len(grid_gdf) - len(rubber_points)}")
    
    # Save as GeoJSON (compatible with existing pipeline)
    print("💾 Saving grid samples...")
    
    # Create final dataset with rubber points only
    final_gdf = rubber_points.copy()
    final_gdf['crop_type'] = 'rubber'
    final_gdf['source'] = 'china_xishuangbanna'
    
    # Keep only necessary columns
    final_gdf = final_gdf[['geometry', 'crop_type', 'source']]
    
    # Save
    final_gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"✅ Saved {len(final_gdf)} rubber samples to: {output_path}")
    print(f"   📍 Sample coordinates:")
    for i, row in final_gdf.head(3).iterrows():
        geom = row.geometry
        print(f"      {i+1}. lat: {geom.y:.6f}, lon: {geom.x:.6f}")
    
    return output_path

if __name__ == "__main__":
    output_file = create_rubber_grid_samples()
    print(f"\n🎯 Grid samples created: {output_file}")
    print("   This file can now be added to the robust_data_extractor.py datasets list")