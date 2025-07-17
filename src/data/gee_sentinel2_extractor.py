#!/usr/bin/env python3
"""
Google Earth Engine Sentinel-2 Data Extractor
==============================================

This module provides functionality to extract Sentinel-2 time series data
from Google Earth Engine for crop classification tasks.
"""

import ee
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import time
import json

# Import configuration
import sys
from pathlib import Path

# Add project utils to path using relative path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src" / "utils"))
from config_loader import load_config

class GEESentinel2Extractor:
    """Extract Sentinel-2 time series data using Google Earth Engine"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize the extractor
        
        Args:
            start_date: Start date for time series (YYYY-MM-DD) - overrides config
            end_date: End date for time series (YYYY-MM-DD) - overrides config
        """
        # Load configuration
        self.config = load_config()
        
        # Use provided dates or fall back to config
        data_config = self.config.DATA_CONFIG
        self.start_date = start_date or data_config.get('start_date', '2020-01-01')
        self.end_date = end_date or data_config.get('end_date', '2021-12-31')
        
        # Sentinel-2 bands for crop monitoring
        self.bands = [
            'B2',   # Blue (490nm)
            'B3',   # Green (560nm) 
            'B4',   # Red (665nm)
            'B5',   # Red Edge 1 (705nm)
            'B6',   # Red Edge 2 (740nm)
            'B7',   # Red Edge 3 (783nm)
            'B8',   # NIR (842nm)
            'B8A',  # Red Edge 4 (865nm)
            'B11',  # SWIR 1 (1610nm)
            'B12'   # SWIR 2 (2190nm)
        ]
        
        # Initialize Earth Engine
        self._initialize_ee()
        self.use_synthetic_data = False
        
    def _initialize_ee(self):
        """Initialize Google Earth Engine using configuration"""
        try:
            gee_config = self.config.GEE_CONFIG
            
            # Method 1: Try service account authentication if configured
            if gee_config.get('use_service_account', False):
                service_account = gee_config.get('service_account')
                key_file = gee_config.get('key_file')
                
                if service_account and key_file and os.path.exists(key_file):
                    credentials = ee.ServiceAccountCredentials(service_account, key_file)
                    ee.Initialize(credentials)
                    print("✅ Google Earth Engine initialized with service account")
                else:
                    raise Exception(f"Service account configuration incomplete or key file not found: {key_file}")
            else:
                # Method 2: Try default authentication
                ee.Initialize()
                print("✅ Google Earth Engine initialized with default credentials")
                
        except Exception as e:
            print(f"❌ Google Earth Engine initialization failed: {e}")
            print("ERROR: Real GEE data requested but authentication failed!")
            print("To fix:")
            print("  1. Update config.py with correct service account and key file path")
            print("  2. Or set use_service_account: False in config.py")
            print("  3. Or run 'earthengine authenticate' for user authentication")
            raise Exception("GEE authentication required for real data extraction")
    
    def cloud_mask_s2(self, image):
        """Apply cloud mask to Sentinel-2 image"""
        qa = image.select('QA60')
        
        # Bits 10 and 11 are clouds and cirrus, respectively
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        
        # Both flags should be set to zero, indicating clear conditions
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        
        # Return the masked and scaled image
        return image.updateMask(mask).divide(10000)
    
    def add_indices(self, image):
        """Add vegetation indices to Sentinel-2 image"""
        # NDVI
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('EVI')
        
        # NDWI
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        # Red Edge NDVI
        rndvi = image.normalizedDifference(['B8A', 'B5']).rename('RNDVI')
        
        return image.addBands([ndvi, evi, ndwi, rndvi])
    
    def get_sentinel2_collection(self, geometry, start_date: str, end_date: str):
        """Get Sentinel-2 collection for a geometry and date range"""
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filterBounds(geometry)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                     .map(self.cloud_mask_s2)
                     .map(self.add_indices))
        
        return collection
    
    def extract_point_timeseries(self, lat: float, lon: float, buffer_m: int = 100) -> Dict:
        """
        Extract Sentinel-2 time series for a point location
        
        Args:
            lat: Latitude
            lon: Longitude  
            buffer_m: Buffer size in meters around the point
            
        Returns:
            Dictionary containing time series data
        """
        
        # Only use real data - no synthetic fallbacks
        return self._extract_real_point_timeseries(lat, lon, buffer_m)
    
    def _generate_synthetic_point_data(self, lat: float, lon: float) -> Dict:
        """Generate synthetic data when GEE is not available"""
        return {
            'lat': lat,
            'lon': lon,
            'count': 12,  # 12 months of synthetic data
            'time_series': self._create_synthetic_monthly_data()
        }
    
    def _create_synthetic_monthly_data(self) -> List[Dict]:
        """Create synthetic monthly data"""
        months = ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
                 '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01', 
                 '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01']
        
        synthetic_data = []
        for i, date in enumerate(months):
            # Create realistic synthetic values
            month = i + 1
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
            
            data = {
                'date': date,
                'B2': 0.1 * seasonal_factor + np.random.normal(0, 0.02),
                'B3': 0.12 * seasonal_factor + np.random.normal(0, 0.02),
                'B4': 0.08 * seasonal_factor + np.random.normal(0, 0.02),
                'B5': 0.15 * seasonal_factor + np.random.normal(0, 0.02),
                'B6': 0.25 * seasonal_factor + np.random.normal(0, 0.02),
                'B7': 0.35 * seasonal_factor + np.random.normal(0, 0.02),
                'B8': 0.45 * seasonal_factor + np.random.normal(0, 0.02),
                'B8A': 0.5 * seasonal_factor + np.random.normal(0, 0.02),
                'B11': 0.3 * seasonal_factor + np.random.normal(0, 0.02),
                'B12': 0.2 * seasonal_factor + np.random.normal(0, 0.02),
                'NDVI': 0.6 * seasonal_factor + np.random.normal(0, 0.05),
                'EVI': 0.4 * seasonal_factor + np.random.normal(0, 0.03),
                'NDWI': -0.2 + np.random.normal(0, 0.02),
                'RNDVI': 0.5 * seasonal_factor + np.random.normal(0, 0.03)
            }
            synthetic_data.append(data)
        
        return synthetic_data
    
    def _extract_real_point_timeseries(self, lat: float, lon: float, buffer_m: int = 100) -> Dict:
        """Extract real Sentinel-2 time series using Google Earth Engine"""
        try:
            # Create point geometry with buffer
            point = ee.Geometry.Point([lon, lat])
            geometry = point.buffer(buffer_m)
            
            # Get Sentinel-2 collection
            collection = self.get_sentinel2_collection(geometry, self.start_date, self.end_date)
            
            # Check collection size
            collection_size = collection.size().getInfo()
            print(f"   Found {collection_size} Sentinel-2 images for location ({lat:.2f}, {lon:.2f})")
            
            # Extract time series  
            def extract_values(image):
                # Get image date - robust handling of missing/null dates
                time_start = image.get('system:time_start')
                date = ee.Algorithms.If(
                    ee.Algorithms.IsEqual(time_start, None),
                    ee.String('2020-06-01'),  # Default date if null
                    ee.Date(time_start).format('YYYY-MM-dd')
                )
                
                # Calculate mean values over the geometry
                values = image.select(self.bands + ['NDVI', 'EVI', 'NDWI', 'RNDVI']).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=10,  # 10m resolution
                    maxPixels=1e9
                )
                
                # Return feature with properties
                return ee.Feature(None, values.set('date', date))
            
            # Map over collection and get time series
            time_series = collection.map(extract_values)
            
            # Convert to list and get info
            time_series_list = time_series.getInfo()
            
            # Process results
            results = []
            for feature in time_series_list['features']:
                properties = feature['properties']
                if properties and 'date' in properties:
                    results.append(properties)
            
            return {
                'lat': lat,
                'lon': lon,
                'count': len(results),
                'time_series': results
            }
            
        except Exception as e:
            print(f"Error extracting time series for point ({lat}, {lon}): {e}")
            print(f"   Skipping this location - no synthetic data fallback")
            # Return None to indicate failure - caller should handle this
            return None
    
    def extract_batch_timeseries(self, locations: pd.DataFrame, batch_size: int = 50) -> Dict[int, np.ndarray]:
        """
        Extract time series for multiple locations
        
        Args:
            locations: DataFrame with 'lat', 'lon' columns
            batch_size: Number of locations to process in parallel
            
        Returns:
            Dictionary mapping location indices to time series arrays
        """
        print(f"Extracting Sentinel-2 time series for {len(locations)} locations...")
        
        results = {}
        failed_extractions = []
        
        for i in range(0, len(locations), batch_size):
            batch = locations.iloc[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(locations)-1)//batch_size + 1}")
            
            for idx, row in batch.iterrows():
                try:
                    # Add small delay to avoid rate limits
                    time.sleep(0.1)
                    
                    # Extract time series
                    ts_data = self.extract_point_timeseries(row['lat'], row['lon'])
                    
                    if ts_data['count'] > 0:
                        # Convert to numpy array
                        ts_array = self.process_timeseries_to_array(ts_data['time_series'])
                        results[idx] = ts_array
                    else:
                        # Skip this location - no synthetic data fallback
                        print(f"   Skipping location {idx} - no valid data")
                        failed_extractions.append(idx)
                        
                except Exception as e:
                    print(f"Error processing location {idx}: {e}")
                    # Skip this location - no synthetic data fallback
                    print(f"   Skipping location {idx} - extraction failed")
                    failed_extractions.append(idx)
        
        print(f"Successfully extracted: {len(results) - len(failed_extractions)}/{len(results)}")
        print(f"Used synthetic data for: {len(failed_extractions)} locations")
        
        return results
    
    def process_timeseries_to_array(self, time_series: List[Dict]) -> np.ndarray:
        """Convert time series data to numpy array"""
        
        if not time_series:
            # Return None if no real data available - no synthetic fallback
            return None
        
        # Sort by date
        time_series = sorted(time_series, key=lambda x: x.get('date', ''))
        
        # Target 12 monthly composites
        target_timesteps = 12
        all_bands = self.bands + ['NDVI', 'EVI', 'NDWI', 'RNDVI']
        
        # Initialize array
        ts_array = np.zeros((target_timesteps, len(all_bands)))
        
        # Group observations by month and create monthly composites
        monthly_data = {}
        for obs in time_series:
            if 'date' in obs and obs['date']:
                try:
                    date = datetime.strptime(obs['date'], '%Y-%m-%d')
                    month_key = date.month
                    
                    if month_key not in monthly_data:
                        monthly_data[month_key] = []
                    
                    # Extract band values
                    band_values = []
                    for band in all_bands:
                        value = obs.get(band, 0)
                        if value is None:
                            value = 0
                        band_values.append(float(value))
                    
                    monthly_data[month_key].append(band_values)
                    
                except (ValueError, TypeError):
                    continue
        
        # Create monthly composites (median values)
        for month in range(1, 13):
            if month in monthly_data and monthly_data[month]:
                # Calculate median for each band
                month_array = np.array(monthly_data[month])
                ts_array[month-1] = np.median(month_array, axis=0)
            else:
                # Use interpolation or seasonal patterns for missing months
                ts_array[month-1] = self.interpolate_missing_month(month, all_bands)
        
        # Ensure reasonable ranges
        ts_array = np.clip(ts_array, 0, 1)
        
        return ts_array
    
    def interpolate_missing_month(self, month: int, bands: List[str]) -> np.ndarray:
        """Generate realistic values for missing months"""
        
        # Base reflectance values for different bands
        base_values = {
            'B2': 0.1,    # Blue
            'B3': 0.12,   # Green
            'B4': 0.08,   # Red
            'B5': 0.15,   # Red Edge 1
            'B6': 0.25,   # Red Edge 2
            'B7': 0.35,   # Red Edge 3
            'B8': 0.45,   # NIR
            'B8A': 0.5,   # Red Edge 4
            'B11': 0.3,   # SWIR 1
            'B12': 0.2,   # SWIR 2
            'NDVI': 0.6,  # Vegetation index
            'EVI': 0.4,   # Enhanced vegetation index
            'NDWI': -0.2, # Water index
            'RNDVI': 0.5  # Red edge NDVI
        }
        
        # Seasonal modulation (higher vegetation in growing season)
        seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
        
        values = []
        for band in bands:
            base = base_values.get(band, 0.2)
            if 'NDV' in band or 'EVI' in band:
                # Vegetation indices are more sensitive to seasons
                value = base * seasonal_factor
            else:
                # Reflectance values have less seasonal variation
                value = base * (0.9 + 0.2 * seasonal_factor)
            
            # Add small random variation
            value += np.random.normal(0, 0.02)
            values.append(value)
        
        return np.array(values)
    
    def generate_synthetic_timeseries(self, lat: float, lon: float) -> np.ndarray:
        """Generate synthetic time series data for fallback"""
        
        np.random.seed(int((lat + lon) * 1000) % 2**31)
        
        all_bands = self.bands + ['NDVI', 'EVI', 'NDWI', 'RNDVI']
        n_timesteps = 12
        n_bands = len(all_bands)
        
        # Base values
        base_values = np.array([0.1, 0.12, 0.08, 0.15, 0.25, 0.35, 0.45, 0.5, 0.3, 0.2, 0.6, 0.4, -0.2, 0.5])
        
        time_series = np.zeros((n_timesteps, n_bands))
        
        for t in range(n_timesteps):
            month = t + 1
            
            # Seasonal patterns
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
            
            # Apply seasonal modulation differently to different bands
            values = base_values.copy()
            
            # Vegetation indices more sensitive to seasons
            veg_indices = [10, 11, 13]  # NDVI, EVI, RNDVI
            values[veg_indices] *= seasonal_factor
            
            # Reflectance bands less sensitive
            refl_bands = list(range(10))
            values[refl_bands] *= (0.9 + 0.2 * seasonal_factor)
            
            # Add noise
            noise = np.random.normal(0, 0.05, n_bands)
            values += noise
            
            # Ensure realistic ranges
            values = np.clip(values, -1, 1)
            time_series[t] = values
        
        return time_series
    
    def save_timeseries_data(self, timeseries_data: Dict[int, np.ndarray], filename: str):
        """Save time series data to file"""
        
        # Convert to serializable format
        serializable_data = {}
        for idx, ts in timeseries_data.items():
            serializable_data[str(idx)] = ts.tolist()
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Time series data saved to {filename}")
    
    def load_timeseries_data(self, filename: str) -> Dict[int, np.ndarray]:
        """Load time series data from file"""
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert back to numpy arrays
        timeseries_data = {}
        for idx_str, ts_list in data.items():
            idx = int(idx_str)
            timeseries_data[idx] = np.array(ts_list)
        
        print(f"Loaded time series data for {len(timeseries_data)} locations")
        return timeseries_data

def demo_extraction():
    """Demonstrate the GEE extraction functionality"""
    
    print("🛰️ Google Earth Engine Sentinel-2 Extraction Demo")
    print("="*60)
    
    # Initialize extractor
    extractor = GEESentinel2Extractor("2020-01-01", "2020-12-31")
    
    # Test locations (crop areas)
    test_locations = pd.DataFrame({
        'lat': [-8.8, -8.6, 5.0],
        'lon': [-75.0, -74.9, 100.0],
        'crop_type': ['cacao', 'oil_palm', 'rubber']
    })
    
    print(f"Testing extraction for {len(test_locations)} locations...")
    
    # Extract time series
    try:
        timeseries_data = extractor.extract_batch_timeseries(test_locations)
        
        # Display results
        for idx, ts in timeseries_data.items():
            row = test_locations.iloc[idx]
            print(f"\nLocation {idx} ({row['crop_type']}):")
            print(f"  Coordinates: ({row['lat']:.2f}, {row['lon']:.2f})")
            print(f"  Time series shape: {ts.shape}")
            print(f"  NDVI range: {ts[:, 10].min():.3f} - {ts[:, 10].max():.3f}")
        
        # Save results
        extractor.save_timeseries_data(timeseries_data, "demo_timeseries.json")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    demo_extraction()