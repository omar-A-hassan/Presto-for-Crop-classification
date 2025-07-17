#!/usr/bin/env python3
"""
Check Vegetation Indices in Training vs Test Data
=================================================

This script compares how vegetation indices are calculated and distributed
in training data vs test data to identify preprocessing discrepancies.
"""

import pickle
import numpy as np
import pandas as pd

def check_training_vegetation_indices():
    """Check vegetation indices in training data"""
    print("🌱 CHECKING TRAINING DATA VEGETATION INDICES")
    print("=" * 60)
    
    # Load training data
    with open('data/extracted/extracted_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    # Check samples from each class
    labels = train_data['all_labels']
    unique_classes = list(set(labels))
    
    for class_name in unique_classes:
        print(f"\n📊 {class_name.upper()} SAMPLES:")
        class_indices = [i for i, label in enumerate(labels) if label == class_name]
        
        # Check first few samples of this class
        for idx in class_indices[:2]:
            sample = train_data['all_timeseries'][idx]
            
            print(f"   Sample {idx}:")
            print(f"     Shape: {sample.shape}")
            print(f"     NDVI (band 10): mean={sample[:, 10].mean():.3f}, std={sample[:, 10].std():.3f}, range=[{sample[:, 10].min():.3f}, {sample[:, 10].max():.3f}]")
            print(f"     EVI (band 11):  mean={sample[:, 11].mean():.3f}, std={sample[:, 11].std():.3f}, range=[{sample[:, 11].min():.3f}, {sample[:, 11].max():.3f}]")
            print(f"     NDWI (band 12): mean={sample[:, 12].mean():.3f}, std={sample[:, 12].std():.3f}, range=[{sample[:, 12].min():.3f}, {sample[:, 12].max():.3f}]")
            print(f"     RNDVI (band 13): mean={sample[:, 13].mean():.3f}, std={sample[:, 13].std():.3f}, range=[{sample[:, 13].min():.3f}, {sample[:, 13].max():.3f}]")
            print(f"     Overall mean: {sample.mean():.3f}")
            print(f"     Raw bands 0-9: {sample[:, :10].mean():.3f}")

def check_test_vegetation_indices():
    """Check how vegetation indices are calculated in test data"""
    print("\n🧪 CHECKING TEST DATA VEGETATION INDICES")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv('data/raw/test.csv')
    
    # Get first location's data
    first_id = test_df['unique_id'].iloc[0]
    location_data = test_df[test_df['unique_id'] == first_id].head(10)  # First 10 timesteps
    
    print(f"📍 Test location: {first_id}")
    print(f"   Timesteps available: {len(location_data)}")
    
    # Raw band values
    band_columns = ['red', 'nir', 'swir16', 'swir22', 'blue', 'green', 'rededge1', 'rededge2', 'rededge3', 'nir08']
    raw_data = location_data[band_columns].values
    
    print(f"   Raw bands shape: {raw_data.shape}")
    print(f"   Raw bands mean: {raw_data.mean():.3f}")
    print(f"   Raw bands std: {raw_data.std():.3f}")
    
    # Calculate vegetation indices as done in inference
    red_vals = location_data['red'].values
    nir_vals = location_data['nir'].values
    green_vals = location_data['green'].values
    blue_vals = location_data['blue'].values
    nir08_vals = location_data['nir08'].values
    rededge1_vals = location_data['rededge1'].values
    
    # NDVI calculation
    ndvi = (nir_vals - red_vals) / (nir_vals + red_vals + 1e-8)
    ndvi = np.clip(ndvi, -1, 1)
    
    # EVI calculation
    evi = 2.5 * ((nir_vals - red_vals) / (nir_vals + 6*red_vals - 7.5*blue_vals + 1 + 1e-8))
    evi = np.clip(evi, -1, 1)
    
    # NDWI calculation
    ndwi = (green_vals - nir_vals) / (green_vals + nir_vals + 1e-8)
    ndwi = np.clip(ndwi, -1, 1)
    
    # RNDVI calculation
    rndvi = (nir08_vals - rededge1_vals) / (nir08_vals + rededge1_vals + 1e-8)
    rndvi = np.clip(rndvi, -1, 1)
    
    print(f"\n   Calculated vegetation indices:")
    print(f"     NDVI:  mean={ndvi.mean():.3f}, std={ndvi.std():.3f}, range=[{ndvi.min():.3f}, {ndvi.max():.3f}]")
    print(f"     EVI:   mean={evi.mean():.3f}, std={evi.std():.3f}, range=[{evi.min():.3f}, {evi.max():.3f}]")
    print(f"     NDWI:  mean={ndwi.mean():.3f}, std={ndwi.std():.3f}, range=[{ndwi.min():.3f}, {ndwi.max():.3f}]")
    print(f"     RNDVI: mean={rndvi.mean():.3f}, std={rndvi.std():.3f}, range=[{rndvi.min():.3f}, {rndvi.max():.3f}]")
    
    # Create full 14-band array as done in inference
    timesteps = len(location_data)
    ts_array = np.zeros((timesteps, 14))
    
    # Map raw bands to positions 0-9
    band_mapping = {
        'blue': 0, 'green': 1, 'red': 2, 'rededge1': 3, 'rededge2': 4,
        'rededge3': 5, 'nir': 6, 'nir08': 7, 'swir16': 8, 'swir22': 9
    }
    
    for test_col, band_idx in band_mapping.items():
        if test_col in location_data.columns:
            ts_array[:, band_idx] = location_data[test_col].values
    
    # Add vegetation indices to positions 10-13
    ts_array[:, 10] = ndvi
    ts_array[:, 11] = evi
    ts_array[:, 12] = ndwi
    ts_array[:, 13] = rndvi
    
    print(f"\n   Combined 14-band array:")
    print(f"     Shape: {ts_array.shape}")
    print(f"     Raw bands (0-9) mean: {ts_array[:, :10].mean():.3f}")
    print(f"     Vegetation indices (10-13) mean: {ts_array[:, 10:].mean():.3f}")
    print(f"     Overall mean: {ts_array.mean():.3f}")
    print(f"     Overall std: {ts_array.std():.3f}")
    
    # Apply normalization as done in inference
    print(f"\n   After normalization (as done in inference):")
    current_mean = ts_array.mean()
    current_std = ts_array.std()
    
    if current_std > 1e-8:
        # Standardize to zero mean, unit variance
        ts_normalized = (ts_array - current_mean) / current_std
        
        # Rescale to training distribution
        target_mean = 0.259
        target_std = 0.176
        ts_normalized = ts_normalized * target_std + target_mean
    else:
        ts_normalized = ts_array - current_mean + 0.259
    
    ts_normalized = np.clip(ts_normalized, 0, 1)
    
    print(f"     Normalized mean: {ts_normalized.mean():.3f}")
    print(f"     Normalized std: {ts_normalized.std():.3f}")
    print(f"     Normalized range: [{ts_normalized.min():.3f}, {ts_normalized.max():.3f}]")
    
    return ts_normalized

def compare_distributions():
    """Compare training vs test data distributions"""
    print("\n🔄 COMPARING TRAINING VS TEST DISTRIBUTIONS")
    print("=" * 60)
    
    # Training data stats
    with open('data/extracted/extracted_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    all_train_means = []
    all_train_stds = []
    all_train_ndvi = []
    all_train_evi = []
    
    for i in range(len(train_data['all_timeseries'])):
        sample = train_data['all_timeseries'][i]
        all_train_means.append(sample.mean())
        all_train_stds.append(sample.std())
        all_train_ndvi.append(sample[:, 10].mean())
        all_train_evi.append(sample[:, 11].mean())
    
    print(f"📊 TRAINING DATA STATISTICS:")
    print(f"   Overall mean: {np.mean(all_train_means):.3f} ± {np.std(all_train_means):.3f}")
    print(f"   Overall std: {np.mean(all_train_stds):.3f} ± {np.std(all_train_stds):.3f}")
    print(f"   NDVI mean: {np.mean(all_train_ndvi):.3f} ± {np.std(all_train_ndvi):.3f}")
    print(f"   EVI mean: {np.mean(all_train_evi):.3f} ± {np.std(all_train_evi):.3f}")
    
    # Test data stats (after processing)
    test_normalized = check_test_vegetation_indices()
    
    print(f"\n📊 TEST DATA STATISTICS (after processing):")
    print(f"   Overall mean: {test_normalized.mean():.3f}")
    print(f"   Overall std: {test_normalized.std():.3f}")
    print(f"   NDVI mean: {test_normalized[:, 10].mean():.3f}")
    print(f"   EVI mean: {test_normalized[:, 11].mean():.3f}")
    
    print(f"\n🎯 KEY DIFFERENCES:")
    mean_diff = abs(test_normalized.mean() - np.mean(all_train_means))
    std_diff = abs(test_normalized.std() - np.mean(all_train_stds))
    ndvi_diff = abs(test_normalized[:, 10].mean() - np.mean(all_train_ndvi))
    
    print(f"   Mean difference: {mean_diff:.3f}")
    print(f"   Std difference: {std_diff:.3f}")
    print(f"   NDVI difference: {ndvi_diff:.3f}")
    
    if mean_diff > 0.05:
        print(f"   ⚠️  Large mean difference detected!")
    if std_diff > 0.05:
        print(f"   ⚠️  Large std difference detected!")
    if ndvi_diff > 0.1:
        print(f"   ⚠️  Large NDVI difference detected!")

def main():
    """Run all checks"""
    print("🔍 VEGETATION INDICES DEBUG ANALYSIS")
    print("=" * 80)
    
    check_training_vegetation_indices()
    test_processed = check_test_vegetation_indices()
    compare_distributions()
    
    print("\n" + "=" * 80)
    print("🎯 ANALYSIS COMPLETE")
    print("Look for significant differences in vegetation indices between training and test data.")

if __name__ == "__main__":
    main()