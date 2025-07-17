#!/usr/bin/env python3
"""
Test Data Analysis and Inference Script
======================================

This script examines the test.csv file structure and prepares inference
using our trained Enhanced PRESTO crop classification model.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "utils"))
sys.path.append(str(project_root / "presto"))
sys.path.append(str(project_root / "presto" / "presto"))

# Import our enhanced model
from enhanced_presto_classifier import (
    EnhancedPrestoClassifier, 
    CropDataset, 
    collate_fn,
    get_optimal_device
)
from config_loader import load_config

def examine_test_data(csv_path: str = None, verbose: bool = False):
    """Examine the structure and content of test.csv"""
    if verbose:
        print("🔍 EXAMINING TEST DATA")
        print("=" * 50)
    
    # Use config to find test data if not specified
    if csv_path is None:
        config = load_config()
        csv_path = Path(config.PATHS_CONFIG['raw_data_dir']) / 'test.csv'
    else:
        csv_path = Path(csv_path)
    
    # Load test data
    if not csv_path.exists():
        print(f"❌ Test file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"📊 Dataset Overview:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        print(f"\\n📋 Column Details:")
        print(df.dtypes)
        
        print(f"\\n🔢 Sample Data (first 5 rows):")
        print(df.head())
        
        print(f"\\n📈 Basic Statistics:")
        print(df.describe())
    else:
        print(f"Loaded test data: {df.shape[0]:,} samples, {len(df['unique_id'].unique()):,} unique locations")
    
    # Analyze unique IDs and timestamps
    if 'unique_id' in df.columns and verbose:
        print(f"\\n🆔 Unique ID Analysis:")
        print(f"   Total unique IDs: {df['unique_id'].nunique()}")
        print(f"   Total rows: {len(df)}")
        print(f"   Average timesteps per ID: {len(df) / df['unique_id'].nunique():.1f}")
        
        # Show some examples of multiple timesteps
        sample_ids = df['unique_id'].value_counts().head()
        print(f"\\n   Sample ID counts:")
        print(sample_ids)
        
        # Show data for one ID
        sample_id = sample_ids.index[0]
        sample_data = df[df['unique_id'] == sample_id]
        print(f"\\n   Sample data for ID {sample_id}:")
        print(sample_data)
    
    # Identify Sentinel-2 band columns
    potential_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    potential_indices = ['NDVI', 'EVI', 'NDWI', 'RNDVI']
    
    band_columns = [col for col in df.columns if col in potential_bands]
    index_columns = [col for col in df.columns if col in potential_indices]
    
    print(f"\\n🛰️ Sentinel-2 Data Analysis:")
    print(f"   Band columns found: {band_columns}")
    print(f"   Index columns found: {index_columns}")
    print(f"   Total feature columns: {len(band_columns) + len(index_columns)}")
    
    # Check for coordinate information
    coord_columns = [col for col in df.columns if col.lower() in ['lat', 'latitude', 'lon', 'longitude']]
    print(f"   Coordinate columns: {coord_columns}")
    
    # Check for timestamp information
    time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    print(f"   Time columns: {time_columns}")
    
    return df

def create_monthly_composites(location_data: pd.DataFrame, column_mapping: Dict, expected_bands: List, max_timesteps: int) -> np.ndarray:
    """Create monthly composites from long time series"""
    import pandas as pd
    from datetime import datetime
    
    # Convert time column to datetime
    location_data = location_data.copy()
    location_data['datetime'] = pd.to_datetime(location_data['time'])
    location_data['year_month'] = location_data['datetime'].dt.to_period('M')
    
    # Group by year-month and calculate median values
    monthly_groups = location_data.groupby('year_month')
    
    n_features = 14
    monthly_data = []
    
    for period, group in monthly_groups:
        monthly_values = np.zeros(n_features)
        
        # Fill band data
        for test_col, band_idx in column_mapping.items():
            if test_col in group.columns:
                monthly_values[band_idx] = group[test_col].median()
        
        # Calculate vegetation indices
        if 'nir' in group.columns and 'red' in group.columns:
            nir_val = group['nir'].median()
            red_val = group['red'].median()
            ndvi = (nir_val - red_val) / (nir_val + red_val + 1e-8)
            monthly_values[10] = np.clip(ndvi, -1, 1)
        
        if all(col in group.columns for col in ['nir', 'red', 'blue']):
            nir_val = group['nir'].median()
            red_val = group['red'].median()
            blue_val = group['blue'].median()
            evi = 2.5 * ((nir_val - red_val) / (nir_val + 6*red_val - 7.5*blue_val + 1 + 1e-8))
            monthly_values[11] = np.clip(evi, -1, 1)
        
        if 'green' in group.columns and 'nir' in group.columns:
            green_val = group['green'].median()
            nir_val = group['nir'].median()
            ndwi = (green_val - nir_val) / (green_val + nir_val + 1e-8)
            monthly_values[12] = np.clip(ndwi, -1, 1)
        
        if 'nir08' in group.columns and 'rededge1' in group.columns:
            nir08_val = group['nir08'].median()
            rededge1_val = group['rededge1'].median()
            rndvi = (nir08_val - rededge1_val) / (nir08_val + rededge1_val + 1e-8)
            monthly_values[13] = np.clip(rndvi, -1, 1)
        
        monthly_data.append(monthly_values)
    
    # Convert to array and limit to max_timesteps
    ts_array = np.array(monthly_data)
    if len(ts_array) > max_timesteps:
        # Take the most recent timesteps
        ts_array = ts_array[-max_timesteps:]
    
    # Clip raw bands to reasonable satellite data range (preserve vegetation indices)
    ts_array[:, :10] = np.clip(ts_array[:, :10], 0, 1)  # Only clip raw bands
    
    # Vegetation indices are already calculated and clipped to [-1, 1], don't normalize them
    
    return ts_array

def prepare_test_timeseries(df: pd.DataFrame, verbose: bool = False) -> Dict:
    """Convert test data to time series format compatible with our model"""
    if verbose:
        print("\\n🔄 PREPARING TEST TIME SERIES")
        print("=" * 50)
    else:
        print("Preparing time series data...")
    
    # Map test data columns to our model's expected band positions
    # Test data has: red, nir, swir16, swir22, blue, green, rededge1, rededge2, rededge3, nir08
    # Our model expects: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, EVI, NDWI, RNDVI]
    # Positions:         [0,  1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11,  12,   13]
    
    column_mapping = {
        # Map test columns to band positions (0-9 for raw bands)
        'blue': 0,          # B2 - Blue (490nm)
        'green': 1,         # B3 - Green (560nm) 
        'red': 2,           # B4 - Red (665nm)
        'rededge1': 3,      # B5 - Red Edge 1 (705nm)
        'rededge2': 4,      # B6 - Red Edge 2 (740nm)
        'rededge3': 5,      # B7 - Red Edge 3 (783nm)
        'nir': 6,           # B8 - NIR (842nm)
        'nir08': 7,         # B8A - Red Edge 4 (865nm)
        'swir16': 8,        # B11 - SWIR 1 (1610nm)
        'swir22': 9         # B12 - SWIR 2 (2190nm)
    }
    
    # Our model expects these bands in this order
    expected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    expected_indices = ['NDVI', 'EVI', 'NDWI', 'RNDVI']
    
    # Check what features we can map
    available_test_bands = [col for col in column_mapping.keys() if col in df.columns]
    mapped_band_indices = [column_mapping[col] for col in available_test_bands]
    
    if verbose:
        print(f"Test data columns: {list(df.columns)}")
        print(f"Available band mappings: {dict(zip(available_test_bands, mapped_band_indices))}")
        print(f"Expected features: {len(expected_bands)} bands + {len(expected_indices)} indices = 14 total")
        print(f"Available features: {len(mapped_band_indices)} bands + 4 indices = {len(mapped_band_indices) + 4} total")
    
    if len(mapped_band_indices) == 0:
        print("❌ No recognizable Sentinel-2 features found!")
        return None
    
    # Group by unique_id to create time series
    unique_ids = df['unique_id'].unique()
    timeseries_data = {}
    coords_data = {}
    
    if verbose:
        print(f"\\n📦 Processing {len(unique_ids)} unique locations...")
    
    for i, uid in enumerate(unique_ids):
        # Get all timesteps for this location
        location_data = df[df['unique_id'] == uid].copy()
        
        # Sort by timestamp if available
        if 'time' in location_data.columns:
            location_data = location_data.sort_values('time')
        
        # Extract feature values
        timesteps = len(location_data)
        n_features = 14  # Our model expects 14 features
        
        # Handle very long time series by creating monthly composites
        # Our model was trained on ~12 timesteps, so we need to compress long series
        max_timesteps = 24  # Allow up to 24 monthly composites
        
        if timesteps > max_timesteps:
            # Create monthly composites from the long time series
            ts_array = create_monthly_composites(location_data, column_mapping, expected_bands, max_timesteps)
        else:
            # Use the time series as-is for shorter sequences
            ts_array = np.zeros((timesteps, n_features))
            
            # Fill available band data using the mapping
            for test_col, band_idx in column_mapping.items():
                if test_col in location_data.columns:
                    ts_array[:, band_idx] = location_data[test_col].values
            
            # Calculate vegetation indices from available bands
            # NDVI = (NIR - Red) / (NIR + Red)
            if 'nir' in location_data.columns and 'red' in location_data.columns:
                nir_vals = location_data['nir'].values
                red_vals = location_data['red'].values
                ndvi = (nir_vals - red_vals) / (nir_vals + red_vals + 1e-8)
                ts_array[:, 10] = np.clip(ndvi, -1, 1)
            else:
                ts_array[:, 10] = 0.5  # Default NDVI
            
            # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
            if all(col in location_data.columns for col in ['nir', 'red', 'blue']):
                nir_vals = location_data['nir'].values
                red_vals = location_data['red'].values
                blue_vals = location_data['blue'].values
                evi = 2.5 * ((nir_vals - red_vals) / (nir_vals + 6*red_vals - 7.5*blue_vals + 1 + 1e-8))
                ts_array[:, 11] = np.clip(evi, -1, 1)
            else:
                ts_array[:, 11] = 0.3  # Default EVI
            
            # NDWI = (Green - NIR) / (Green + NIR)
            if 'green' in location_data.columns and 'nir' in location_data.columns:
                green_vals = location_data['green'].values
                nir_vals = location_data['nir'].values
                ndwi = (green_vals - nir_vals) / (green_vals + nir_vals + 1e-8)
                ts_array[:, 12] = np.clip(ndwi, -1, 1)
            else:
                ts_array[:, 12] = -0.1  # Default NDWI
            
            # RNDVI = (NIR08 - RedEdge1) / (NIR08 + RedEdge1)
            if 'nir08' in location_data.columns and 'rededge1' in location_data.columns:
                nir08_vals = location_data['nir08'].values
                rededge1_vals = location_data['rededge1'].values
                rndvi = (nir08_vals - rededge1_vals) / (nir08_vals + rededge1_vals + 1e-8)
                ts_array[:, 13] = np.clip(rndvi, -1, 1)
            else:
                ts_array[:, 13] = 0.4  # Default RNDVI
            
            # Clip raw bands to reasonable satellite data range (preserve vegetation indices)
            ts_array[:, :10] = np.clip(ts_array[:, :10], 0, 1)  # Only clip raw bands
            
            # Vegetation indices are already calculated and clipped to [-1, 1], don't normalize them
        
        timeseries_data[i] = ts_array
        
        # Extract coordinates (test data uses x, y which are translated coordinates)
        if 'x' in location_data.columns and 'y' in location_data.columns:
            coords_data[i] = (location_data['y'].iloc[0], location_data['x'].iloc[0])  # (lat, lon) format
        else:
            coords_data[i] = (0.0, 0.0)  # Default coordinates
        
        if verbose and i % 1000 == 0:
            print(f"   Processed {i+1}/{len(unique_ids)} locations...")
        elif not verbose and i % 2000 == 0:
            print(f"Processed {i+1:,}/{len(unique_ids):,} locations...")
    
    print(f"✅ Prepared {len(timeseries_data):,} time series")
    if verbose:
        print(f"   Time series shape example: {timeseries_data[0].shape}")
        print(f"   Average timesteps per location: {sum(ts.shape[0] for ts in timeseries_data.values()) / len(timeseries_data):.1f}")
    
    # Create ID mapping for submission
    id_mapping = {i: uid for i, uid in enumerate(unique_ids)}
    
    return {
        'timeseries_data': timeseries_data,
        'coords_data': coords_data,
        'id_mapping': id_mapping,
        'feature_info': {
            'available_bands': mapped_band_indices,
            'calculated_indices': ['NDVI', 'EVI', 'NDWI', 'RNDVI'],
            'total_features': len(mapped_band_indices) + 4
        }
    }

def load_trained_model(model_path: str = None):
    """Load our trained Enhanced PRESTO model"""
    print("\\n🤖 LOADING TRAINED MODEL")
    print("=" * 50)
    
    # Use config to find model if not specified
    if model_path is None:
        config = load_config()
        model_path = Path(config.PATHS_CONFIG['model_final_dir']) / 'enhanced_presto_crop_classifier.pth'
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    # Load model checkpoint with weights_only=False for sklearn objects
    # This is safe since we trust our own saved model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    label_encoder = checkpoint['label_encoder']
    
    print(f"📋 Model Configuration:")
    print(f"   Number of classes: {model_config['num_classes']}")
    print(f"   Class names: {list(label_encoder.classes_)}")
    
    # Create model WITHOUT loading pretrained weights
    # We'll load our trained weights which include the fine-tuned PRESTO
    model = EnhancedPrestoClassifier(
        num_classes=model_config['num_classes'],
        freeze_backbone=False,  # For inference
        unfreeze_layers=model_config['unfreeze_layers'],
        load_pretrained=False  # Don't load pretrained weights, we'll load our trained weights
    )
    
    # Load trained weights (this should include fine-tuned PRESTO backbone)
    print("🔄 Loading trained model weights (including fine-tuned PRESTO backbone)...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Trained model weights loaded successfully")
    
    print(f"✅ Model loaded successfully")
    
    return model, label_encoder

def run_inference(model, test_data: Dict, label_encoder, verbose: bool = False) -> pd.DataFrame:
    """Run inference on test data"""
    if verbose:
        print("\\n🔮 RUNNING INFERENCE")
        print("=" * 50)
    else:
        print("Running inference...")
    
    device = get_optimal_device()
    model = model.to(device)
    
    print(f"Device: {device}")
    
    # Create test dataset
    timeseries_data = test_data['timeseries_data']
    coords_data = test_data['coords_data']
    id_mapping = test_data['id_mapping']
    
    # Create dummy labels for dataset compatibility
    dummy_labels = {i: 0 for i in range(len(timeseries_data))}
    
    test_dataset = CropDataset(timeseries_data, dummy_labels, coords_data)
    
    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Run inference
    all_predictions = []
    all_probabilities = []
    all_indices = []
    
    print(f"Processing {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move to device
            x = batch['x'].to(device)
            dynamic_world = batch['dynamic_world'].to(device)
            latlons = batch['latlons'].to(device)
            month = batch['month'].to(device)
            
            # Forward pass
            logits = model(x, dynamic_world, latlons, month=month)
            
            # Get predictions and probabilities
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Track indices for this batch
            batch_start = batch_idx * test_loader.batch_size
            batch_end = min(batch_start + test_loader.batch_size, len(test_dataset))
            all_indices.extend(range(batch_start, batch_end))
            
            if batch_idx % 10 == 0:
                print(f"   Processed batch {batch_idx+1}/{len(test_loader)}")
    
    # Create submission dataframe
    submissions = []
    
    for i, (pred, probs) in enumerate(zip(all_predictions, all_probabilities)):
        # Get original unique_id
        unique_id = id_mapping[all_indices[i]]
        
        # Get predicted class name
        predicted_class = label_encoder.classes_[pred]
        
        # Get confidence scores
        class_probs = {label_encoder.classes_[j]: probs[j] for j in range(len(label_encoder.classes_))}
        
        submission = {
            'unique_id': unique_id,
            'predicted_class': predicted_class,
            'confidence': probs[pred],
            **{f'prob_{cls}': class_probs[cls] for cls in label_encoder.classes_}
        }
        
        submissions.append(submission)
    
    submission_df = pd.DataFrame(submissions)
    
    print(f"✅ Inference completed!")
    print(f"   Total predictions: {len(submission_df)}")
    
    # Show prediction distribution
    pred_counts = submission_df['predicted_class'].value_counts()
    print(f"\\n📊 Prediction Distribution:")
    for class_name, count in pred_counts.items():
        percentage = (count / len(submission_df)) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    return submission_df

def create_competition_submission(submission_df: pd.DataFrame, output_file: str = None):
    """Create final submission file in competition format"""
    print("\\n📝 CREATING COMPETITION SUBMISSION")
    print("=" * 50)
    
    # Use config to determine output path if not specified
    if output_file is None:
        config = load_config()
        submissions_dir = Path(config.PATHS_CONFIG['submissions_dir'])
        submissions_dir.mkdir(parents=True, exist_ok=True)
        output_file = submissions_dir / 'submission.csv'
    else:
        output_file = Path(output_file)
    
    # Competition expects probability format: unique_id, crop_type_cocoa, crop_type_oil, crop_type_rubber
    # Map our class names to the expected column names
    class_to_column = {
        'cacao': 'crop_type_cocoa',
        'oil_palm': 'crop_type_oil', 
        'rubber': 'crop_type_rubber'
    }
    
    # Create final submission in the expected format
    final_submission = pd.DataFrame()
    final_submission['unique_id'] = submission_df['unique_id']
    
    # Initialize all probability columns to 0
    final_submission['crop_type_cocoa'] = 0.0
    final_submission['crop_type_oil'] = 0.0
    final_submission['crop_type_rubber'] = 0.0
    
    # Fill in the actual probabilities from our model
    for idx, row in submission_df.iterrows():
        final_submission.loc[idx, 'crop_type_cocoa'] = row['prob_cacao']
        final_submission.loc[idx, 'crop_type_oil'] = row['prob_oil_palm']
        final_submission.loc[idx, 'crop_type_rubber'] = row['prob_rubber']
    
    # Save submission
    final_submission.to_csv(output_file, index=False)
    
    print(f"✅ Submission saved to: {output_file}")
    print(f"   Format: {list(final_submission.columns)}")
    print(f"   Rows: {len(final_submission)}")
    
    # Also save detailed results with class names for reference
    detailed_file = str(output_file).replace('.csv', '_detailed.csv')
    submission_df.to_csv(detailed_file, index=False)
    print(f"   Detailed results saved to: {detailed_file}")
    
    # Show sample probabilities
    print(f"\\n📊 Sample Probabilities:")
    print(final_submission.head())
    
    return final_submission

def main(verbose: bool = False):
    """Main execution pipeline"""
    if verbose:
        print("🚀 ENHANCED PRESTO INFERENCE PIPELINE")
        print("=" * 80)
    else:
        print("🚀 Running Enhanced PRESTO Inference...")
        print("-" * 50)
    
    try:
        # 1. Examine test data
        test_df = examine_test_data("data/raw/test.csv", verbose=verbose)
        if test_df is None:
            return
        
        # 2. Prepare time series data
        test_data = prepare_test_timeseries(test_df, verbose=verbose)
        if test_data is None:
            return
        
        # 3. Load trained model
        model, label_encoder = load_trained_model("models/final/enhanced_presto_crop_classifier.pth")
        if model is None:
            return
        
        # 4. Run inference
        submission_df = run_inference(model, test_data, label_encoder, verbose=verbose)
        
        # 5. Create competition submission
        final_submission = create_competition_submission(submission_df)
        
        print("\\n🎉 INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Files created:")
        print("   - submission.csv (competition format)")
        print("   - submission_detailed.csv (with probabilities)")
        
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced PRESTO Inference Pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output for debugging")
    args = parser.parse_args()
    
    exit_code = main(verbose=args.verbose)
    sys.exit(exit_code)