#!/usr/bin/env python3
"""
Final Diagnosis Script
=====================

Test the normalization and model response directly
"""

import pickle
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "utils"))

from enhanced_presto_classifier import EnhancedPrestoClassifier
from config_loader import load_config

def test_normalization():
    """Test if normalization is working"""
    print("🔍 TESTING NORMALIZATION")
    print("=" * 50)
    
    # Load test data sample
    test_df = pd.read_csv('data/raw/test.csv')
    test_bands = ['red', 'nir', 'swir16', 'swir22', 'blue', 'green', 'rededge1', 'rededge2', 'rededge3', 'nir08']
    
    # Get raw sample
    raw_sample = test_df[test_bands].iloc[0].values
    print(f"Raw test sample:")
    print(f"   Mean: {raw_sample.mean():.3f}")
    print(f"   Std: {raw_sample.std():.3f}")
    print(f"   Range: [{raw_sample.min():.3f}, {raw_sample.max():.3f}]")
    
    # Apply normalization (same as in inference script)
    ts_array = np.clip(raw_sample, 0, 1)
    current_mean = ts_array.mean()
    current_std = ts_array.std()
    
    if current_std > 1e-8:
        # Standardize to zero mean, unit variance
        ts_array = (ts_array - current_mean) / current_std
        
        # Rescale to training distribution
        target_mean = 0.259
        target_std = 0.176
        ts_array = ts_array * target_std + target_mean
    else:
        ts_array = ts_array - current_mean + 0.259
    
    ts_array = np.clip(ts_array, 0, 1)
    
    print(f"\nNormalized test sample:")
    print(f"   Mean: {ts_array.mean():.3f}")
    print(f"   Std: {ts_array.std():.3f}")
    print(f"   Range: [{ts_array.min():.3f}, {ts_array.max():.3f}]")
    
    # Load training data for comparison
    with open('data/extracted/extracted_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    train_sample = train_data['all_timeseries'][0]
    print(f"\nTraining sample:")
    print(f"   Mean: {train_sample.mean():.3f}")
    print(f"   Std: {train_sample.std():.3f}")
    print(f"   Range: [{train_sample.min():.3f}, {train_sample.max():.3f}]")
    
    return ts_array, train_sample

def test_model_response():
    """Test model response to different inputs"""
    print("\n🧪 TESTING MODEL RESPONSE")
    print("=" * 50)
    
    # Load model
    model_path = 'results/enhanced_presto_crop_classifier.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = EnhancedPrestoClassifier(
        num_classes=3,
        freeze_backbone=False,
        unfreeze_layers=2,
        load_pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    label_encoder = checkpoint['label_encoder']
    device = torch.device('cpu')
    model = model.to(device)
    
    # Test on actual training samples from each class
    with open('data/extracted/extracted_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    labels = train_data['all_labels']
    
    # Test each class
    for class_name in ['cacao', 'oil_palm', 'rubber']:
        class_indices = [i for i, label in enumerate(labels) if label == class_name]
        if class_indices:
            sample_idx = class_indices[0]
            sample_ts = train_data['all_timeseries'][sample_idx]
            
            # Convert to tensor
            x = torch.from_numpy(sample_ts).float().unsqueeze(0).to(device)
            timesteps = x.shape[1]
            dynamic_world = torch.full((1, timesteps), 9, dtype=torch.long).to(device)
            latlons = torch.zeros(1, 2, dtype=torch.float).to(device)
            month = torch.tensor([6], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(x, dynamic_world, latlons, month=month)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(logits, dim=1)
            
            predicted_class = label_encoder.classes_[pred_idx.item()]
            
            print(f"{class_name} training sample:")
            print(f"   Predicted: {predicted_class}")
            print(f"   Probabilities: {probs.numpy()[0]}")
            print(f"   Correct: {predicted_class == class_name}")

def test_synthetic_data():
    """Test model on synthetic data with different distributions"""
    print("\n🎭 TESTING SYNTHETIC DATA")
    print("=" * 50)
    
    # Load model
    model_path = 'results/enhanced_presto_crop_classifier.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = EnhancedPrestoClassifier(
        num_classes=3,
        freeze_backbone=False,
        unfreeze_layers=2,
        load_pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    label_encoder = checkpoint['label_encoder']
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create synthetic data with different means
    test_means = [0.1, 0.259, 0.4, 0.6, 0.8]
    
    for mean_val in test_means:
        # Create synthetic time series
        synthetic_ts = np.random.normal(mean_val, 0.176, (12, 14))
        synthetic_ts = np.clip(synthetic_ts, 0, 1)
        
        # Test prediction
        x = torch.from_numpy(synthetic_ts).float().unsqueeze(0).to(device)
        timesteps = x.shape[1]
        dynamic_world = torch.full((1, timesteps), 9, dtype=torch.long).to(device)
        latlons = torch.zeros(1, 2, dtype=torch.float).to(device)
        month = torch.tensor([6], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(x, dynamic_world, latlons, month=month)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1)
        
        predicted_class = label_encoder.classes_[pred_idx.item()]
        
        print(f"Synthetic data (mean={mean_val:.3f}):")
        print(f"   Predicted: {predicted_class}")
        print(f"   Probabilities: {probs.numpy()[0]}")
        print(f"   Max prob: {probs.max().item():.3f}")

def main():
    """Run all tests"""
    print("🔬 FINAL DIAGNOSIS")
    print("=" * 80)
    
    # Test normalization
    norm_test, train_sample = test_normalization()
    
    # Test model response
    test_model_response()
    
    # Test synthetic data
    test_synthetic_data()
    
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    main()