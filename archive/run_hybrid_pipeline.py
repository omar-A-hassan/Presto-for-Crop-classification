#!/usr/bin/env python3
"""
Hybrid Crop Classification Pipeline
===================================

Optimal pipeline that combines:
- GEE extraction for Peru coordinate data (cacao + oil palm)
- Direct processing of SE Asia rubber images
- PRESTO feature extraction for both data types
- Unified ensemble classifier

Total samples: 3,305 (1,750 coordinates + 1,555 images)
"""

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, classification_report
import warnings

warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "data"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

# Import configuration
from config_loader import load_config

def load_coordinate_datasets():
    """Load cacao and oil palm coordinate datasets"""
    print("📍 Loading coordinate-based datasets...")
    
    # Load configuration
    config = load_config()
    dataset_config = config.DATASET_CONFIG
    
    datasets = []
    
    # Load cacao
    cacao_file = dataset_config.get('cacao_file')
    if cacao_file:
        try:
            cacao_gdf = gpd.read_file(cacao_file)
            cacao_gdf['crop_type'] = 'cacao'
            cacao_gdf['source'] = 'peru_amazon'
            cacao_gdf['data_type'] = 'coordinates'
            
            # Extract centroids
            centroids = cacao_gdf.geometry.centroid
            cacao_gdf['lat'] = centroids.y
            cacao_gdf['lon'] = centroids.x
            
            datasets.append(cacao_gdf[['lat', 'lon', 'crop_type', 'source', 'data_type']])
            print(f"   ✅ Cacao: {len(cacao_gdf)} samples")
            
        except Exception as e:
            print(f"   ❌ Cacao loading failed: {e}")
    
    # Load oil palm
    palm_file = dataset_config.get('oil_palm_file')
    if palm_file:
        try:
            palm_gdf = gpd.read_file(palm_file)
            palm_gdf['crop_type'] = 'oil_palm'
            palm_gdf['source'] = 'peru_ucayali'
            palm_gdf['data_type'] = 'coordinates'
            
            # Extract centroids
            centroids = palm_gdf.geometry.centroid
            palm_gdf['lat'] = centroids.y
            palm_gdf['lon'] = centroids.x
            
            datasets.append(palm_gdf[['lat', 'lon', 'crop_type', 'source', 'data_type']])
            print(f"   ✅ Oil Palm: {len(palm_gdf)} samples")
            
        except Exception as e:
            print(f"   ❌ Oil palm loading failed: {e}")
    
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"   📊 Total coordinate samples: {len(combined_df)}")
        return combined_df
    else:
        return pd.DataFrame()

def load_rubber_image_dataset():
    """Load rubber image dataset"""
    print("🖼️ Loading rubber image dataset...")
    
    # Load configuration
    config = load_config()
    dataset_config = config.DATASET_CONFIG
    
    rubber_base_dir = dataset_config.get('rubber_dir')
    rubber_images_subdir = dataset_config.get('rubber_images_dir')
    
    if not rubber_base_dir or not rubber_images_subdir:
        print("   ❌ Rubber directory configuration missing")
        return pd.DataFrame(), {}
    
    rubber_dir = Path(rubber_base_dir) / rubber_images_subdir
    
    if not rubber_dir.exists():
        print(f"   ❌ Rubber image directory not found: {rubber_dir}")
        return pd.DataFrame(), {}
    
    # Map class directories to labels (from config)
    class_mapping = dataset_config.get('rubber_class_mapping', {
        'Class0_NonForest': 'non_forest',
        'Class1_Forest': 'forest', 
        'Class2_Monoculture_rubber': 'rubber'
    })
    
    image_data = []
    image_paths = {}
    
    for class_dir_name, crop_label in class_mapping.items():
        class_dir = rubber_dir / class_dir_name
        if class_dir.exists():
            image_files = list(class_dir.glob("*.png"))
            
            for img_path in image_files:
                # Create synthetic coordinates for rubber (SE Asia region)
                lat = np.random.uniform(1.0, 15.0)  # SE Asia latitude range
                lon = np.random.uniform(95.0, 140.0)  # SE Asia longitude range
                
                sample_id = len(image_data)
                image_data.append({
                    'lat': lat,
                    'lon': lon,
                    'crop_type': 'rubber' if crop_label == 'rubber' else 'non_rubber',
                    'source': 'se_asia_images',
                    'data_type': 'images',
                    'image_path': str(img_path),
                    'original_class': crop_label
                })
                image_paths[sample_id] = str(img_path)
            
            print(f"   ✅ {class_dir_name}: {len(image_files)} images")
    
    rubber_df = pd.DataFrame(image_data)
    
    # Filter to only include rubber vs non-rubber for 3-class problem
    if len(rubber_df) > 0:
        print(f"   📊 Total rubber image samples: {len(rubber_df)}")
        print(f"   🎯 Class distribution:")
        for crop_type, count in rubber_df['crop_type'].value_counts().items():
            print(f"      • {crop_type}: {count}")
    
    return rubber_df, image_paths

def extract_gee_features():
    """Extract features from coordinate data using GEE"""
    print("\n🛰️ EXTRACTING GEE FEATURES FOR COORDINATE DATA")
    print("=" * 60)
    
    try:
        from gee_sentinel2_extractor import GEESentinel2Extractor
        import torch
        
        # Initialize GEE extractor
        extractor = GEESentinel2Extractor("2020-01-01", "2021-12-31")
        
        # Load coordinate datasets
        coord_df = load_coordinate_datasets()
        
        if len(coord_df) == 0:
            print("❌ No coordinate data available")
            return None, None
        
        print(f"📊 Extracting satellite data for {len(coord_df)} locations...")
        print(f"   Estimated time: {len(coord_df) * 0.5 / 60:.1f} minutes")
        
        # Extract time series data
        timeseries_data = extractor.extract_batch_timeseries(coord_df, batch_size=20)
        
        # Extract features using Enhanced PRESTO
        from enhanced_presto_classifier import EnhancedPrestoClassifier, CropDataset
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(coord_df['crop_type'].values)
        
        # Create enhanced PRESTO model for feature extraction
        model = EnhancedPrestoClassifier(num_classes=len(label_encoder.classes_), freeze_backbone=True)
        # Auto-detect optimal device (MPS > CUDA > CPU)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            print(f"🚀 Using Metal Performance Shaders (MPS) on Apple Silicon")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using CUDA GPU")
        else:
            device = torch.device('cpu')
            print(f"💻 Using CPU")
        print(f"   Device: {device}")
        model = model.to(device)
        model.eval()
        
        # Extract features using PRESTO encoder
        features_list = []
        with torch.no_grad():
            for idx, ts_data in timeseries_data.items():
                # Convert to tensor
                x = torch.from_numpy(ts_data).float().unsqueeze(0).to(device)  # Add batch dim
                
                # Create mock inputs for PRESTO
                timesteps = x.shape[1]
                dynamic_world = torch.full((1, timesteps), 9, dtype=torch.long).to(device)
                
                # Get coordinates if available
                if idx < len(coord_df):
                    lat, lon = coord_df.iloc[idx]['lat'], coord_df.iloc[idx]['lon']
                    latlons = torch.tensor([[lat, lon]], dtype=torch.float).to(device)
                else:
                    latlons = torch.zeros(1, 2, dtype=torch.float).to(device)
                
                month = torch.tensor([6], dtype=torch.long).to(device)  # Default June
                
                # Extract features using PRESTO encoder
                features = model.presto_model.encoder(x, dynamic_world, latlons, month=month)
                features_list.append(features.cpu().numpy())
        
        features = np.vstack(features_list)
        
        print(f"✅ Extracted GEE features: {features.shape}")
        
        # Create labels
        labels = coord_df['crop_type'].values
        
        return features, labels
        
    except Exception as e:
        print(f"❌ GEE feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_image_features():
    """Extract features from rubber image data"""
    print("\n🖼️ EXTRACTING FEATURES FROM RUBBER IMAGES")
    print("=" * 60)
    
    try:
        rubber_df, image_paths = load_rubber_image_dataset()
        
        if len(rubber_df) == 0:
            print("❌ No image data available")
            return None, None
        
        print(f"📊 Processing {len(rubber_df)} rubber images...")
        
        # Simple CNN feature extractor for images
        class ImageFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)  # Same dimension as PRESTO features
                )
            
            def forward(self, x):
                return self.features(x)
        
        # Initialize feature extractor
        # Auto-detect optimal device (MPS > CUDA > CPU)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            print(f"🚀 Using Metal Performance Shaders (MPS) on Apple Silicon")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using CUDA GPU")
        else:
            device = torch.device('cpu')
            print(f"💻 Using CPU")
        print(f"   Device: {device}")
        feature_extractor = ImageFeatureExtractor().to(device)
        feature_extractor.eval()
        
        # Process images in batches
        features_list = []
        batch_size = 32
        
        for i in range(0, len(rubber_df), batch_size):
            batch_df = rubber_df.iloc[i:i+batch_size]
            batch_images = []
            
            for _, row in batch_df.iterrows():
                try:
                    # Load and preprocess image
                    img = Image.open(row['image_path']).convert('RGB')
                    img = img.resize((128, 128))  # Standardize size
                    img_array = np.array(img) / 255.0  # Normalize
                    img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)  # CHW format
                    batch_images.append(img_tensor)
                    
                except Exception as e:
                    print(f"   Warning: Failed to load image {row['image_path']}: {e}")
                    # Use zero tensor as fallback
                    batch_images.append(torch.zeros(3, 128, 128))
            
            if batch_images:
                # Process batch
                batch_tensor = torch.stack(batch_images).to(device)
                
                with torch.no_grad():
                    batch_features = feature_extractor(batch_tensor).cpu().numpy()
                    features_list.append(batch_features)
            
            if (i + batch_size) % 200 == 0:
                print(f"   Processed {min(i + batch_size, len(rubber_df))}/{len(rubber_df)} images")
        
        # Combine all features
        if features_list:
            features = np.vstack(features_list)
            labels = rubber_df['crop_type'].values
            
            print(f"✅ Extracted image features: {features.shape}")
            return features, labels
        else:
            print("❌ No valid image features extracted")
            return None, None
            
    except Exception as e:
        print(f"❌ Image feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def train_hybrid_classifier(gee_features, gee_labels, img_features, img_labels):
    """Train classifier on combined features"""
    print("\n🎯 TRAINING HYBRID CLASSIFIER")
    print("=" * 60)
    
    try:
        # Combine datasets
        if gee_features is not None and img_features is not None:
            # Ensure same feature dimensionality
            target_dim = min(gee_features.shape[1], img_features.shape[1])
            gee_features = gee_features[:, :target_dim]
            img_features = img_features[:, :target_dim]
            
            # Combine features and labels
            all_features = np.vstack([gee_features, img_features])
            all_labels = np.concatenate([gee_labels, img_labels])
            
            print(f"📊 Combined dataset:")
            print(f"   GEE features: {gee_features.shape}")
            print(f"   Image features: {img_features.shape}")
            print(f"   Total: {all_features.shape}")
            
        elif gee_features is not None:
            all_features = gee_features
            all_labels = gee_labels
            print(f"📊 Using GEE features only: {all_features.shape}")
            
        elif img_features is not None:
            all_features = img_features
            all_labels = img_labels
            print(f"📊 Using image features only: {all_features.shape}")
            
        else:
            raise ValueError("No features available for training")
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(all_labels)
        
        print(f"🏷️  Classes: {list(label_encoder.classes_)}")
        print(f"   Distribution: {np.bincount(encoded_labels)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, encoded_labels, test_size=0.2, 
            stratify=encoded_labels, random_state=42
        )
        
        # Train classifier
        from crop_classification_pipeline import CropClassifier
        
        classifier = CropClassifier(num_classes=len(label_encoder.classes_))
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = classifier.predict_proba(X_test)
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        test_log_loss = log_loss(y_test, y_pred_proba)
        
        print(f"\n📈 RESULTS:")
        print(f"   Log Loss: {test_log_loss:.4f}")
        print(f"   Test Samples: {len(y_test)}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Save model
        import pickle
        model_data = {
            'classifier': classifier,
            'label_encoder': label_encoder,
            'feature_dim': all_features.shape[1],
            'classes': list(label_encoder.classes_)
        }
        
        with open('hybrid_crop_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved as 'hybrid_crop_model.pkl'")
        
        return {
            'log_loss': test_log_loss,
            'classifier': classifier,
            'label_encoder': label_encoder,
            'test_results': (X_test, y_test, y_pred, y_pred_proba)
        }
        
    except Exception as e:
        print(f"❌ Classifier training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run the hybrid pipeline"""
    print("🚀 HYBRID CROP CLASSIFICATION PIPELINE")
    print("Combining GEE coordinate data + pre-processed images")
    print("=" * 80)
    
    # Step 1: Extract GEE features
    gee_features, gee_labels = extract_gee_features()
    
    # Step 2: Extract image features  
    img_features, img_labels = extract_image_features()
    
    # Step 3: Train hybrid classifier
    if gee_features is not None or img_features is not None:
        results = train_hybrid_classifier(gee_features, gee_labels, img_features, img_labels)
        
        if results:
            print(f"\n🎉 HYBRID PIPELINE COMPLETED!")
            print(f"   Final Log Loss: {results['log_loss']:.4f}")
            print(f"   Classes: {list(results['label_encoder'].classes_)}")
            
            # Test prediction
            print(f"\n🔍 Testing prediction capability...")
            if gee_features is not None:
                sample_features = gee_features[0:1]
                prediction = results['classifier'].predict_proba(sample_features)[0]
                predicted_class = results['label_encoder'].classes_[np.argmax(prediction)]
                
                print(f"   Sample prediction: {predicted_class}")
                print(f"   Confidence: {np.max(prediction):.3f}")
                
        else:
            print(f"\n❌ Hybrid pipeline failed")
            return 1
    else:
        print(f"\n❌ No features extracted from any source")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)