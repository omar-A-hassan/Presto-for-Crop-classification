#!/usr/bin/env python3
"""
Enhanced PRESTO Training Script with Two-Stage Fine-tuning
=========================================================

This script implements the strategic two-stage fine-tuning approach for
robust crop classification using the PRESTO foundation model.

Strategy:
1. Stage 1: Freeze PRESTO backbone, train classification head (10-15 epochs)
2. Stage 2: Unfreeze last 2-3 PRESTO layers, fine-tune with low LR (5-10 epochs)

Features:
- Geographic stratification for robust validation
- Pre-trained PRESTO weights loading
- Focal loss and label smoothing for better calibration
- Attention pooling for variable timesteps
- Comprehensive evaluation metrics
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "data"))
sys.path.append(str(project_root / "src" / "utils"))
sys.path.append(str(project_root / "presto"))
sys.path.append(str(project_root / "presto" / "presto"))

# Import our modules
from enhanced_presto_classifier import (
    EnhancedPrestoClassifier, 
    EnhancedPrestoTrainer, 
    CropDataset, 
    collate_fn,
    create_geographic_splits
)
from config_loader import load_config
from gee_sentinel2_extractor import GEESentinel2Extractor


def load_hybrid_datasets():
    """Load both coordinate-based and image-based datasets"""
    print("📊 LOADING HYBRID DATASETS")
    print("=" * 50)
    
    config = load_config()
    dataset_config = config.DATASET_CONFIG
    
    # Initialize containers
    all_timeseries = {}
    all_labels = []
    all_coords = {}
    all_sources = []
    
    sample_idx = 0
    
    # 1. Load coordinate-based data with robust extractor
    print("\n1️⃣ Loading coordinate-based datasets...")
    try:
        # Use the robust data extractor
        from src.data.robust_data_extractor import RobustDataExtractor
        
        # Use config-based path to find existing extracted data
        config = load_config()
        extractor = RobustDataExtractor(output_dir=config.PATHS_CONFIG['extracted_data_dir'])
        
        # Check if we have existing complete data
        if extractor.load_existing_data() and len(extractor.all_labels) > 0:
            print("   📂 Using existing extracted data (no re-extraction needed)")
            coord_timeseries = extractor.all_timeseries
            coord_labels = extractor.all_labels
            coord_coords = extractor.all_coords
            coord_sources = extractor.all_sources
        else:
            print("   🛰️ No existing data found, starting fresh extraction...")
            coord_timeseries, coord_labels, coord_coords, coord_sources = extractor.extract_all_datasets()
        
        # Merge with existing data
        for idx, ts_data in coord_timeseries.items():
            all_timeseries[sample_idx] = ts_data
            sample_idx += 1
        
        all_labels.extend(coord_labels)
        all_coords.update(coord_coords)
        all_sources.extend(coord_sources)
        
        print(f"   ✅ Successfully loaded {len(coord_labels)} coordinate-based samples")
            
    except Exception as e:
        print(f"   ⚠️ Coordinate data extraction failed: {e}")
        print(f"   🔍 Error type: {type(e).__name__}")
        print("      The training will continue with available image data only.")
    
    # 2. Load image-based data (rubber)
    print("\n2️⃣ Loading rubber image dataset...")
    try:
        rubber_base_dir = Path(dataset_config.get('rubber_dir', ''))
        rubber_images_subdir = dataset_config.get('rubber_images_dir', '')
        
        if rubber_base_dir.exists():
            rubber_dir = rubber_base_dir / rubber_images_subdir
            
            if rubber_dir.exists():
                # Process rubber images 
                class_mapping = dataset_config.get('rubber_class_mapping', {
                    'Class2_Monoculture_rubber': 'rubber'
                })
                
                rubber_count = 0
                for class_dir_name, crop_label in class_mapping.items():
                    if crop_label == 'rubber':  # Only process rubber class
                        class_dir = rubber_dir / class_dir_name
                        if class_dir.exists():
                            image_files = list(class_dir.glob("*.png"))
                            
                            # Convert images to synthetic time series (simplified)
                            for img_path in image_files[:500]:  # Limit for demo
                                # Create synthetic time series for rubber images
                                # In practice, you'd extract features from images
                                synthetic_ts = create_synthetic_rubber_timeseries()
                                
                                all_timeseries[sample_idx] = synthetic_ts
                                all_labels.append('rubber')
                                
                                # Synthetic SE Asia coordinates
                                lat = np.random.uniform(1.0, 15.0)
                                lon = np.random.uniform(95.0, 140.0)
                                all_coords[sample_idx] = (lat, lon)
                                all_sources.append('se_asia_images')
                                
                                sample_idx += 1
                                rubber_count += 1
                
                print(f"   🖼️ Processed {rubber_count} rubber image samples")
                
    except Exception as e:
        print(f"   ⚠️ Rubber image processing failed: {e}")
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total samples: {len(all_timeseries)}")
    
    # Count by class
    from collections import Counter
    class_counts = Counter(all_labels)
    for crop_type, count in class_counts.items():
        print(f"   {crop_type}: {count} samples")
    
    # Count by source
    source_counts = Counter(all_sources)
    for source, count in source_counts.items():
        print(f"   {source}: {count} samples")
    
    return all_timeseries, all_labels, all_coords, all_sources


def create_synthetic_rubber_timeseries():
    """Create synthetic time series for rubber images (simplified approach)"""
    # This is a simplified approach - in practice you'd extract features from the actual images
    n_timesteps = 12
    n_bands = 14  # Match satellite data: 10 bands + 4 vegetation indices
    
    # Initialize with rubber-specific patterns (higher NIR, specific seasonal patterns)
    ts = np.random.normal(0.3, 0.1, (n_timesteps, n_bands))
    
    # Add seasonal patterns typical of rubber
    for t in range(n_timesteps):
        month = t + 1
        seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (month - 3) / 12)
        
        # Rubber has distinct dry season (leaf fall)
        if month in [2, 3, 4]:  # Dry season
            ts[t, 7:9] *= 0.6  # Lower NIR during leaf fall
            ts[t, 10] *= 0.5    # Lower NDVI during leaf fall
            ts[t, 11] *= 0.5    # Lower EVI during leaf fall
        else:
            ts[t, 7:9] *= 1.2   # Higher NIR during growing season
            ts[t, 10] *= seasonal_factor  # NDVI follows season
            ts[t, 11] *= seasonal_factor  # EVI follows season
        
        # Set vegetation indices (bands 10-13: NDVI, EVI, NDWI, RNDVI)
        # NDVI: typical rubber values 0.4-0.8
        ts[t, 10] = np.clip(0.6 * seasonal_factor + np.random.normal(0, 0.05), 0.2, 0.9)
        # EVI: enhanced vegetation index
        ts[t, 11] = np.clip(0.4 * seasonal_factor + np.random.normal(0, 0.03), 0.1, 0.7)
        # NDWI: water index (rubber plantations often well-watered)
        ts[t, 12] = np.clip(-0.1 + np.random.normal(0, 0.02), -0.5, 0.2)
        # RNDVI: red edge NDVI
        ts[t, 13] = np.clip(0.5 * seasonal_factor + np.random.normal(0, 0.03), 0.2, 0.8)
    
    return np.clip(ts, 0, 1)


def prepare_datasets(timeseries_data, labels, coords, sources):
    """Prepare datasets with geographic stratification"""
    print("\n🔀 PREPARING DATASETS WITH GEOGRAPHIC STRATIFICATION")
    print("=" * 60)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Create geographic splits (with class-aware stratification)
    train_mask, val_mask, test_mask = create_geographic_splits(
        sources, labels=labels, test_size=0.2, val_size=0.15, random_state=42
    )
    
    # Split data
    train_indices = [i for i, mask in enumerate(train_mask) if mask]
    val_indices = [i for i, mask in enumerate(val_mask) if mask]
    test_indices = [i for i, mask in enumerate(test_mask) if mask]
    
    print(f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create dataset objects
    def create_subset_data(indices):
        subset_ts = {i: timeseries_data[idx] for i, idx in enumerate(indices)}
        subset_labels = {i: encoded_labels[idx] for i, idx in enumerate(indices)}
        subset_coords = {i: coords[idx] for i, idx in enumerate(indices)} if coords else None
        return subset_ts, subset_labels, subset_coords
    
    train_ts, train_labels, train_coords = create_subset_data(train_indices)
    val_ts, val_labels, val_coords = create_subset_data(val_indices)
    test_ts, test_labels, test_coords = create_subset_data(test_indices)
    
    # Create datasets
    train_dataset = CropDataset(train_ts, train_labels, train_coords)
    val_dataset = CropDataset(val_ts, val_labels, val_coords)
    test_dataset = CropDataset(test_ts, test_labels, test_coords)
    
    print(f"✅ Datasets created successfully")
    
    return train_dataset, val_dataset, test_dataset, label_encoder


def train_two_stage_model_with_logging(train_dataset, val_dataset, label_encoder, writer):
    """Train model using two-stage strategy with TensorBoard logging"""
    print("\n🎯 TWO-STAGE TRAINING STRATEGY")
    print("=" * 60)
    
    # Setup
    from enhanced_presto_classifier import get_optimal_device, print_device_info
    device = get_optimal_device()
    print_device_info(device)
    
    # Create model
    model = EnhancedPrestoClassifier(
        num_classes=len(label_encoder.classes_),
        freeze_backbone=True,  # Start with frozen backbone
        unfreeze_layers=2
    )
    
    # Create trainer
    trainer = EnhancedPrestoTrainer(
        model=model,
        device=device,
        use_focal_loss=True,
        use_label_smoothing=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # STAGE 1: Train classification head only
    print("\n🔒 STAGE 1: Training Classification Head (PRESTO Frozen)")
    print("-" * 50)
    
    trainer.setup_stage1_training(lr=1e-3)
    
    stage1_epochs = 15
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    stage1_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_log_loss': []}
    
    for epoch in range(stage1_epochs):
        print(f"\nStage 1 Epoch {epoch+1}/{stage1_epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, trainer.stage1_optimizer, "Stage 1")
        
        # Validate
        val_loss, val_acc, val_log_loss = trainer.validate(val_loader)
        
        # Store history
        stage1_history['train_loss'].append(train_loss)
        stage1_history['train_acc'].append(train_acc)
        stage1_history['val_loss'].append(val_loss)
        stage1_history['val_acc'].append(val_acc)
        stage1_history['val_log_loss'].append(val_log_loss)
        
        # TensorBoard logging
        writer.add_scalar('stage1/train_loss', train_loss, epoch)
        writer.add_scalar('stage1/train_acc', train_acc, epoch)
        writer.add_scalar('stage1/val_loss', val_loss, epoch)
        writer.add_scalar('stage1/val_acc', val_acc, epoch)
        writer.add_scalar('stage1/val_log_loss', val_log_loss, epoch)
        
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, LogLoss={val_log_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best Stage 1 model to checkpoints directory
            stage1_path = Path('models/checkpoints/best_stage1_model.pth')
            stage1_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), stage1_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    # Load best Stage 1 model
    stage1_path = Path('models/checkpoints/best_stage1_model.pth')
    model.load_state_dict(torch.load(stage1_path))
    print(f"✅ Stage 1 completed. Best val loss: {best_val_loss:.4f}")
    
    # STAGE 2: Fine-tune last PRESTO layers
    print("\n🔓 STAGE 2: Fine-tuning PRESTO Layers (Selective Unfreezing)")
    print("-" * 50)
    
    trainer.setup_stage2_training(lr=1e-5)
    
    stage2_epochs = 10
    best_val_loss = float('inf')
    patience_counter = 0
    
    stage2_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_log_loss': []}
    
    for epoch in range(stage2_epochs):
        print(f"\nStage 2 Epoch {epoch+1}/{stage2_epochs}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, trainer.stage2_optimizer, "Stage 2")
        
        # Validate
        val_loss, val_acc, val_log_loss = trainer.validate(val_loader)
        
        # Store history
        stage2_history['train_loss'].append(train_loss)
        stage2_history['train_acc'].append(train_acc)
        stage2_history['val_loss'].append(val_loss)
        stage2_history['val_acc'].append(val_acc)
        stage2_history['val_log_loss'].append(val_log_loss)
        
        # TensorBoard logging
        writer.add_scalar('stage2/train_loss', train_loss, epoch)
        writer.add_scalar('stage2/train_acc', train_acc, epoch)
        writer.add_scalar('stage2/val_loss', val_loss, epoch)
        writer.add_scalar('stage2/val_acc', val_acc, epoch)
        writer.add_scalar('stage2/val_log_loss', val_log_loss, epoch)
        
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, LogLoss={val_log_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best final model to checkpoints directory
            final_path = Path('models/checkpoints/best_final_model.pth')
            final_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), final_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    # Load best final model
    final_path = Path('models/checkpoints/best_final_model.pth')
    model.load_state_dict(torch.load(final_path))
    print(f"✅ Stage 2 completed. Best val loss: {best_val_loss:.4f}")
    
    return model, trainer, stage1_history, stage2_history


def evaluate_model_with_logging(model, trainer, test_dataset, label_encoder, writer):
    """Comprehensive model evaluation with TensorBoard logging"""
    print("\n📊 FINAL MODEL EVALUATION")
    print("=" * 50)
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Evaluate
    test_loss, test_acc, test_log_loss = trainer.validate(test_loader)
    
    print(f"📈 Test Results:")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   Log Loss: {test_log_loss:.4f}")
    print(f"   Cross-entropy Loss: {test_loss:.4f}")
    
    # TensorBoard logging
    writer.add_scalar('test/accuracy', test_acc, 0)
    writer.add_scalar('test/log_loss', test_log_loss, 0)
    writer.add_scalar('test/cross_entropy_loss', test_loss, 0)
    
    # Detailed predictions for classification report
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            dynamic_world = batch['dynamic_world'].to(trainer.device)
            latlons = batch['latlons'].to(trainer.device)
            month = batch['month'].to(trainer.device)
            
            logits = model(x, dynamic_world, latlons, month=month)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n🔢 Confusion Matrix:")
    print(cm)
    
    # Log confusion matrix to TensorBoard
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_, ax=ax)
    ax.set_title('Test Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    writer.add_figure('test/confusion_matrix', fig, 0)
    plt.close(fig)
    
    return {
        'test_accuracy': test_acc,
        'test_log_loss': test_log_loss,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }


def save_model_and_results(model, label_encoder, results, history):
    """Save trained model and results"""
    print("\n💾 SAVING MODEL AND RESULTS")
    print("=" * 40)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Create final model directory
    final_model_dir = Path('models/final')
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model in final directory
    model_path = final_model_dir / 'enhanced_presto_crop_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'model_config': {
            'num_classes': model.num_classes,
            'freeze_backbone': model.freeze_backbone,
            'unfreeze_layers': model.unfreeze_layers
        }
    }, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    
    # Save results
    results_path = results_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'test_accuracy': float(results['test_accuracy']),
            'test_log_loss': float(results['test_log_loss']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'class_names': list(label_encoder.classes_)
        }, f, indent=2)
    
    print(f"✅ Results saved to: {results_path}")
    
    # Save training history
    history_path = results_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training history saved to: {history_path}")


def main():
    """Main training pipeline"""
    print("🚀 ENHANCED PRESTO CROP CLASSIFICATION TRAINING")
    print("=" * 80)
    
    # Setup TensorBoard logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = Path("logs/tensorboard") / f"enhanced_presto_{timestamp}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"📊 TensorBoard logs: {tb_dir}")
    
    try:
        # Load datasets
        timeseries_data, labels, coords, sources = load_hybrid_datasets()
        
        if len(timeseries_data) == 0:
            print("❌ No data loaded. Exiting.")
            return
        
        # Log data statistics
        label_counts = Counter(labels)
        source_counts = Counter(sources)
        
        print(f"📈 Logging data statistics to TensorBoard")
        
        # Log label distribution
        for label, count in label_counts.items():
            writer.add_scalar(f'data/label_count_{label}', count, 0)
        
        # Log source distribution  
        for source, count in source_counts.items():
            writer.add_scalar(f'data/source_count_{source}', count, 0)
        
        # Log overall statistics
        all_means = [timeseries_data[i].mean() for i in range(len(timeseries_data))]
        all_stds = [timeseries_data[i].std() for i in range(len(timeseries_data))]
        
        writer.add_scalar('data/overall_mean', np.mean(all_means), 0)
        writer.add_scalar('data/overall_std', np.mean(all_stds), 0)
        writer.add_scalar('data/total_samples', len(timeseries_data), 0)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset, label_encoder = prepare_datasets(
            timeseries_data, labels, coords, sources
        )
        
        # Train model with logging
        model, trainer, stage1_history, stage2_history = train_two_stage_model_with_logging(
            train_dataset, val_dataset, label_encoder, writer
        )
        
        # Evaluate model
        results = evaluate_model_with_logging(model, trainer, test_dataset, label_encoder, writer)
        
        # Save everything
        history = {
            'stage1': stage1_history,
            'stage2': stage2_history
        }
        save_model_and_results(model, label_encoder, results, history)
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final Test Accuracy: {results['test_accuracy']:.2f}%")
        print(f"Final Test Log Loss: {results['test_log_loss']:.4f}")
        print(f"\n🔍 To view TensorBoard logs, run:")
        print(f"   tensorboard --logdir {tb_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        writer.close()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)