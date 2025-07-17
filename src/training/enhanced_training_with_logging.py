#!/usr/bin/env python3
"""
Enhanced PRESTO Training with Robust Logging and Validation
==========================================================

This script implements comprehensive training with:
- TensorBoard logging
- Cross-validation
- Proper experiment tracking
- Config-based path management
- Data leakage prevention
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
import uuid

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
    create_geographic_splits,
    get_optimal_device,
    print_device_info
)
from config_loader import load_config


class ExperimentTracker:
    """Track experiments with proper logging and path management"""
    
    def __init__(self, experiment_name: str = None, config: dict = None):
        self.config = config or load_config()
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_id = str(uuid.uuid4())[:8]
        
        # Create experiment directories
        self.setup_experiment_dirs()
        
        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(
            log_dir=str(self.tensorboard_dir),
            comment=f"_{self.experiment_name}"
        )
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'git_commit': self.get_git_commit(),
            'device': str(get_optimal_device())
        }
        
        print(f"🧪 Experiment: {self.experiment_name} (ID: {self.experiment_id})")
        print(f"📊 TensorBoard: {self.tensorboard_dir}")
        print(f"💾 Logs: {self.log_dir}")
    
    def setup_experiment_dirs(self):
        """Create experiment-specific directories"""
        paths = self.config.PATHS_CONFIG
        
        # Create base directories
        self.log_dir = Path(paths['experiment_logs_dir']) / self.experiment_name
        self.tensorboard_dir = Path(paths['tensorboard_dir']) / self.experiment_name
        self.checkpoint_dir = Path(paths['model_checkpoints_dir']) / self.experiment_name
        self.results_dir = Path(paths['evaluations_dir']) / self.experiment_name
        
        # Create directories
        for dir_path in [self.log_dir, self.tensorboard_dir, self.checkpoint_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_git_commit(self):
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=project_root)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def log_hyperparameters(self, hparams: dict):
        """Log hyperparameters to TensorBoard"""
        # Convert complex objects to strings for TensorBoard
        clean_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)
        
        self.tb_writer.add_hparams(clean_hparams, {})
        
        # Save to JSON
        with open(self.log_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
    
    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log metrics to TensorBoard and JSON"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names, step: int, split: str = ""):
        """Log confusion matrix to TensorBoard"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Confusion Matrix - {split}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Log to TensorBoard
        self.tb_writer.add_figure(f'confusion_matrix/{split}', fig, step)
        plt.close(fig)
        
        # Save to results directory
        np.save(self.results_dir / f'confusion_matrix_{split}_step_{step}.npy', cm)
    
    def save_model_checkpoint(self, model, optimizer, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'experiment_id': self.experiment_id,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved: {best_path}")
    
    def finalize_experiment(self, final_metrics: dict):
        """Finalize experiment and save metadata"""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['final_metrics'] = final_metrics
        self.metadata['status'] = 'completed'
        
        # Save experiment metadata
        with open(self.log_dir / 'experiment_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        self.tb_writer.close()
        print(f"✅ Experiment completed: {self.experiment_name}")


class RobustCrossValidator:
    """Robust cross-validation with data leakage prevention"""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, random_state: int = 42):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
    
    def create_splits(self, timeseries_data: dict, labels: list, sources: list, coords: dict):
        """Create robust train/val/test splits preventing data leakage"""
        
        # First, create holdout test set (never used during training)
        indices = list(range(len(labels)))
        train_val_indices, test_indices = train_test_split(
            indices, test_size=self.test_size, stratify=labels, random_state=self.random_state
        )
        
        # Extract train/val data
        train_val_labels = [labels[i] for i in train_val_indices]
        train_val_sources = [sources[i] for i in train_val_indices]
        
        # Create cross-validation folds on train/val data
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        cv_splits = []
        
        for fold_idx, (train_idx_rel, val_idx_rel) in enumerate(skf.split(train_val_indices, train_val_labels)):
            # Convert relative indices back to absolute indices
            train_indices = [train_val_indices[i] for i in train_idx_rel]
            val_indices = [train_val_indices[i] for i in val_idx_rel]
            
            # Create data splits
            train_data = self.create_data_split(train_indices, timeseries_data, labels, coords)
            val_data = self.create_data_split(val_indices, timeseries_data, labels, coords)
            
            cv_splits.append({
                'fold': fold_idx,
                'train': train_data,
                'val': val_data,
                'train_indices': train_indices,
                'val_indices': val_indices
            })
        
        # Create test split
        test_data = self.create_data_split(test_indices, timeseries_data, labels, coords)
        
        return cv_splits, test_data, test_indices
    
    def create_data_split(self, indices: list, timeseries_data: dict, labels: list, coords: dict):
        """Create a data split from indices"""
        split_timeseries = {i: timeseries_data[idx] for i, idx in enumerate(indices)}
        split_labels = {i: labels[idx] for i, idx in enumerate(indices)}
        split_coords = {i: coords[idx] for i, idx in enumerate(indices)} if coords else None
        
        return {
            'timeseries': split_timeseries,
            'labels': split_labels,
            'coords': split_coords,
            'original_indices': indices
        }


def load_data_with_config():
    """Load data using config paths"""
    print("📊 LOADING DATA WITH CONFIG PATHS")
    print("=" * 50)
    
    config = load_config()
    
    # Use the robust data extractor with config paths
    from robust_data_extractor import RobustDataExtractor
    
    # Initialize with config-based output directory
    extractor = RobustDataExtractor(
        output_dir=config.PATHS_CONFIG['extracted_data_dir']
    )
    
    # Load existing data if available
    if extractor.load_existing_data() and len(extractor.all_labels) > 0:
        print("   📂 Using existing extracted data")
        return extractor.all_timeseries, extractor.all_labels, extractor.all_coords, extractor.all_sources
    else:
        print("   🛰️ No existing data found, would need fresh extraction")
        print("   🔧 For now, returning mock data for testing")
        
        # Create mock data for testing the pipeline
        mock_timeseries = {}
        mock_labels = []
        mock_coords = {}
        mock_sources = []
        
        # Create balanced mock data
        crops = ['cacao', 'oil_palm', 'rubber']
        for i in range(300):  # 100 samples per crop
            crop = crops[i % 3]
            mock_timeseries[i] = np.random.rand(12, 14)  # 12 timesteps, 14 features
            mock_labels.append(crop)
            mock_coords[i] = (np.random.uniform(-10, 10), np.random.uniform(-80, -70))
            mock_sources.append(f"mock_{crop}_region")
        
        print(f"   🎭 Created mock data: {len(mock_labels)} samples")
        return mock_timeseries, mock_labels, mock_coords, mock_sources


def train_with_cross_validation(experiment_name: str = None):
    """Main training function with cross-validation"""
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name)
    config = tracker.config
    
    try:
        # Load data
        timeseries_data, labels, coords, sources = load_data_with_config()
        
        # Create label encoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Initialize cross-validator
        cv = RobustCrossValidator(
            n_splits=config.MODEL_CONFIG.get('cv_folds', 5),
            test_size=config.MODEL_CONFIG.get('test_size', 0.2),
            random_state=42
        )
        
        # Create splits
        cv_splits, test_data, test_indices = cv.create_splits(
            timeseries_data, labels, sources, coords
        )
        
        print(f"📊 Created {len(cv_splits)} CV folds + holdout test set")
        
        # Training hyperparameters
        hparams = {
            'num_folds': len(cv_splits),
            'learning_rate_stage1': 1e-3,
            'learning_rate_stage2': 1e-5,
            'batch_size': 16,
            'epochs_stage1': 15,
            'epochs_stage2': 10,
            'patience': 5,
            'use_focal_loss': True,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'num_classes': len(label_encoder.classes_),
            'unfreeze_layers': 2
        }
        
        tracker.log_hyperparameters(hparams)
        
        # Cross-validation training
        fold_results = []
        device = get_optimal_device()
        print_device_info(device)
        
        for fold_data in cv_splits:
            fold_idx = fold_data['fold']
            print(f"\\n🔄 TRAINING FOLD {fold_idx + 1}/{len(cv_splits)}")
            print("-" * 40)
            
            # Create datasets
            train_dataset = CropDataset(
                fold_data['train']['timeseries'],
                fold_data['train']['labels'],
                fold_data['train']['coords']
            )
            
            val_dataset = CropDataset(
                fold_data['val']['timeseries'],
                fold_data['val']['labels'],
                fold_data['val']['coords']
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=hparams['batch_size'], 
                shuffle=True, collate_fn=collate_fn, num_workers=2
            )
            
            val_loader = DataLoader(
                val_dataset, batch_size=hparams['batch_size'], 
                shuffle=False, collate_fn=collate_fn, num_workers=2
            )
            
            # Initialize model
            model = EnhancedPrestoClassifier(
                num_classes=hparams['num_classes'],
                freeze_backbone=True,
                unfreeze_layers=hparams['unfreeze_layers']
            )
            
            # Initialize trainer
            trainer = EnhancedPrestoTrainer(
                model=model, device=device,
                use_focal_loss=hparams['use_focal_loss'],
                use_label_smoothing=True
            )
            
            # Train fold
            fold_result = train_single_fold(
                trainer, train_loader, val_loader, 
                hparams, tracker, fold_idx, label_encoder
            )
            
            fold_results.append(fold_result)
        
        # Aggregate CV results
        cv_metrics = aggregate_cv_results(fold_results)
        print(f"\\n📊 CROSS-VALIDATION RESULTS:")
        print(f"   Mean Val Accuracy: {cv_metrics['mean_val_acc']:.3f} ± {cv_metrics['std_val_acc']:.3f}")
        print(f"   Mean Val Log Loss: {cv_metrics['mean_val_log_loss']:.3f} ± {cv_metrics['std_val_log_loss']:.3f}")
        
        # Log CV results
        tracker.log_metrics(cv_metrics, step=0, prefix="cross_validation")
        
        # Train final model on all training data and evaluate on test set
        print(f"\\n🎯 TRAINING FINAL MODEL")
        print("-" * 40)
        
        final_model_result = train_final_model(
            cv_splits, test_data, hparams, tracker, label_encoder
        )
        
        # Finalize experiment
        final_metrics = {**cv_metrics, **final_model_result}
        tracker.finalize_experiment(final_metrics)
        
        return final_metrics
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        tracker.metadata['status'] = 'failed'
        tracker.metadata['error'] = str(e)
        tracker.finalize_experiment({'status': 'failed'})
        
        raise e


def train_single_fold(trainer, train_loader, val_loader, hparams, tracker, fold_idx, label_encoder):
    """Train a single fold"""
    
    # Stage 1: Train classification head
    trainer.setup_stage1_training(lr=hparams['learning_rate_stage1'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(hparams['epochs_stage1']):
        # Train
        train_loss, train_acc = trainer.train_epoch(
            train_loader, trainer.stage1_optimizer, f"Fold{fold_idx}_Stage1"
        )
        
        # Validate
        val_loss, val_acc, val_log_loss = trainer.validate(val_loader)
        
        # Log metrics
        step = fold_idx * (hparams['epochs_stage1'] + hparams['epochs_stage2']) + epoch
        tracker.log_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_log_loss': val_log_loss
        }, step, f"fold_{fold_idx}/stage1")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hparams['patience']:
                print(f"   Early stopping at epoch {epoch + 1}")
                break
    
    # Stage 2: Fine-tune with unfrozen layers
    trainer.setup_stage2_training(lr=hparams['learning_rate_stage2'])
    
    for epoch in range(hparams['epochs_stage2']):
        # Train
        train_loss, train_acc = trainer.train_epoch(
            train_loader, trainer.stage2_optimizer, f"Fold{fold_idx}_Stage2"
        )
        
        # Validate
        val_loss, val_acc, val_log_loss = trainer.validate(val_loader)
        
        # Log metrics
        step = fold_idx * (hparams['epochs_stage1'] + hparams['epochs_stage2']) + hparams['epochs_stage1'] + epoch
        tracker.log_metrics({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_log_loss': val_log_loss
        }, step, f"fold_{fold_idx}/stage2")
    
    # Final validation
    final_val_loss, final_val_acc, final_val_log_loss = trainer.validate(val_loader)
    
    return {
        'fold': fold_idx,
        'val_acc': final_val_acc,
        'val_loss': final_val_loss,
        'val_log_loss': final_val_log_loss
    }


def aggregate_cv_results(fold_results):
    """Aggregate cross-validation results"""
    val_accs = [r['val_acc'] for r in fold_results]
    val_losses = [r['val_loss'] for r in fold_results]
    val_log_losses = [r['val_log_loss'] for r in fold_results]
    
    return {
        'mean_val_acc': np.mean(val_accs),
        'std_val_acc': np.std(val_accs),
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses),
        'mean_val_log_loss': np.mean(val_log_losses),
        'std_val_log_loss': np.std(val_log_losses),
        'fold_results': fold_results
    }


def train_final_model(cv_splits, test_data, hparams, tracker, label_encoder):
    """Train final model on all training data"""
    
    # Combine all training data from CV splits
    all_train_timeseries = {}
    all_train_labels = {}
    all_train_coords = {}
    
    sample_idx = 0
    for fold_data in cv_splits:
        for fold_type in ['train', 'val']:
            data = fold_data[fold_type]
            for i in range(len(data['labels'])):
                all_train_timeseries[sample_idx] = data['timeseries'][i]
                all_train_labels[sample_idx] = data['labels'][i]
                if data['coords']:
                    all_train_coords[sample_idx] = data['coords'][i]
                sample_idx += 1
    
    # Create datasets
    train_dataset = CropDataset(all_train_timeseries, all_train_labels, all_train_coords)
    test_dataset = CropDataset(test_data['timeseries'], test_data['labels'], test_data['coords'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Train final model (abbreviated training)
    device = get_optimal_device()
    model = EnhancedPrestoClassifier(
        num_classes=hparams['num_classes'],
        freeze_backbone=True,
        unfreeze_layers=hparams['unfreeze_layers']
    )
    
    trainer = EnhancedPrestoTrainer(model=model, device=device)
    
    # Quick training (fewer epochs for final model)
    trainer.setup_stage1_training(lr=hparams['learning_rate_stage1'])
    for epoch in range(5):  # Reduced epochs
        trainer.train_epoch(train_loader, trainer.stage1_optimizer, "Final_Stage1")
    
    trainer.setup_stage2_training(lr=hparams['learning_rate_stage2'])
    for epoch in range(3):  # Reduced epochs
        trainer.train_epoch(train_loader, trainer.stage2_optimizer, "Final_Stage2")
    
    # Evaluate on test set
    test_loss, test_acc, test_log_loss = trainer.validate(test_loader)
    
    print(f"🎯 FINAL TEST RESULTS:")
    print(f"   Test Accuracy: {test_acc:.3f}%")
    print(f"   Test Log Loss: {test_log_loss:.4f}")
    
    # Save final model
    model_path = tracker.checkpoint_dir / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'test_metrics': {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'test_log_loss': test_log_loss
        }
    }, model_path)
    
    return {
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_log_loss': test_log_loss,
        'final_model_path': str(model_path)
    }


def main():
    """Main training pipeline"""
    print("🚀 ENHANCED PRESTO TRAINING WITH ROBUST VALIDATION")
    print("=" * 80)
    
    # Create experiment name with timestamp
    experiment_name = f"enhanced_presto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run training
    results = train_with_cross_validation(experiment_name)
    
    print("\\n🎉 TRAINING COMPLETED!")
    print(f"Final Results: {results}")
    
    return results


if __name__ == "__main__":
    main()