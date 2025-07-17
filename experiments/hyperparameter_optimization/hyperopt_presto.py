#!/usr/bin/env python3
"""
Hyperparameter Optimization for Enhanced PRESTO
==============================================

This script implements Bayesian optimization for hyperparameter tuning
to minimize log loss on our crop classification task.
"""

import os
import sys
import optuna
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src" / "training"))
sys.path.append(str(project_root / "src" / "utils"))

from enhanced_training_with_logging import train_with_cross_validation, load_data_with_config
from config_loader import load_config


class PrestoHyperoptObjective:
    """Optuna objective function for PRESTO hyperparameter optimization"""
    
    def __init__(self, data_cache=None):
        self.config = load_config()
        self.data_cache = data_cache
        self.trial_count = 0
    
    def __call__(self, trial):
        """Objective function to minimize (log loss)"""
        self.trial_count += 1
        
        # Sample hyperparameters
        hparams = self.sample_hyperparameters(trial)
        
        # Create experiment name with trial info
        experiment_name = f"hyperopt_trial_{trial.number}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            # Run training with sampled hyperparameters
            results = train_with_cross_validation_objective(
                hparams, experiment_name, self.data_cache
            )
            
            # Return mean validation log loss (what we want to minimize)
            mean_log_loss = results['mean_val_log_loss']
            
            # Log additional metrics to trial
            trial.set_user_attr('mean_val_acc', results['mean_val_acc'])
            trial.set_user_attr('std_val_log_loss', results['std_val_log_loss'])
            trial.set_user_attr('experiment_name', experiment_name)
            
            print(f"Trial {trial.number}: Log Loss = {mean_log_loss:.4f}")
            
            return mean_log_loss
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            # Return a high log loss for failed trials
            return 10.0
    
    def sample_hyperparameters(self, trial):
        """Sample hyperparameters using Optuna"""
        
        # Learning rates
        lr_stage1 = trial.suggest_float('lr_stage1', 1e-4, 5e-3, log=True)
        lr_stage2 = trial.suggest_float('lr_stage2', 1e-6, 1e-4, log=True)
        
        # Architecture parameters
        unfreeze_layers = trial.suggest_int('unfreeze_layers', 1, 4)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Training parameters
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        epochs_stage1 = trial.suggest_int('epochs_stage1', 10, 25)
        epochs_stage2 = trial.suggest_int('epochs_stage2', 5, 15)
        
        # Loss function parameters
        use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
        focal_alpha = trial.suggest_float('focal_alpha', 0.25, 2.0) if use_focal_loss else 1.0
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0) if use_focal_loss else 2.0
        
        # Label smoothing
        use_label_smoothing = trial.suggest_categorical('use_label_smoothing', [True, False])
        label_smoothing = trial.suggest_float('label_smoothing', 0.05, 0.2) if use_label_smoothing else 0.1
        
        # Regularization
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        return {
            'lr_stage1': lr_stage1,
            'lr_stage2': lr_stage2,
            'unfreeze_layers': unfreeze_layers,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'epochs_stage1': epochs_stage1,
            'epochs_stage2': epochs_stage2,
            'use_focal_loss': use_focal_loss,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'use_label_smoothing': use_label_smoothing,
            'label_smoothing': label_smoothing,
            'weight_decay': weight_decay,
            'patience': 7,  # Fixed
            'num_folds': 3  # Reduced for faster hyperopt
        }


def train_with_cross_validation_objective(hparams, experiment_name, data_cache=None):
    """Simplified training function for hyperparameter optimization"""
    
    # Import training functions
    from enhanced_training_with_logging import (
        ExperimentTracker, RobustCrossValidator, 
        train_single_fold, aggregate_cv_results
    )
    from enhanced_presto_classifier import (
        EnhancedPrestoClassifier, EnhancedPrestoTrainer, 
        CropDataset, collate_fn, get_optimal_device
    )
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader
    
    # Load data (use cache if available)
    if data_cache is None:
        timeseries_data, labels, coords, sources = load_data_with_config()
    else:
        timeseries_data, labels, coords, sources = data_cache
    
    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Initialize cross-validator (reduced folds for speed)
    cv = RobustCrossValidator(
        n_splits=hparams['num_folds'],
        test_size=0.2,
        random_state=42
    )
    
    # Create splits
    cv_splits, test_data, test_indices = cv.create_splits(
        timeseries_data, labels, sources, coords
    )
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name)
    tracker.log_hyperparameters(hparams)
    
    # Cross-validation training
    fold_results = []
    device = get_optimal_device()
    
    for fold_data in cv_splits:
        fold_idx = fold_data['fold']
        
        # Extract labels from dictionaries and encode for this fold
        train_labels_list = list(fold_data['train']['labels'].values())
        val_labels_list = list(fold_data['val']['labels'].values())
        train_labels_encoded = label_encoder.transform(train_labels_list)
        val_labels_encoded = label_encoder.transform(val_labels_list)
        
        # Create datasets
        train_dataset = CropDataset(
            fold_data['train']['timeseries'],
            train_labels_encoded,
            fold_data['train']['coords']
        )
        
        val_dataset = CropDataset(
            fold_data['val']['timeseries'],
            val_labels_encoded,
            fold_data['val']['coords']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=hparams['batch_size'], 
            shuffle=True, collate_fn=collate_fn, num_workers=1  # Reduced workers
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=hparams['batch_size'], 
            shuffle=False, collate_fn=collate_fn, num_workers=1
        )
        
        # Initialize model
        model = EnhancedPrestoClassifier(
            num_classes=len(label_encoder.classes_),
            freeze_backbone=True,
            unfreeze_layers=hparams['unfreeze_layers']
        )
        
        # Update model dropout if needed
        if hasattr(model.classification_head, 'classifier'):
            for layer in model.classification_head.classifier:
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = hparams['dropout_rate']
        
        # Initialize trainer
        trainer = EnhancedPrestoTrainer(
            model=model, device=device,
            use_focal_loss=hparams['use_focal_loss'],
            use_label_smoothing=hparams['use_label_smoothing']
        )
        
        # Update trainer loss function parameters
        if hparams['use_focal_loss']:
            trainer.criterion.alpha = hparams['focal_alpha']
            trainer.criterion.gamma = hparams['focal_gamma']
        
        # Train fold (simplified)
        fold_result = train_fold_hyperopt(
            trainer, train_loader, val_loader, 
            hparams, tracker, fold_idx
        )
        
        fold_results.append(fold_result)
    
    # Aggregate CV results
    cv_metrics = aggregate_cv_results(fold_results)
    
    # Cleanup
    tracker.finalize_experiment(cv_metrics)
    
    return cv_metrics


def train_fold_hyperopt(trainer, train_loader, val_loader, hparams, tracker, fold_idx):
    """Simplified fold training for hyperparameter optimization"""
    
    # Stage 1: Train classification head
    trainer.setup_stage1_training(lr=hparams['lr_stage1'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(hparams['epochs_stage1']):
        # Train
        train_loss, train_acc = trainer.train_epoch(
            train_loader, trainer.stage1_optimizer, f"HyperOpt_Fold{fold_idx}_Stage1"
        )
        
        # Validate
        val_loss, val_acc, val_log_loss = trainer.validate(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hparams['patience']:
                break
    
    # Stage 2: Fine-tune with unfrozen layers
    trainer.setup_stage2_training(lr=hparams['lr_stage2'])
    
    for epoch in range(hparams['epochs_stage2']):
        # Train
        train_loss, train_acc = trainer.train_epoch(
            train_loader, trainer.stage2_optimizer, f"HyperOpt_Fold{fold_idx}_Stage2"
        )
        
        # Validate
        val_loss, val_acc, val_log_loss = trainer.validate(val_loader)
    
    # Final validation
    final_val_loss, final_val_acc, final_val_log_loss = trainer.validate(val_loader)
    
    return {
        'fold': fold_idx,
        'val_acc': final_val_acc,
        'val_loss': final_val_loss,
        'val_log_loss': final_val_log_loss
    }


def run_hyperparameter_optimization(n_trials: int = 50, study_name: str = None):
    """Run Bayesian hyperparameter optimization"""
    
    print("🔍 HYPERPARAMETER OPTIMIZATION FOR ENHANCED PRESTO")
    print("=" * 80)
    
    # Create study name with timestamp
    if study_name is None:
        study_name = f"presto_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load data once and cache it
    print("📊 Loading data for hyperparameter optimization...")
    data_cache = load_data_with_config()
    print(f"   Loaded {len(data_cache[1])} samples")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',  # We want to minimize log loss
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    )
    
    # Create objective function
    objective = PrestoHyperoptObjective(data_cache=data_cache)
    
    # Run optimization
    print(f"🚀 Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=None)
    
    # Print results
    print("\\n🏆 HYPERPARAMETER OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best log loss: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    config = load_config()
    results_dir = Path(config.PATHS_CONFIG['hyperopt_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save study (using joblib instead of optuna.save_study which doesn't exist)
    import joblib
    study_file = results_dir / f"{study_name}_study.pkl"
    joblib.dump(study, str(study_file))
    
    # Save best parameters
    best_params_file = results_dir / f"{study_name}_best_params.json"
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'study_name': study_name
        }, f, indent=2)
    
    # Save all trials
    trials_file = results_dir / f"{study_name}_trials.json"
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
            'user_attrs': trial.user_attrs
        }
        trials_data.append(trial_data)
    
    with open(trials_file, 'w') as f:
        json.dump(trials_data, f, indent=2)
    
    print(f"\\n💾 Results saved:")
    print(f"   Study: {study_file}")
    print(f"   Best params: {best_params_file}")
    print(f"   All trials: {trials_file}")
    
    return study


def create_optimized_config(study, output_file: str = None):
    """Create an optimized configuration file based on best hyperparameters"""
    
    if output_file is None:
        config = load_config()
        config_dir = Path(config.PATHS_CONFIG['hyperconfig_dir'])
        config_dir.mkdir(parents=True, exist_ok=True)
        output_file = config_dir / f"optimized_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create optimized configuration
    optimized_config = {
        'optimization_info': {
            'study_name': study.study_name,
            'best_trial': study.best_trial.number,
            'best_log_loss': study.best_value,
            'n_trials': len(study.trials),
            'optimization_date': datetime.now().isoformat()
        },
        'model_config': {
            'learning_rate_stage1': study.best_params['lr_stage1'],
            'learning_rate_stage2': study.best_params['lr_stage2'],
            'unfreeze_layers': study.best_params['unfreeze_layers'],
            'dropout_rate': study.best_params['dropout_rate'],
            'batch_size': study.best_params['batch_size'],
            'epochs_stage1': study.best_params['epochs_stage1'],
            'epochs_stage2': study.best_params['epochs_stage2'],
            'use_focal_loss': study.best_params['use_focal_loss'],
            'focal_alpha': study.best_params['focal_alpha'],
            'focal_gamma': study.best_params['focal_gamma'],
            'use_label_smoothing': study.best_params['use_label_smoothing'],
            'label_smoothing': study.best_params['label_smoothing'],
            'weight_decay': study.best_params['weight_decay']
        }
    }
    
    # Save optimized configuration
    with open(output_file, 'w') as f:
        json.dump(optimized_config, f, indent=2)
    
    print(f"📄 Optimized configuration saved: {output_file}")
    
    return optimized_config


def main():
    """Main hyperparameter optimization pipeline"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Enhanced PRESTO')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--study_name', type=str, default=None, help='Study name')
    
    args = parser.parse_args()
    
    # Run hyperparameter optimization
    study = run_hyperparameter_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name
    )
    
    # Create optimized configuration
    optimized_config = create_optimized_config(study)
    
    print("\\n🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!")
    print(f"Use the optimized configuration for final training.")
    
    return study, optimized_config


if __name__ == "__main__":
    study, config = main()