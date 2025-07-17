#!/usr/bin/env python3
"""
Enhanced PRESTO-based Crop Classification with Strategic Fine-tuning
==================================================================

This module implements a sophisticated fine-tuning approach for PRESTO foundation model
tailored for robust 3-class crop classification (rubber, oil palm, cacao).

Key Features:
- Pre-trained PRESTO weights loading
- Two-stage fine-tuning strategy
- Attention pooling for variable timesteps
- Geographic stratification
- Label smoothing and focal loss
- Ensemble predictions
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "presto"))
sys.path.append(str(project_root / "presto" / "presto"))  # Add presto subpackage path
sys.path.append(str(project_root / "src" / "utils"))

# Import PRESTO and configuration
PRESTO_AVAILABLE = False

# Try different PRESTO import strategies
try:
    # First try the installed PRESTO package
    import presto
    from presto import Presto
    PRESTO_AVAILABLE = True
    # Only print once per Python session
    if not hasattr(presto, '_presto_loaded_message_shown'):
        print("✅ Official PRESTO package loaded successfully")
        presto._presto_loaded_message_shown = True
except ImportError:
    try:
        # Try single file PRESTO
        from single_file_presto import Presto
        PRESTO_AVAILABLE = True
        print("✅ Single-file PRESTO loaded successfully")
    except ImportError:
        print("❌ PRESTO not available")
        PRESTO_AVAILABLE = False

from config_loader import load_config

warnings.filterwarnings('ignore')


def get_optimal_device():
    """Get the best available device: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_device_info(device):
    """Print device information"""
    if device.type == 'mps':
        print(f"🚀 Using Metal Performance Shaders (MPS) on Apple Silicon")
    elif device.type == 'cuda':
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        print(f"💻 Using CPU")
    print(f"   Device: {device}")


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples and class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better calibration"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AttentionPooling(nn.Module):
    """Attention mechanism for pooling variable timesteps"""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        weights = self.attention(x)  # [batch_size, seq_len, 1]
        weighted_features = torch.sum(x * weights, dim=1)  # [batch_size, feature_dim]
        return weighted_features


class CropClassificationHead(nn.Module):
    """Enhanced classification head with attention pooling"""
    
    def __init__(self, input_dim, num_classes=3, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # Attention pooling for temporal features
        self.attention_pooling = AttentionPooling(input_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Handle both single and multiple timesteps
        if len(x.shape) == 3:  # [batch, timesteps, features]
            x = self.attention_pooling(x)
        
        return self.classifier(x)


class EnhancedPrestoClassifier(nn.Module):
    """Enhanced PRESTO model with fine-tuning capabilities"""
    
    def __init__(self, num_classes=3, freeze_backbone=True, unfreeze_layers=2, load_pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers
        
        # Load pre-trained PRESTO model (skip during inference loading)
        if load_pretrained:
            self.load_presto_model()
        else:
            # Create empty PRESTO model for loading trained weights
            if not PRESTO_AVAILABLE:
                raise ImportError("PRESTO not available. Cannot create model.")
            self.presto_model = Presto.construct()
            print("🔄 Created empty PRESTO model for loading trained weights")
        
        # Add classification head
        encoder_dim = 128  # PRESTO encoder output dimension
        self.classification_head = CropClassificationHead(encoder_dim, num_classes)
        
        # Configure fine-tuning
        if load_pretrained:
            self.configure_fine_tuning()
    
    def load_presto_model(self):
        """Load pre-trained PRESTO model with proper weights"""
        if not PRESTO_AVAILABLE:
            raise ImportError("PRESTO not available. Cannot load pre-trained model.")
        
        try:
            # First try to load with official pre-trained weights
            config = load_config()
            presto_model_path = config.MODEL_CONFIG.get('presto_model_path')
            
            if presto_model_path and Path(presto_model_path).exists():
                print(f"Loading PRESTO weights from: {presto_model_path}")
                self.presto_model = Presto.construct()
                self.presto_model.load_state_dict(torch.load(presto_model_path, map_location='cpu'))
            else:
                # Try to load from default location
                default_weights_path = project_root / "presto" / "data" / "default_model.pt"
                if default_weights_path.exists():
                    print(f"Loading PRESTO weights from default location: {default_weights_path}")
                    self.presto_model = Presto.construct()
                    self.presto_model.load_state_dict(torch.load(default_weights_path, map_location='cpu'))
                else:
                    # Try using load_pretrained if available
                    if hasattr(Presto, 'load_pretrained'):
                        print("Loading PRESTO pre-trained weights using load_pretrained()")
                        self.presto_model = Presto.load_pretrained()
                    else:
                        print("⚠️  No pre-trained weights found, using randomly initialized PRESTO")
                        self.presto_model = Presto.construct()
            
            print("✅ PRESTO model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading PRESTO model: {e}")
            print("   Using randomly initialized PRESTO model")
            self.presto_model = Presto.construct()
    
    def configure_fine_tuning(self):
        """Configure fine-tuning strategy"""
        if self.freeze_backbone:
            # Stage 1: Freeze entire PRESTO backbone
            for param in self.presto_model.parameters():
                param.requires_grad = False
            print(f"🔒 PRESTO backbone frozen for Stage 1 training")
        else:
            # Stage 2: Selective unfreezing of last layers
            self._selective_unfreeze()
    
    def _selective_unfreeze(self):
        """Selectively unfreeze last N transformer layers"""
        # Freeze all parameters first
        for param in self.presto_model.parameters():
            param.requires_grad = False
        
        # Unfreeze encoder layers (last N blocks)
        if hasattr(self.presto_model, 'encoder') and hasattr(self.presto_model.encoder, 'blocks'):
            total_blocks = len(self.presto_model.encoder.blocks)
            unfreeze_start = max(0, total_blocks - self.unfreeze_layers)
            
            for i in range(unfreeze_start, total_blocks):
                for param in self.presto_model.encoder.blocks[i].parameters():
                    param.requires_grad = True
            
            # Also unfreeze final norm layer
            if hasattr(self.presto_model.encoder, 'norm'):
                for param in self.presto_model.encoder.norm.parameters():
                    param.requires_grad = True
            
            print(f"🔓 Unfroze last {self.unfreeze_layers} PRESTO encoder layers for Stage 2")
    
    def enable_stage2_fine_tuning(self):
        """Enable Stage 2 fine-tuning with selective unfreezing"""
        self.freeze_backbone = False
        self._selective_unfreeze()
    
    def forward(self, x, dynamic_world=None, latlons=None, mask=None, month=None):
        """Forward pass through PRESTO + classification head"""
        # Extract features using PRESTO encoder
        if hasattr(self.presto_model, 'encoder'):
            features = self.presto_model.encoder(
                x, 
                dynamic_world=dynamic_world, 
                latlons=latlons, 
                mask=mask, 
                month=month
            )
        else:
            # Fallback for simple models
            features = self.presto_model(x)
        
        # Classification
        logits = self.classification_head(features)
        return logits


class CropDataset(Dataset):
    """Dataset class for crop classification with PRESTO preprocessing"""
    
    def __init__(self, timeseries_data, labels, crop_coords=None):
        self.timeseries_data = timeseries_data
        self.labels = labels
        self.crop_coords = crop_coords
        self.indices = list(timeseries_data.keys())
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        
        # Get time series data
        ts_data = self.timeseries_data[data_idx]
        
        # Convert to tensor and handle dimensions
        if isinstance(ts_data, np.ndarray):
            x = torch.from_numpy(ts_data).float()
        else:
            x = ts_data
        
        # Ensure proper shape [timesteps, bands]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add timestep dimension
        
        # Get label
        label = self.labels[data_idx] if isinstance(self.labels, dict) else self.labels[idx]
        
        # Create mock dynamic world and coordinates if not provided
        timesteps = x.shape[0]
        dynamic_world = torch.full((timesteps,), 9, dtype=torch.long)  # Unknown class
        
        if self.crop_coords and data_idx in self.crop_coords:
            lat, lon = self.crop_coords[data_idx]
            latlons = torch.tensor([lat, lon], dtype=torch.float)
        else:
            latlons = torch.zeros(2, dtype=torch.float)
        
        # Month (can be improved with actual dates)
        month = torch.tensor(6, dtype=torch.long)  # Default to June
        
        return {
            'x': x,
            'label': torch.tensor(label, dtype=torch.long),
            'dynamic_world': dynamic_world,
            'latlons': latlons,
            'month': month
        }


class EnhancedPrestoTrainer:
    """Trainer for enhanced PRESTO crop classification"""
    
    def __init__(self, model, device='auto', use_focal_loss=True, use_label_smoothing=True):
        # Auto-detect optimal device if not specified
        if device == 'auto':
            device = get_optimal_device()
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.model = model.to(device)
        
        # Print device info
        print_device_info(device)
        
        # Loss functions
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif use_label_smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizers for two-stage training
        self.stage1_optimizer = None
        self.stage2_optimizer = None
        
        print(f"✅ Enhanced PRESTO trainer initialized on {device}")
    
    def setup_stage1_training(self, lr=1e-3):
        """Setup Stage 1: Train only classification head"""
        # Only optimize classification head parameters
        trainable_params = list(self.model.classification_head.parameters())
        self.stage1_optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"🎯 Stage 1 optimizer setup - training {len(trainable_params)} parameter groups")
    
    def setup_stage2_training(self, lr=1e-5):
        """Setup Stage 2: Fine-tune unfrozen PRESTO layers + head"""
        # Enable stage 2 fine-tuning
        self.model.enable_stage2_fine_tuning()
        
        # Optimize unfrozen PRESTO parameters + classification head
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.stage2_optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        print(f"🎯 Stage 2 optimizer setup - training {len(trainable_params)} parameters")
    
    def train_epoch(self, dataloader, optimizer, stage="Stage 1"):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            x = batch['x'].to(self.device)
            labels = batch['label'].to(self.device)
            dynamic_world = batch['dynamic_world'].to(self.device)
            latlons = batch['latlons'].to(self.device)
            month = batch['month'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(x, dynamic_world, latlons, month=month)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"   {stage} Batch {batch_idx}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                labels = batch['label'].to(self.device)
                dynamic_world = batch['dynamic_world'].to(self.device)
                latlons = batch['latlons'].to(self.device)
                month = batch['month'].to(self.device)
                
                logits = self.model(x, dynamic_world, latlons, month=month)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for log loss calculation
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        val_loss = total_loss / len(dataloader)
        val_acc = 100. * correct / total
        
        # Calculate log loss
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        from sklearn.metrics import log_loss
        # Handle single-class case for log_loss
        try:
            unique_labels = np.unique(all_labels)
            if len(unique_labels) > 1:
                val_log_loss = log_loss(all_labels, all_probs)
            else:
                # For single class, use cross-entropy manually
                # Avoid log(0) by clipping probabilities
                probs_clipped = np.clip(all_probs[:, unique_labels[0]], 1e-15, 1 - 1e-15)
                val_log_loss = -np.mean(np.log(probs_clipped))
        except Exception as e:
            print(f"Warning: Could not compute log_loss: {e}")
            val_log_loss = float('inf')
        
        return val_loss, val_acc, val_log_loss


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    # Find max sequence length
    max_seq_len = max([item['x'].shape[0] for item in batch])
    batch_size = len(batch)
    feature_dim = batch[0]['x'].shape[-1]
    
    # Initialize tensors
    x_padded = torch.zeros(batch_size, max_seq_len, feature_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)
    dynamic_world = torch.full((batch_size, max_seq_len), 9, dtype=torch.long)
    latlons = torch.zeros(batch_size, 2)
    months = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        seq_len = item['x'].shape[0]
        x_padded[i, :seq_len] = item['x']
        labels[i] = item['label']
        dynamic_world[i, :seq_len] = item['dynamic_world'][:seq_len]
        latlons[i] = item['latlons']
        months[i] = item['month']
    
    return {
        'x': x_padded,
        'label': labels,
        'dynamic_world': dynamic_world,
        'latlons': latlons,
        'month': months
    }


def create_geographic_splits(sources, labels=None, test_size=0.2, val_size=0.1, random_state=42):
    """Create stratified splits ensuring all classes in each split"""
    from sklearn.model_selection import train_test_split
    
    # Check if this is a multi-crop scenario where sources map to specific crops
    unique_sources = list(set(sources))
    
    # If we have labels, check if geographic split would cause class imbalance
    if labels is not None:
        source_to_labels = {}
        for i, source in enumerate(sources):
            if source not in source_to_labels:
                source_to_labels[source] = set()
            source_to_labels[source].add(labels[i])
        
        # Check if any source contains only one class
        single_class_sources = [s for s, lbls in source_to_labels.items() if len(lbls) == 1]
        
        if len(single_class_sources) == len(unique_sources):
            print("⚠️  WARNING: Each geographic source contains only one crop type!")
            print("   This would create train/val/test sets with different classes.")
            print("   Using stratified random split instead to ensure class balance.")
            
            # Use stratified random split to ensure all classes in each split
            indices = list(range(len(sources)))
            train_indices, test_indices = train_test_split(
                indices, test_size=test_size, stratify=labels, random_state=random_state
            )
            
            if val_size > 0:
                train_labels = [labels[i] for i in train_indices]
                train_indices, val_indices = train_test_split(
                    train_indices, test_size=val_size/(1-test_size), 
                    stratify=train_labels, random_state=random_state
                )
            else:
                val_indices = []
            
            train_mask = [i in train_indices for i in range(len(sources))]
            val_mask = [i in val_indices for i in range(len(sources))] if val_indices else [False] * len(sources)
            test_mask = [i in test_indices for i in range(len(sources))]
            
            # Report the stratified split
            if labels:
                from collections import Counter
                train_classes = Counter([labels[i] for i in train_indices])
                val_classes = Counter([labels[i] for i in val_indices]) if val_indices else {}
                test_classes = Counter([labels[i] for i in test_indices])
                
                print(f"Stratified split: Train classes={dict(train_classes)}, Val classes={dict(val_classes)}, Test classes={dict(test_classes)}")
            
            return train_mask, val_mask, test_mask
    
    # Original geographic split logic for cases where it makes sense
    if len(unique_sources) > 1:
        # Split by geographic regions
        train_sources, test_sources = train_test_split(
            unique_sources, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            train_sources, val_sources = train_test_split(
                train_sources, test_size=val_size/(1-test_size), random_state=random_state
            )
        else:
            val_sources = []
        
        # Create masks
        train_mask = [s in train_sources for s in sources]
        val_mask = [s in val_sources for s in sources] if val_sources else [False] * len(sources)
        test_mask = [s in test_sources for s in sources]
        
        print(f"Geographic split: Train sources={train_sources}, Val sources={val_sources}, Test sources={test_sources}")
        
    else:
        # Fall back to random split if only one source
        print("Single source detected, using random split")
        indices = list(range(len(sources)))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            train_indices, val_indices = train_test_split(
                train_indices, test_size=val_size/(1-test_size), random_state=random_state
            )
        else:
            val_indices = []
        
        train_mask = [i in train_indices for i in range(len(sources))]
        val_mask = [i in val_indices for i in range(len(sources))] if val_indices else [False] * len(sources)
        test_mask = [i in test_indices for i in range(len(sources))]
    
    return train_mask, val_mask, test_mask