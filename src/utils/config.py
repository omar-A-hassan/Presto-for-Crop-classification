#!/usr/bin/env python3
"""
Configuration for Crop Classification Pipeline
==============================================

Configuration with support for environment variables and relative paths.
Sensitive values can be overridden with environment variables.

Environment Variables:
    GEE_SERVICE_ACCOUNT: Service account email
    GEE_PROJECT_ID: Google Cloud project ID
    GEE_KEY_FILE: Path to service account key file
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Google Earth Engine Configuration
GEE_CONFIG = {
    # Service account email - Can be overridden with GEE_SERVICE_ACCOUNT env var
    'service_account': os.getenv('GEE_SERVICE_ACCOUNT', 'your-service-account@your-project.iam.gserviceaccount.com'),
    
    # Path to service account key file - Can be overridden with GEE_KEY_FILE env var
    'key_file': os.getenv('GEE_KEY_FILE', str(PROJECT_ROOT / 'credentials/service-account-key.json')),
    
    # Project ID for GEE - Can be overridden with GEE_PROJECT_ID env var
    'project_id': os.getenv('GEE_PROJECT_ID', 'your-gcp-project-id'),
    
    # Use service account (True) or user authentication (False)
    'use_service_account': True,
    
    # API endpoints (usually don't need to change)
    'api_url': 'https://earthengine.googleapis.com'
}

# Data Configuration
DATA_CONFIG = {
    # Date range for satellite data
    'start_date': '2020-01-01',
    'end_date': '2021-12-31',
    
    # Spatial configuration
    'buffer_meters': 100,  # Buffer around point locations
    'spatial_resolution': 10,  # Sentinel-2 resolution in meters
    
    # Temporal configuration
    'cloud_threshold': 20,  # Maximum cloud percentage
    'temporal_resolution': 'monthly'  # 'monthly' or 'weekly'
}

# Model Configuration
MODEL_CONFIG = {
    # PRESTO model settings
    'use_presto': True,
    'presto_model_path': 'presto/data/default_model.pt',  # Path to pre-trained PRESTO weights
    
    # Classifier settings
    'ensemble_weights': {
        'pytorch': 0.7,
        'random_forest': 0.3
    },
    
    # Training parameters
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10,  # Early stopping patience
    
    # Cross-validation
    'cv_folds': 5,
    'test_size': 0.2,
    'validation_split': 0.2
}

# Dataset Configuration
DATASET_CONFIG = {
    # Dataset file paths - Using organized data structure
    'cacao_file': str(PROJECT_ROOT / 'data/raw/03 cacao_ucayali_v2.json'),
    'oil_palm_file': str(PROJECT_ROOT / 'data/raw/04 Dataset_Ucayali_Palm_V2.geojson'),
    'rubber_dir': str(PROJECT_ROOT / 'dataset_rubber_planting_and_deforestation'),
    
    # Rubber image subdirectory
    'rubber_images_dir': 'Screenshots_reference_dataset/Sorted_by_MapClass',
    
    # Sample sizes (None for all data)
    'max_samples_per_class': None,
    'min_samples_per_class': 50,
    
    # Data balancing
    'balance_classes': True,
    'use_synthetic_rubber': False,  # We have real rubber images
    
    # Class mappings for rubber images
    'rubber_class_mapping': {
        'Class0_NonForest': 'non_forest',
        'Class1_Forest': 'forest', 
        'Class2_Monoculture_rubber': 'rubber'
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_models': True,
    'save_features': True,
    'save_predictions': True,
    'create_visualizations': True,
    
    # Output directories (using organized structure with absolute paths)
    'model_dir': str(PROJECT_ROOT / 'models/final'),
    'checkpoints_dir': str(PROJECT_ROOT / 'models/checkpoints'),
    'results_dir': str(PROJECT_ROOT / 'results/evaluations'),
    'submissions_dir': str(PROJECT_ROOT / 'results/submissions'),
    'plots_dir': str(PROJECT_ROOT / 'results/plots')
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_file': 'crop_classification.log',
    'console_output': True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'num_workers': 4,  # For data loading
    'pin_memory': True,
    'mixed_precision': False,
    'batch_size_gee': 20,  # Batch size for GEE extraction
    'batch_size_images': 32  # Batch size for image processing
}

# File Paths Configuration
PATHS_CONFIG = {
    # Project root directory
    'project_root': str(PROJECT_ROOT),
    
    # Data directories
    'data_dir': str(PROJECT_ROOT / 'data'),
    'raw_data_dir': str(PROJECT_ROOT / 'data/raw'),
    'processed_data_dir': str(PROJECT_ROOT / 'data/processed'),
    'extracted_data_dir': str(PROJECT_ROOT / 'data/extracted'),
    
    # Model directories
    'model_dir': str(PROJECT_ROOT / 'models'),
    'model_checkpoints_dir': str(PROJECT_ROOT / 'models/checkpoints'),
    'model_final_dir': str(PROJECT_ROOT / 'models/final'),
    'model_experiments_dir': str(PROJECT_ROOT / 'models/experiments'),
    
    # Results directories
    'results_dir': str(PROJECT_ROOT / 'results'),
    'submissions_dir': str(PROJECT_ROOT / 'results/submissions'),
    'evaluations_dir': str(PROJECT_ROOT / 'results/evaluations'),
    'plots_dir': str(PROJECT_ROOT / 'results/plots'),
    
    # Logging directories
    'log_dir': str(PROJECT_ROOT / 'logs'),
    'tensorboard_dir': str(PROJECT_ROOT / 'logs/tensorboard'),
    'training_logs_dir': str(PROJECT_ROOT / 'logs/training'),
    'experiment_logs_dir': str(PROJECT_ROOT / 'logs/experiments'),
    
    # Experiment directories
    'experiments_dir': str(PROJECT_ROOT / 'experiments'),
    'hyperopt_dir': str(PROJECT_ROOT / 'experiments/hyperparameter_optimization'),
    'ablation_dir': str(PROJECT_ROOT / 'experiments/ablation_studies'),
    
    # Configuration directories
    'config_dir': str(PROJECT_ROOT / 'config'),
    'hyperconfig_dir': str(PROJECT_ROOT / 'config/hyperparameter_configs'),
    
    # External dependencies
    'presto_dir': str(PROJECT_ROOT / 'presto'),
    
    # Specific file paths
    'log_file': str(PROJECT_ROOT / 'logs/training/crop_classification.log')
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    # Pipeline mode: 'hybrid', 'gee_only', 'images_only'
    'mode': 'hybrid',
    
    # Feature extraction method
    'feature_extractor': 'presto',  # 'presto' or 'cnn'
    
    # Image processing settings
    'image_size': (128, 128),  # Resize images to this size
    'normalize_images': True,
    
    # Validation settings
    'random_seed': 42,
    'shuffle_data': True,
    'stratify_splits': True
}