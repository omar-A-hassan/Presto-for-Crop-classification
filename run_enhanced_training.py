#!/usr/bin/env python3
"""
Enhanced PRESTO Training Launcher
================================

This script launches the enhanced PRESTO training pipeline with:
- Pre-trained PRESTO weights
- Two-stage fine-tuning strategy  
- Geographic stratification
- Advanced loss functions
- Comprehensive evaluation

Usage:
    python run_enhanced_training.py
"""

import sys
import os
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src" / "training"))
sys.path.append(str(project_root / "src" / "models"))
sys.path.append(str(project_root / "src" / "data"))
sys.path.append(str(project_root / "src" / "utils"))
sys.path.append(str(project_root / "presto"))
sys.path.append(str(project_root / "presto" / "presto"))

def main():
    """Launch enhanced training pipeline"""
    print("🚀 LAUNCHING ENHANCED PRESTO CROP CLASSIFICATION")
    print("=" * 80)
    
    # Set up environment for PRESTO
    presto_path = str(project_root / "presto")
    if presto_path not in os.environ.get('PYTHONPATH', ''):
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{presto_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = presto_path
    
    print("📋 Training Configuration:")
    print("   • Pre-trained PRESTO foundation model")
    print("   • Two-stage fine-tuning strategy")
    print("   • Geographic stratification for robust validation")
    print("   • Focal loss + label smoothing for calibration")
    print("   • Attention pooling for variable timesteps")
    print("   • Comprehensive evaluation with log loss optimization")
    print()
    
    try:
        # Import and run training
        from train_enhanced_presto import main as train_main
        
        print("🎯 Starting enhanced training pipeline...")
        result = train_main()
        
        if result == 0:
            print("\n🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            print("\n📊 Key Improvements Implemented:")
            print("   ✅ Pre-trained PRESTO weights loaded")
            print("   ✅ Two-stage fine-tuning strategy")
            print("   ✅ Geographic cross-validation")
            print("   ✅ Advanced loss functions for calibration")
            print("   ✅ Attention pooling for temporal aggregation")
            print("   ✅ Comprehensive evaluation metrics")
            print("\n📁 Results saved in 'results/' directory")
            print("   • enhanced_presto_crop_classifier.pth - Trained model")
            print("   • evaluation_results.json - Test metrics")
            print("   • training_history.json - Training curves")
            
        else:
            print("\n❌ Training failed. Check logs for details.")
            
    except Exception as e:
        print(f"\n❌ Failed to launch training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return result

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)