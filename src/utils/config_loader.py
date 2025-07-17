#!/usr/bin/env python3
"""
Configuration Loader for Crop Classification Pipeline
====================================================

Loads configuration from config.py or falls back to config_template.py
"""

import os
import sys
from pathlib import Path

def load_config():
    """Load configuration from config.py or template"""
    
    # Try to load user config first - check both current dir and src/utils
    config_file = Path('config.py')
    if not config_file.exists():
        config_file = Path(__file__).parent / 'config.py'
    
    template_file = Path('config_template.py')
    if not template_file.exists():
        template_file = Path(__file__).parent / 'config_template.py'
    
    if config_file.exists():
        print("📄 Loading configuration from config.py")
        try:
            # Add current directory to path for import
            sys.path.insert(0, str(Path.cwd()))
            import config
            return config
        except Exception as e:
            print(f"❌ Error loading config.py: {e}")
            print("📄 Falling back to template configuration...")
    
    # Fall back to template
    if template_file.exists():
        print("📄 Loading configuration from config_template.py")
        print("⚠️  Consider copying config_template.py to config.py and customizing it")
        try:
            import config_template as config
            return config
        except Exception as e:
            print(f"❌ Error loading config_template.py: {e}")
            raise
    
    raise FileNotFoundError("No configuration file found. Please ensure config.py exists in src/utils/ directory.")

def get_config_value(config_dict, key, default=None):
    """Safely get a configuration value with default fallback"""
    return config_dict.get(key, default)

def validate_config(config):
    """Validate configuration and provide helpful error messages"""
    print("🔍 Validating configuration...")
    
    issues = []
    
    # Check GEE configuration
    if hasattr(config, 'GEE_CONFIG'):
        gee_config = config.GEE_CONFIG
        
        # Check service account
        if gee_config.get('use_service_account', False):
            key_file = gee_config.get('key_file')
            if not key_file or not os.path.exists(key_file):
                issues.append(f"❌ GEE service account key file not found: {key_file}")
            else:
                print(f"✅ GEE service account key found: {key_file}")
        
        # Check required fields
        required_gee_fields = ['service_account', 'project_id']
        for field in required_gee_fields:
            if not gee_config.get(field):
                issues.append(f"❌ Missing required GEE config field: {field}")
    else:
        issues.append("❌ GEE_CONFIG not found in configuration")
    
    # Check dataset configuration
    if hasattr(config, 'DATASET_CONFIG'):
        dataset_config = config.DATASET_CONFIG
        
        # Check dataset files
        cacao_file = dataset_config.get('cacao_file')
        palm_file = dataset_config.get('oil_palm_file')
        rubber_dir = dataset_config.get('rubber_dir')
        
        if cacao_file and not os.path.exists(cacao_file):
            issues.append(f"⚠️  Cacao dataset file not found: {cacao_file}")
        else:
            print(f"✅ Cacao dataset found: {cacao_file}")
            
        if palm_file and not os.path.exists(palm_file):
            issues.append(f"⚠️  Oil palm dataset file not found: {palm_file}")
        else:
            print(f"✅ Oil palm dataset found: {palm_file}")
            
        if rubber_dir and not os.path.exists(rubber_dir):
            issues.append(f"⚠️  Rubber dataset directory not found: {rubber_dir}")
        else:
            print(f"✅ Rubber dataset directory found: {rubber_dir}")
    
    # Check paths configuration
    if hasattr(config, 'PATHS_CONFIG'):
        paths_config = config.PATHS_CONFIG
        
        # Create output directories if they don't exist
        output_dirs = ['output_dir', 'model_dir', 'results_dir', 'plots_dir', 'temp_dir', 'log_dir']
        for dir_key in output_dirs:
            dir_path = paths_config.get(dir_key)
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"✅ Created/verified directory: {dir_path}")
    
    # Report issues
    if issues:
        print(f"\n⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   {issue}")
        
        if any("❌" in issue for issue in issues):
            print(f"\n🔧 To fix critical issues:")
            print(f"   1. Copy config_template.py to config.py")
            print(f"   2. Update the settings in config.py")
            print(f"   3. Ensure all required files exist")
            return False
    else:
        print("✅ Configuration validation passed!")
    
    return True

def create_user_config():
    """Create a user config file from template"""
    template_file = Path('config_template.py')
    config_file = Path('config.py')
    
    if not template_file.exists():
        print("❌ config_template.py not found")
        return False
    
    if config_file.exists():
        response = input(f"config.py already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return False
    
    # Copy template to config
    import shutil
    shutil.copy2(template_file, config_file)
    
    print(f"✅ Created config.py from template")
    print(f"📝 Please edit config.py with your specific settings")
    
    return True

def main():
    """Test configuration loading"""
    print("🧪 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        config = load_config()
        
        if validate_config(config):
            print(f"\n✅ Configuration loaded successfully!")
            
            # Show some key settings
            print(f"\n📊 Key Settings:")
            if hasattr(config, 'GEE_CONFIG'):
                print(f"   GEE Service Account: {config.GEE_CONFIG.get('service_account', 'Not set')}")
                print(f"   GEE Project: {config.GEE_CONFIG.get('project_id', 'Not set')}")
            
            if hasattr(config, 'DATASET_CONFIG'):
                print(f"   Cacao file: {config.DATASET_CONFIG.get('cacao_file', 'Not set')}")
                print(f"   Oil palm file: {config.DATASET_CONFIG.get('oil_palm_file', 'Not set')}")
                print(f"   Rubber dir: {config.DATASET_CONFIG.get('rubber_dir', 'Not set')}")
            
            if hasattr(config, 'PIPELINE_CONFIG'):
                print(f"   Pipeline mode: {config.PIPELINE_CONFIG.get('mode', 'Not set')}")
                print(f"   Feature extractor: {config.PIPELINE_CONFIG.get('feature_extractor', 'Not set')}")
        else:
            print(f"\n❌ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print(f"\n🔧 To create a configuration file:")
        print(f"   python config_loader.py --create")
    
    sys.exit(0 if success else 1)