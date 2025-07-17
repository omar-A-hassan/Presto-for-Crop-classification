#!/usr/bin/env python3
"""
Robust Data Extractor with Progress Saving and Error Recovery
============================================================

This script adds resilience to the satellite data extraction process:
- Saves progress after each batch
- Handles timeouts gracefully  
- Resumes from where it left off
- Skips problematic batches
- Provides detailed error reporting
"""

import json
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time
import signal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the GEE extractor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gee_sentinel2_extractor import GEESentinel2Extractor

class RobustDataExtractor:
    """Robust satellite data extractor with progress saving"""
    
    def __init__(self, output_dir=None):
        # Use config to determine output directory if not specified
        if output_dir is None:
            import sys
            # Add project utils to path using relative path
            project_root = Path(__file__).parent.parent.parent
            sys.path.append(str(project_root / "src" / "utils"))
            from config_loader import load_config
            config = load_config()
            output_dir = config.PATHS_CONFIG['extracted_data_dir']
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "extraction_progress.json"
        self.data_file = self.output_dir / "extracted_data.pkl"
        self.failed_batches_file = self.output_dir / "failed_batches.json"
        
        # Configuration
        self.batch_timeout = 300  # 5 minutes per batch
        self.max_retries = 2
        self.batch_size = 10
        
        # Initialize extractor
        self.extractor = GEESentinel2Extractor("2020-01-01", "2021-12-31")
        
        # Data storage
        self.all_timeseries = {}
        self.all_labels = []
        self.all_coords = {}
        self.all_sources = []
        self.failed_batches = []
        
        # Load existing progress
        self.load_progress()
    
    def save_progress(self, dataset_name, batch_idx, total_batches):
        """Save current progress"""
        progress = {
            'dataset_name': dataset_name,
            'batch_idx': batch_idx,
            'total_batches': total_batches,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.all_labels),
            'failed_batches': len(self.failed_batches)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Save data
        data = {
            'all_timeseries': self.all_timeseries,
            'all_labels': self.all_labels,
            'all_coords': self.all_coords,
            'all_sources': self.all_sources
        }
        
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Save failed batches
        with open(self.failed_batches_file, 'w') as f:
            json.dump(self.failed_batches, f, indent=2)
        
        print(f"💾 Progress saved: {dataset_name} batch {batch_idx}/{total_batches}")
    
    def load_progress(self):
        """Load existing progress if available"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
            print(f"📂 Found existing progress: {progress['dataset_name']} at batch {progress['batch_idx']}/{progress['total_batches']}")
            
            # Load data
            if self.data_file.exists():
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                self.all_timeseries = data['all_timeseries']
                self.all_labels = data['all_labels']
                self.all_coords = data['all_coords']
                self.all_sources = data['all_sources']
                print(f"📊 Loaded {len(self.all_labels)} existing samples")
            
            # Load failed batches
            if self.failed_batches_file.exists():
                with open(self.failed_batches_file, 'r') as f:
                    self.failed_batches = json.load(f)
                print(f"⚠️  Found {len(self.failed_batches)} previously failed batches")
            
            return progress
        return None
    
    def timeout_handler(self, signum, frame):
        """Handle batch timeout"""
        raise TimeoutError("Batch processing timeout")
    
    def extract_batch_with_timeout(self, coord_df, batch_start, batch_end, crop_type, source_name):
        """Extract batch with timeout protection"""
        # Reset index to ensure 0-based indexing for the batch
        batch_df = coord_df.iloc[batch_start:batch_end].reset_index(drop=True)
        
        # Set up timeout
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.batch_timeout)
        
        try:
            print(f"   🔄 Processing locations {batch_start}-{batch_end} ({len(batch_df)} samples)")
            timeseries_data = self.extractor.extract_batch_timeseries(batch_df, batch_size=len(batch_df))
            signal.alarm(0)  # Cancel timeout
            
            # Store successful extractions
            sample_idx = len(self.all_labels)
            successful_count = 0
            
            for batch_idx, ts_data in timeseries_data.items():
                # Skip None results (failed extractions) - no synthetic data fallback
                if ts_data is not None and len(ts_data) > 0:
                    self.all_timeseries[sample_idx] = ts_data
                    self.all_labels.append(crop_type)
                    
                    # Get coordinates from the batch (batch_idx should now be 0-based)
                    row = batch_df.iloc[batch_idx]
                    self.all_coords[sample_idx] = (row['lat'], row['lon'])
                    self.all_sources.append(source_name)
                    
                    sample_idx += 1
                    successful_count += 1
                else:
                    print(f"   ⚠️  Skipping sample {batch_idx} - no valid data (no synthetic fallback)")
            
            print(f"   ✅ Successfully extracted {successful_count}/{len(batch_df)} samples")
            return True, successful_count
            
        except TimeoutError:
            signal.alarm(0)
            print(f"   ⏰ Batch {batch_start}-{batch_end} timed out after {self.batch_timeout}s")
            return False, 0
        except Exception as e:
            signal.alarm(0)
            print(f"   ❌ Batch {batch_start}-{batch_end} failed: {e}")
            return False, 0
    
    def extract_coordinate_dataset(self, file_path, crop_type, source_name, resume_batch=0):
        """Extract satellite data for coordinate dataset with progress saving"""
        print(f"\n🗺️  EXTRACTING {crop_type.upper()} DATASET")
        print(f"📄 File: {file_path}")
        
        try:
            # Load dataset
            gdf = gpd.read_file(file_path)
            print(f"📍 Loaded {len(gdf)} features")
            
            # Extract coordinates
            if 'geometry' in gdf.columns:
                centroids = gdf.geometry.centroid
                coord_df = pd.DataFrame({
                    'lat': centroids.y,
                    'lon': centroids.x,
                    'crop_type': crop_type,
                    'source': source_name
                })
            else:
                print(f"❌ No geometry column found")
                return 0
            
            # Calculate batches
            total_samples = len(coord_df)
            total_batches = (total_samples + self.batch_size - 1) // self.batch_size
            
            print(f"🔢 Processing {total_samples} samples in {total_batches} batches")
            
            successful_extractions = 0
            
            # Process batches
            for batch_idx in range(resume_batch, total_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, total_samples)
                
                print(f"\n📦 Batch {batch_idx + 1}/{total_batches}")
                
                # Try extraction with retries
                success = False
                for retry in range(self.max_retries):
                    if retry > 0:
                        print(f"   🔄 Retry {retry}/{self.max_retries}")
                        time.sleep(30)  # Wait before retry
                    
                    success, batch_count = self.extract_batch_with_timeout(
                        coord_df, batch_start, batch_end, crop_type, source_name
                    )
                    
                    if success:
                        successful_extractions += batch_count
                        break
                
                if not success:
                    # Record failed batch
                    failed_batch = {
                        'dataset': crop_type,
                        'batch_idx': batch_idx,
                        'batch_start': batch_start,
                        'batch_end': batch_end,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.failed_batches.append(failed_batch)
                    print(f"   💀 Batch {batch_idx + 1} failed permanently - skipping")
                
                # Save progress after each batch
                self.save_progress(crop_type, batch_idx + 1, total_batches)
            
            print(f"\n✅ {crop_type.upper()} EXTRACTION COMPLETE")
            print(f"   📊 Successfully extracted: {successful_extractions}/{total_samples}")
            print(f"   ❌ Failed batches: {len([b for b in self.failed_batches if b['dataset'] == crop_type])}")
            
            return successful_extractions
            
        except Exception as e:
            print(f"❌ Error processing {crop_type} dataset: {e}")
            return 0
    
    def extract_all_datasets(self):
        """Extract all coordinate-based datasets"""
        print("🚀 ROBUST SATELLITE DATA EXTRACTION")
        print("=" * 50)
        
        # Check for existing progress
        progress = self.load_progress()
        
        # Check existing extraction report to skip completed sources
        completed_sources = self.get_completed_sources()
        
        datasets = [
            ("data/raw/03 cacao_ucayali_v2.json", "cacao", "peru_amazon"),
            ("data/raw/04 Dataset_Ucayali_Palm_V2.geojson", "oil_palm", "peru_ucayali"),
            ("data/raw/rubber_grid_samples.geojson", "rubber", "china_xishuangbanna"),
            ("data/raw/ghana_cocoa_samples.geojson", "cacao", "ghana_cocoa")
        ]
        
        total_extracted = 0
        
        for file_path, crop_type, source_name in datasets:
            if not Path(file_path).exists():
                print(f"⚠️  File not found: {file_path}")
                continue
            
            # Skip if source is already completely extracted
            if source_name in completed_sources:
                print(f"✅ {source_name.upper()} already extracted ({completed_sources[source_name]} samples) - skipping")
                continue
            
            # Determine resume point
            resume_batch = 0
            if progress and progress['dataset_name'] == crop_type:
                resume_batch = progress['batch_idx']
                print(f"📂 Resuming {crop_type} from batch {resume_batch}")
            
            extracted = self.extract_coordinate_dataset(file_path, crop_type, source_name, resume_batch)
            total_extracted += extracted
        
        print(f"\n🎯 EXTRACTION SUMMARY")
        print("=" * 30)
        print(f"Total samples extracted: {total_extracted}")
        print(f"Total failed batches: {len(self.failed_batches)}")
        
        # Generate final report
        self.generate_extraction_report()
        
        return self.all_timeseries, self.all_labels, self.all_coords, self.all_sources
    
    def generate_extraction_report(self):
        """Generate detailed extraction report"""
        report = {
            'summary': {
                'total_samples': len(self.all_labels),
                'total_failed_batches': len(self.failed_batches),
                'extraction_date': datetime.now().isoformat()
            },
            'by_crop': {},
            'by_source': {},
            'failed_batches': self.failed_batches
        }
        
        # Count by crop type
        for label in set(self.all_labels):
            count = self.all_labels.count(label)
            failed = len([b for b in self.failed_batches if b['dataset'] == label])
            report['by_crop'][label] = {
                'successful': count,
                'failed_batches': failed
            }
        
        # Count by source
        for source in set(self.all_sources):
            count = self.all_sources.count(source)
            # Map source to crop type for failed batch counting
            source_crop_map = {
                'peru_amazon': 'cacao',
                'peru_ucayali': 'oil_palm',
                'china_xishuangbanna': 'rubber',
                'ghana_cocoa': 'cacao'
            }
            crop_type = source_crop_map.get(source, source)
            failed = len([b for b in self.failed_batches if b['dataset'] == crop_type])
            report['by_source'][source] = {
                'successful': count,
                'failed_batches': failed,
                'crop_type': crop_type
            }
        
        # Save report
        report_file = self.output_dir / "extraction_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Detailed report saved to: {report_file}")
    
    def load_existing_data(self):
        """Load existing extracted data without re-extraction"""
        if self.data_file.exists():
            print("📂 Loading existing extracted data...")
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            
            self.all_timeseries = data.get('all_timeseries', {})
            self.all_labels = data.get('all_labels', [])
            self.all_coords = data.get('all_coords', {})
            self.all_sources = data.get('all_sources', [])
            
            print(f"✅ Loaded {len(self.all_labels)} existing samples")
            return True
        else:
            print("⚠️  No existing extracted data found")
            return False
    
    def has_complete_extraction(self):
        """Check if extraction is complete"""
        if not self.progress_file.exists():
            return False
        
        with open(self.progress_file, 'r') as f:
            progress = json.load(f)
        
        # Check if extraction finished successfully
        return progress.get('batch_idx', 0) == progress.get('total_batches', 0)
    
    def get_completed_sources(self):
        """Get list of source datasets that have been completely extracted"""
        completed_sources = {}
        
        # Check extraction report
        report_file = self.output_dir / "extraction_report.json"
        if report_file.exists():
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                # Get completed sources from report (new format)
                if 'by_source' in report:
                    for source, data in report['by_source'].items():
                        if data.get('successful', 0) > 0:
                            completed_sources[source] = data['successful']
                else:
                    # Legacy format - convert crops to sources for backward compatibility
                    legacy_source_map = {
                        'cacao': 'peru_amazon',
                        'oil_palm': 'peru_ucayali', 
                        'rubber': 'china_xishuangbanna'
                    }
                    for crop, data in report.get('by_crop', {}).items():
                        if data.get('successful', 0) > 0 and crop in legacy_source_map:
                            completed_sources[legacy_source_map[crop]] = data['successful']
                
                print(f"📋 Found extraction report with {len(completed_sources)} completed sources:")
                for source, count in completed_sources.items():
                    print(f"   ✅ {source}: {count} samples")
                        
            except Exception as e:
                print(f"⚠️  Error reading extraction report: {e}")
        
        return completed_sources
    
    def clean_progress(self):
        """Clean all progress files to start fresh"""
        for file in [self.progress_file, self.data_file, self.failed_batches_file]:
            if file.exists():
                file.unlink()
        print("🧹 Progress files cleaned - starting fresh")

def main():
    """Main extraction function"""
    extractor = RobustDataExtractor()
    
    # Option to clean and start fresh
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        extractor.clean_progress()
    
    # Extract all datasets
    timeseries, labels, coords, sources = extractor.extract_all_datasets()
    
    print(f"\n✅ EXTRACTION COMPLETE!")
    print(f"Ready for training with {len(labels)} samples")

if __name__ == "__main__":
    main()