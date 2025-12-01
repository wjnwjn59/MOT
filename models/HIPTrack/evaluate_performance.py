import os
import sys
import torch
import numpy as np
import pickle
import shutil
import tempfile
from pathlib import Path

# Add the lib directory to Python path
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if lib_path not in sys.path:
    sys.path.append(lib_path)

# Import framework modules
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation.datasets import get_dataset
from lib.test.evaluation.tracker import trackerlist
from lib.test.evaluation.environment import env_settings

import os
import shutil
from pathlib import Path

def prepare_results_for_framework(source_dir, temp_dir):
    """
    Copy and rename result files to match framework expectations.

    Old structure:
      source_dir/sequence_name/sequence_name_001.txt
    New structure (your screenshot):
      source_dir/1-Boat.txt, source_dir/1-Boat_time.txt, ...

    Framework expects:
      temp_dir/<sequence_name>.txt   (no *_time.txt)
    """
    print("Preparing results for framework evaluation...")

    source_dir = Path(source_dir)
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    # --- 1) New structure: flat files in source_dir ---
    flat_result_files = [
        p for p in source_dir.glob("*.txt")
        if p.is_file() and not p.name.endswith("_time.txt")
    ]

    for src in sorted(flat_result_files):
        seq_name = src.stem  # "1-Boat" from "1-Boat.txt"
        dst = temp_dir / f"{seq_name}.txt"
        shutil.copy2(src, dst)
        copied_count += 1
        print(f"  ✅ {seq_name}: {src.name} -> {dst.name}")

    # --- 2) Fallback: Old structure with per-sequence subfolders ---
    if copied_count == 0:
        sequence_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

        for seq_path in sorted(sequence_dirs):
            seq_name = seq_path.name

            result_files = sorted(
                [p for p in seq_path.glob("*.txt") if not p.name.endswith("_time.txt")]
            )

            if not result_files:
                print(f"  ❌ {seq_name}: No result file found")
                continue

            src = result_files[0]
            dst = temp_dir / f"{seq_name}.txt"
            shutil.copy2(src, dst)
            copied_count += 1
            print(f"  ✅ {seq_name}: {src.name} -> {dst.name}")

    print(f"Prepared {copied_count} result files")
    return copied_count


def main():
    """Main function to evaluate SimTrack performance"""
    
    print("=" * 60)
    print("SimTrack Performance Evaluation on MVTD Dataset")
    print("=" * 60)
    
    # Configuration
    original_results_dir = "/home/vli/thangdd_workspace/MOT/models/HIPTrack/output/test/tracking_results/hiptrack/hiptrack/got10k"
    parameter_name = original_results_dir.split("/")[-2]

    
    try:
        # Load dataset - try both val and test splits
        print("Loading MVTD dataset...")
        
        # Try to load validation split first (more common for evaluation)
        try:
            dataset = get_dataset('got10k_val') # 'got10k_val' is baseline for mvtd
            dataset_split = 'val'
            print(f"✅ Loaded GOT-10k validation dataset with {len(dataset)} sequences")
        except:
            try:
                dataset = get_dataset('got10k_test')
                dataset_split = 'test'
                print(f"✅ Loaded GOT-10k test dataset with {len(dataset)} sequences")
            except Exception as e:
                print(f"❌ Failed to load GOT-10k dataset: {e}")
                print("Please check your dataset path in local.py")
                return
        
        # Check if original results exist
        if not os.path.exists(original_results_dir):
            print(f"❌ Results directory not found: {original_results_dir}")
            print("Please make sure you have run the tracker and generated results first.")
            return
        
        print(f"✅ Found results in: {original_results_dir}")
        
        # Create temporary directory for framework-compatible results
        with tempfile.TemporaryDirectory() as temp_results_dir:
            print(f"Using temporary directory: {temp_results_dir}")
            
            # Prepare results in the format expected by the framework
            copied_count = prepare_results_for_framework(original_results_dir, temp_results_dir)
            
            if copied_count == 0:
                print("❌ No result files could be prepared!")
                return
            
            # Create tracker list for SimTrack
            print("\nSetting up tracker...")
            
            tracker_name = 'hiptrack_cls'
            # parameter_name = 'baseline_got10k_only_50e_64bs'
            
            trackers = trackerlist(
                name=tracker_name,
                parameter_name=parameter_name,
                dataset_name='got10k',
                run_ids=None,
                display_name='HIPTrackCls',
                result_only=True
            )
            
            # Override the results directory to point to our temporary directory
            trackers[0].results_dir = temp_results_dir
            
            print(f"✅ Created tracker: {trackers[0].name} with parameter: {trackers[0].parameter_name}")
            print(f"Results directory: {trackers[0].results_dir}")
            
            # Evaluate using the framework's evaluation functions
            print("\nEvaluating performance using official framework...")
            
            report_name = f"hiptrack_{parameter_name}_{dataset_split}_evaluation"
            
            # Use the framework's print_results function
            print_results(
                trackers=trackers,
                dataset=dataset, 
                report_name=report_name,
                merge_results=False,
                plot_types=('success', 'prec', 'norm_prec')
            )
            
            print("\n" + "=" * 60)
            print("Evaluation completed successfully!")
            print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()