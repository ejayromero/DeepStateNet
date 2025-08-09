#!/usr/bin/env python3
"""
Multiple model training script using subprocess
Usage: python run_multiple_models_updated.py
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def run_command(command, model_name):
    """Run a command and handle output"""
    print(f"\n{'='*50}")
    print(f"🚀 Starting: {model_name}")
    print(f"Command: {' '.join(command)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(
            command, 
            capture_output=False,  # Show output in real-time
            text=True,
            check=True  # Raise exception if command fails
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {model_name} completed successfully!")
        print(f"Duration: {duration/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ {model_name} failed!")
        print(f"Error code: {e.returncode}")
        print(f"Duration: {duration/60:.1f} minutes")
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️ {model_name} interrupted by user!")
        return False

def main():
    """Main function to run multiple models"""
    
    print("="*60)
    print("🔬 MULTIPLE MODEL TRAINING SCRIPT")
    print("="*60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Test if we can find the ms.py script
    script_path = "Code/subject_dependent/ms.py"
    if not os.path.exists(script_path):
        print(f"❌ Could not find {script_path}")
        print("Make sure you're running this from the Master-Thesis root directory")
        print("Or update the script_path variable")
        return
    else:
        print(f"✅ Found script at: {os.path.abspath(script_path)}")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import numpy
        print("✅ numpy works")
    except ImportError:
        print("❌ numpy not found - make sure virtual environment is activated")
        return
        
    try:
        import pandas
        print("✅ pandas works")
    except ImportError:
        print("❌ pandas not found - make sure virtual environment is activated")
        return
    
    # Configuration - START WITH DEBUG PARAMETERS
    config = {
        'n_clusters': 12,
        'n_subjects': 50,     # Debug: Start with 1 subject
        'epochs': 100,        # Debug: Start with 10 epochs
        'n_folds': 4,        # Debug: Start with 2 folds
        'batch_size': 32,
        'lr': 1e-3
    }
    
    # Models to train
    models = [
        'microsnet',
        'multiscale_microsnet', 
        'embedded_microsnet'
    ]
    
    print(f"\nConfiguration: {config}")
    print(f"Models to train: {models}")
    print(f"Total models: {len(models)}")
    
    # Ask for confirmation
    response = input(f"\nProceed with training {len(models)} models? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run each model
    for i, model in enumerate(models, 1):
        print(f"\n📊 Progress: {i}/{len(models)} models")
        
        # Build command - USE sys.executable to ensure same Python environment
        command = [
            sys.executable, script_path,  # Use current Python and correct path
            '--n-clusters', str(config['n_clusters']),
            '--model-name', model,
            '--n-subjects', str(config['n_subjects']),
            '--epochs', str(config['epochs']),
            '--n-folds', str(config['n_folds']),
            '--batch-size', str(config['batch_size']),
            '--lr', str(config['lr'])
        ]
        
        # Run the model
        success = run_command(command, model)
        results[model] = success
        
        if not success:
            response = input(f"\n{model} failed. Continue with remaining models? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Training stopped by user.")
                break
    
    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful = [model for model, success in results.items() if success]
    failed = [model for model, success in results.items() if not success]
    
    print(f"\n✅ Successful ({len(successful)}): {successful}")
    if failed:
        print(f"❌ Failed ({len(failed)}): {failed}")
    
    print(f"\nSuccess rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()