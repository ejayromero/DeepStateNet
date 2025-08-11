#!/usr/bin/env python3
"""
Multiple model training script using subprocess
Supports multiple clusters and embedding variants
Usage: python run_multiple_models_updated.py
"""

import subprocess
import sys
import time
import os
from datetime import datetime
import itertools

def run_command(command, model_name, description=""):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
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
        
        print(f"\n✅ {description} completed successfully!")
        print(f"Duration: {duration/60:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ {description} failed!")
        print(f"Error code: {e.returncode}")
        print(f"Duration: {duration/60:.1f} minutes")
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user!")
        return False

def create_model_combinations():
    """Create all combinations of models and settings"""
    
    # Configuration
    clusters = [5,12]  # Multiple cluster sizes
    n_subjects = 50
    epochs = 100
    n_folds = 4
    batch_size = 32
    lr = 1e-3
    
    # Model configurations
    model_configs = [
        {'model': 'microstatenet', 'embedding': False, 'name': 'MicroStateNet'},
        {'model': 'multiscale_microstatenet', 'embedding': False, 'name': 'MultiScale MicroStateNet'},
        {'model': 'microstatenet', 'embedding': True, 'name': 'MicroStateNet (Embedded)'},
        {'model': 'multiscale_microstatenet', 'embedding': True, 'name': 'MultiScale MicroStateNet (Embedded)'},
        # {'model': 'embedded_microstatenet', 'embedding': False, 'name': 'Legacy Embedded MicroStateNet'}  # Legacy model
    ]
    
    # Create all combinations
    combinations = []
    for cluster_size in clusters:
        for model_config in model_configs:
            combination = {
                'clusters': cluster_size,
                'model': model_config['model'],
                'embedding': model_config['embedding'],
                'name': model_config['name'],
                'description': f"{model_config['name']} (c={cluster_size})",
                'n_subjects': n_subjects,
                'epochs': epochs,
                'n_folds': n_folds,
                'batch_size': batch_size,
                'lr': lr
            }
            combinations.append(combination)
    
    return combinations

def build_command(script_path, config):
    """Build command for a specific configuration"""
    command = [
        sys.executable, script_path,
        '--n-clusters', str(config['clusters']),
        '--model-name', config['model'],
        '--n-subjects', str(config['n_subjects']),
        '--epochs', str(config['epochs']),
        '--n-folds', str(config['n_folds']),
        '--batch-size', str(config['batch_size']),
        '--lr', str(config['lr'])
    ]
    
    # Add embedding flag if needed
    if config['embedding']:
        command.append('--use-embedding')
    
    return command

def main():
    """Main function to run multiple models"""
    
    print("="*70)
    print("🔬 MULTIPLE MODEL TRAINING SCRIPT - CLUSTERS & EMBEDDINGS")
    print("="*70)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Test if we can find the msn.py script
    script_path = "Code/subject_dependent/msn.py"
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
    
    # Create all model combinations
    combinations = create_model_combinations()
    
    print(f"\n📋 TRAINING PLAN:")
    print("=" * 50)
    for i, combo in enumerate(combinations, 1):
        embedding_str = " + Embedding" if combo['embedding'] else ""
        print(f"{i:2d}. {combo['name']} (c={combo['clusters']}){embedding_str}")
    
    print(f"\nTotal combinations: {len(combinations)}")
    print(f"Estimated time: ~{len(combinations) * 90} minutes (assuming 90 min per model)")
    
    # Show configuration
    sample_config = combinations[0]
    print(f"\n⚙️ CONFIGURATION:")
    print(f"Subjects: {sample_config['n_subjects']}")
    print(f"Epochs: {sample_config['epochs']}")
    print(f"Folds: {sample_config['n_folds']}")
    print(f"Batch size: {sample_config['batch_size']}")
    print(f"Learning rate: {sample_config['lr']}")
    
    # Ask for confirmation
    response = input(f"\nProceed with training {len(combinations)} model combinations? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run each combination
    for i, config in enumerate(combinations, 1):
        print(f"\n📊 Progress: {i}/{len(combinations)} combinations")
        print(f"⏰ Estimated remaining: {(len(combinations) - i) * 30:.0f} minutes")
        
        # Build command
        command = build_command(script_path, config)
        
        # Create unique identifier for this combination
        combo_id = f"{config['model']}_c{config['clusters']}"
        if config['embedding']:
            combo_id += "_embedded"
        
        # Run the model
        success = run_command(command, combo_id, config['description'])
        results[combo_id] = {
            'success': success,
            'description': config['description'],
            'config': config
        }
        
        if not success:
            response = input(f"\n{config['description']} failed. Continue with remaining models? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Training stopped by user.")
                break
    
    # Final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*70}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Group results by success/failure
    successful = []
    failed = []
    
    for combo_id, result in results.items():
        if result['success']:
            successful.append(result['description'])
        else:
            failed.append(result['description'])
    
    print(f"\n✅ SUCCESSFUL ({len(successful)}):")
    for desc in successful:
        print(f"  ✓ {desc}")
    
    if failed:
        print(f"\n❌ FAILED ({len(failed)}):")
        for desc in failed:
            print(f"  ✗ {desc}")
    
    print(f"\n📊 SUCCESS RATE: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    # Results by cluster size
    print(f"\n📈 RESULTS BY CLUSTER SIZE:")
    for cluster_size in [5, 12]:
        cluster_results = [r for combo_id, r in results.items() if r['config']['clusters'] == cluster_size]
        cluster_successful = [r for r in cluster_results if r['success']]
        print(f"  Cluster {cluster_size}: {len(cluster_successful)}/{len(cluster_results)} successful")
    
    # Results by model type
    print(f"\n🤖 RESULTS BY MODEL TYPE:")
    model_types = {}
    for combo_id, result in results.items():
        model_key = result['config']['model']
        if result['config']['embedding']:
            model_key += " (embedded)"
        
        if model_key not in model_types:
            model_types[model_key] = {'total': 0, 'successful': 0}
        
        model_types[model_key]['total'] += 1
        if result['success']:
            model_types[model_key]['successful'] += 1
    
    for model_type, stats in model_types.items():
        success_rate = stats['successful'] / stats['total'] * 100
        print(f"  {model_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")

if __name__ == "__main__":
    main()