#!/usr/bin/env python3
"""
Multiple independent model training script using subprocess
Supports DCN_INDEP, MSN_INDEP (MicroStateNet), and DSN_INDEP (DeepStateNet) models
Usage: python run_multiple_models_indep.py
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
    """Create all combinations of independent models and settings"""
    
    # ============ CONFIGURATION ============
    # Which models to run - MODIFY THIS LIST to control what gets trained
    models_to_run = ['dcn_indep', 'msn_indep', 'dsn_indep']  # Options: 'dcn_indep', 'msn_indep', 'dsn_indep'
    
    # Training parameters
    clusters = [5]  # Multiple cluster sizes (for MSN and DSN)
    n_subjects = 50  # LOSO: each subject becomes test subject once
    epochs = 100
    n_folds = 4      # 4-fold CV on remaining 49 subjects
    batch_size = 32
    lr = 1e-3
    
    # MSN/DSN Model configurations (clusters matter for these)
    msn_model_configs = [
        {'model': 'msn', 'embedding': False, 'name': 'MSN LOSO'},
        {'model': 'multiscale_msn', 'embedding': False, 'name': 'MultiScale MSN LOSO'},
        {'model': 'msn', 'embedding': True, 'name': 'MSN LOSO (Embedded)'},
        {'model': 'multiscale_msn', 'embedding': True, 'name': 'MultiScale MSN LOSO (Embedded)'},
    ]
    
    # DCN configuration (clusters don't matter for DCN)
    dcn_config = {
        'model': 'dcn_indep',
        'embedding': False,
        'name': 'DCN LOSO',
        'clusters': None  # DCN doesn't use clusters
    }
    
    combinations = []
    
    # Add DCN_INDEP combinations (only if requested)
    if 'dcn_indep' in models_to_run:
        combination = {
            'script': 'dcn_indep',
            'clusters': None,  # DCN doesn't use clusters
            'model': 'dcn',     # Internal model name
            'embedding': dcn_config['embedding'],
            'name': dcn_config['name'],
            'description': 'DCN LOSO (Leave-One-Subject-Out)',
            'n_subjects': n_subjects,
            'epochs': epochs,
            'n_folds': n_folds,
            'batch_size': batch_size,
            'lr': lr
        }
        combinations.append(combination)
    
    # Add MSN_INDEP combinations (only if requested)
    if 'msn_indep' in models_to_run:
        for cluster_size in clusters:
            for model_config in msn_model_configs:
                combination = {
                    'script': 'msn_indep',
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
    
    # Add DSN_INDEP combinations (only if requested and requires DCN+MSN pretrained models)
    if 'dsn_indep' in models_to_run:
        print("⚠️  WARNING: DSN_INDEP requires pretrained DCN and MSN models from dependent training.")
        print("    Please ensure you have run dcn.py and msn.py (dependent versions) first.")
        
        for cluster_size in clusters:
            for model_config in msn_model_configs:
                combination = {
                    'script': 'dsn_indep',
                    'clusters': cluster_size,
                    'model': model_config['model'],
                    'embedding': model_config['embedding'],
                    'name': f"DSN LOSO + {model_config['name']}",
                    'description': f"DeepStateNet LOSO + {model_config['name']} (c={cluster_size})",
                    'n_subjects': n_subjects,
                    'epochs': epochs,
                    'n_folds': n_folds,
                    'batch_size': batch_size,
                    'lr': lr
                }
                combinations.append(combination)
    
    return combinations, models_to_run

def build_command(script_name, config):
    """Build command for a specific configuration"""
    script_path = f"Code/subject_independent/{script_name}.py"
    
    # DCN_INDEP can run with just defaults - no additional parameters needed
    if script_name == 'dcn_indep':
        command = [sys.executable, script_path]
        # DCN_INDEP will use all its defaults:
        # --batch-size 32, --epochs 100, --lr 1e-3, --n-subjects 50, 
        # --type-of-subject independent, --n-folds 4, --save-model True
        return command
    
    # For MSN_INDEP and DSN_INDEP, build full command with parameters
    command = [
        sys.executable, script_path,
        '--n-subjects', str(config['n_subjects']),
        '--epochs', str(config['epochs']),
        '--n-folds', str(config['n_folds']),
        '--batch-size', str(config['batch_size']),
        '--lr', str(config['lr'])
    ]
    
    # Add model-specific parameters for MSN_INDEP and DSN_INDEP
    if script_name in ['msn_indep', 'dsn_indep']:
        command.extend([
            '--n-clusters', str(config['clusters']),
            '--model-name', config['model']
        ])
        
        # Add embedding flag if needed
        if config['embedding']:
            command.append('--use-embedding')
    
    return command

def check_prerequisites(models_to_run):
    """Check if all required scripts exist"""
    script_dir = "Code/subject_independent"
    missing_scripts = []
    
    for model in models_to_run:
        script_path = f"{script_dir}/{model}.py"
        if not os.path.exists(script_path):
            missing_scripts.append(script_path)
        else:
            print(f"✅ Found script: {os.path.abspath(script_path)}")
    
    if missing_scripts:
        print(f"\n❌ Missing scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        return False
    
    # Check for pretrained models if DSN_INDEP is requested
    if 'dsn_indep' in models_to_run:
        print(f"\n🔍 Checking for pretrained models (required for DSN_INDEP)...")
        output_dir = "Output/ica_rest_all/dependent"
        
        # Check for DCN dependent models
        dcn_results_path = f"{output_dir}/dependent_dcn_4fold_results/dependent_dcn_4fold_results.npy"
        if not os.path.exists(dcn_results_path):
            print(f"⚠️  Missing DCN dependent results: {dcn_results_path}")
            print("    Please run dcn.py (dependent version) first.")
        else:
            print(f"✅ Found DCN dependent results")
        
        # Check for MSN dependent models (at least basic MSN)
        msn_basic_path = f"{output_dir}/dependent_msn_c12_4fold_results/dependent_msn_c12_4fold_results.npy"
        if not os.path.exists(msn_basic_path):
            print(f"⚠️  Missing MSN dependent results: {msn_basic_path}")
            print("    Please run msn.py (dependent version) first.")
        else:
            print(f"✅ Found MSN dependent results")
    
    return True

def main():
    """Main function to run multiple independent models"""
    
    print("="*70)
    print("🔬 MULTIPLE INDEPENDENT MODEL TRAINING SCRIPT - DCN, MSN, DSN LOSO")
    print("="*70)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Create all model combinations
    combinations, models_to_run = create_model_combinations()
    
    print(f"\n📋 INDEPENDENT MODELS TO RUN: {', '.join(models_to_run)}")
    print(f"🎯 Training Method: LOSO (Leave-One-Subject-Out)")
    print(f"📊 Each subject becomes test subject once, 4-fold CV on remaining 49 subjects")
    
    # Check prerequisites
    print(f"\n🔍 Checking prerequisites...")
    if not check_prerequisites(models_to_run):
        print("❌ Prerequisites not met. Please ensure all scripts exist.")
        return
    
    # Test imports
    print(f"\n🧪 Testing imports...")
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
    
    if not combinations:
        print("❌ No valid combinations generated. Check your configuration.")
        return
    
    print(f"\n📋 TRAINING PLAN:")
    print("=" * 50)
    
    # Group by script type for better organization
    script_groups = {}
    for combo in combinations:
        script = combo['script'].upper()
        if script not in script_groups:
            script_groups[script] = []
        script_groups[script].append(combo)
    
    combo_num = 1
    for script_name, script_combos in script_groups.items():
        print(f"\n{script_name} Models:")
        for combo in script_combos:
            if combo['clusters']:
                cluster_str = f" (c={combo['clusters']})"
            else:
                cluster_str = ""
            
            embedding_str = " + Embedding" if combo['embedding'] else ""
            print(f"  {combo_num:2d}. {combo['name']}{cluster_str}{embedding_str}")
            combo_num += 1
    
    print(f"\nTotal combinations: {len(combinations)}")
    
    # Estimate time based on model type (LOSO takes much longer!)
    dcn_count = len([c for c in combinations if c['script'] == 'dcn_indep'])
    msn_count = len([c for c in combinations if c['script'] == 'msn_indep'])
    dsn_count = len([c for c in combinations if c['script'] == 'dsn_indep'])
    
    # LOSO training times: DCN/MSN train full models, DSN just combines features
    estimated_time = (dcn_count * 3000) + (msn_count * 4500) + (dsn_count * 500)  # minutes
    print(f"Estimated time: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")
    print(f"  - DCN LOSO: {dcn_count} models × ~50 hours each (50 subjects × ~60 min)")
    print(f"  - MSN LOSO: {msn_count} models × ~75 hours each (50 subjects × ~90 min)") 
    print(f"  - DSN LOSO: {dsn_count} models × ~8 hours each (50 subjects × ~10 min)")
    print(f"⏰ Note: DSN is much faster as it only trains classifier on pre-extracted features")
    print(f"⏰ WARNING: LOSO training is VERY time-consuming!")
    
    # Show configuration
    if combinations:
        sample_config = combinations[0]
        print(f"\n⚙️ CONFIGURATION:")
        if sample_config['script'] == 'dcn_indep':
            print("DCN_INDEP Configuration (using all defaults):")
            print("  Subjects: 50 (LOSO), Epochs: 100, Folds: 4, Batch size: 32, LR: 1e-3")
            print("  Type: independent, Save model: True")
        else:
            print(f"MSN_INDEP/DSN_INDEP Configuration:")
            print(f"  Subjects: {sample_config['n_subjects']} (LOSO)")
            print(f"  Epochs: {sample_config['epochs']}")
            print(f"  Folds: {sample_config['n_folds']} (CV on remaining 49 subjects)")
            print(f"  Batch size: {sample_config['batch_size']}")
            print(f"  Learning rate: {sample_config['lr']}")
        
        if any(c['clusters'] for c in combinations):
            cluster_list = list(set(c['clusters'] for c in combinations if c['clusters']))
            print(f"  Cluster sizes: {cluster_list}")
    
    # Special warning for LOSO
    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"  - LOSO training takes 50x longer than dependent training")
    print(f"  - Each model trains 50 times (once per test subject)")
    print(f"  - Results are saved after each subject for resume capability")
    print(f"  - You can interrupt and resume training later")
    
    # Ask for confirmation
    response = input(f"\nProceed with LOSO training of {len(combinations)} model combinations? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run each combination
    for i, config in enumerate(combinations, 1):
        print(f"\n📊 Progress: {i}/{len(combinations)} combinations")
        
        # Calculate estimated remaining time based on model types remaining
        remaining_combos = combinations[i:]
        remaining_dcn = len([c for c in remaining_combos if c['script'] == 'dcn_indep'])
        remaining_msn = len([c for c in remaining_combos if c['script'] == 'msn_indep'])
        remaining_dsn = len([c for c in remaining_combos if c['script'] == 'dsn_indep'])
        remaining_time = (remaining_dcn * 3000) + (remaining_msn * 4500) + (remaining_dsn * 500)
        
        print(f"⏰ Estimated remaining: {remaining_time:.0f} minutes ({remaining_time/60:.1f} hours)")
        
        # Build command
        command = build_command(config['script'], config)
        
        # Create unique identifier for this combination
        combo_id = f"{config['script']}_{config['model']}"
        if config['clusters']:
            combo_id += f"_c{config['clusters']}"
        if config['embedding']:
            combo_id += "_embedded"
        
        # Run the model
        success = run_command(command, combo_id, config['description'])
        results[combo_id] = {
            'success': success,
            'description': config['description'],
            'config': config,
            'script': config['script']
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
    
    # Results by script type
    print(f"\n🤖 RESULTS BY MODEL SCRIPT:")
    script_results = {}
    for combo_id, result in results.items():
        script = result['script'].upper()
        if script not in script_results:
            script_results[script] = {'total': 0, 'successful': 0}
        
        script_results[script]['total'] += 1
        if result['success']:
            script_results[script]['successful'] += 1
    
    for script_type, stats in script_results.items():
        success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {script_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
    
    # Results by cluster size (for MSN/DSN)
    cluster_results = {}
    for combo_id, result in results.items():
        if result['config']['clusters'] is not None:
            cluster_size = result['config']['clusters']
            if cluster_size not in cluster_results:
                cluster_results[cluster_size] = {'total': 0, 'successful': 0}
            
            cluster_results[cluster_size]['total'] += 1
            if result['success']:
                cluster_results[cluster_size]['successful'] += 1
    
    if cluster_results:
        print(f"\n📈 RESULTS BY CLUSTER SIZE:")
        for cluster_size, stats in sorted(cluster_results.items()):
            success_rate = stats['successful'] / stats['total'] * 100
            print(f"  Cluster {cluster_size}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
    
    # Results by model type
    print(f"\n🔬 RESULTS BY MODEL TYPE:")
    model_types = {}
    for combo_id, result in results.items():
        model_key = result['config']['name']
        
        if model_key not in model_types:
            model_types[model_key] = {'total': 0, 'successful': 0}
        
        model_types[model_key]['total'] += 1
        if result['success']:
            model_types[model_key]['successful'] += 1
    
    for model_type, stats in model_types.items():
        success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {model_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")

if __name__ == "__main__":
    main()