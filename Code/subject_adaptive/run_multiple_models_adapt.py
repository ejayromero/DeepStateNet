#!/usr/bin/env python3
"""
Multiple adaptive model training script using subprocess
Supports DCN_ADAPT, MSN_ADAPT, and DSN_ADAPT models
Loads pretrained subject-independent models and fine-tunes them on individual subjects
Usage: python run_multiple_models_adapt.py
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
    print(f"🔄 Starting: {description}")
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
    """Create all combinations of adaptive models and settings"""
    
    # ============ CONFIGURATION ============
    # Which models to run - MODIFY THIS LIST to control what gets trained
    models_to_run = ['dcn_adapt', 'msn_adapt', 'dsn_adapt']  # Options: 'dcn_adapt', 'msn_adapt', 'dsn_adapt'
    
    # Training parameters for adaptive fine-tuning
    clusters = [5, 12]  # Multiple cluster sizes (for MSN and DSN)
    n_subjects = 50  # Fine-tune on all 50 subjects individually
    epochs = 50     # Reduced epochs for fine-tuning (was 100 for training from scratch)
    n_folds = 4     # 4-fold CV on 90% of each subject's data
    batch_size = 32
    lr = 1e-3       # Base LR (will be adapted based on pretrained performance)
    early_stopping_patience = 15
    
    # MSN/DSN Model configurations (clusters matter for these)
    msn_model_configs = [
        # {'model': 'msn', 'embedding': False, 'name': 'MSN Adaptive'},
        # {'model': 'multiscale_msn', 'embedding': False, 'name': 'MultiScale MSN Adaptive'},
        {'model': 'msn', 'embedding': True, 'name': 'MSN Adaptive (Embedded)'},
        # {'model': 'multiscale_msn', 'embedding': True, 'name': 'MultiScale MSN Adaptive (Embedded)'},
    ]
    
    # DCN configuration (clusters don't matter for DCN)
    dcn_config = {
        'model': 'dcn_adapt',
        'embedding': False,
        'name': 'DCN Adaptive',
        'clusters': None  # DCN doesn't use clusters
    }
    
    combinations = []
    
    # Add DCN_ADAPT combinations (only if requested)
    if 'dcn_adapt' in models_to_run:
        combination = {
            'script': 'dcn_adapt',
            'clusters': None,  # DCN doesn't use clusters
            'model': 'dcn',     # Internal model name
            'embedding': dcn_config['embedding'],
            'name': dcn_config['name'],
            'description': 'DCN Adaptive Fine-tuning (90/10 split + 4-fold CV)',
            'n_subjects': n_subjects,
            'epochs': epochs,
            'n_folds': n_folds,
            'batch_size': batch_size,
            'lr': lr,
            'early_stopping_patience': early_stopping_patience
        }
        combinations.append(combination)
    
    # Add MSN_ADAPT combinations (only if requested)
    if 'msn_adapt' in models_to_run:
        for cluster_size in clusters:
            for model_config in msn_model_configs:
                combination = {
                    'script': 'msn_adapt',
                    'clusters': cluster_size,
                    'model': model_config['model'],
                    'embedding': model_config['embedding'],
                    'name': model_config['name'],
                    'description': f"{model_config['name']} Fine-tuning (c={cluster_size})",
                    'n_subjects': n_subjects,
                    'epochs': epochs,
                    'n_folds': n_folds,
                    'batch_size': batch_size,
                    'lr': lr,
                    'early_stopping_patience': early_stopping_patience
                }
                combinations.append(combination)
    
    # Add DSN_ADAPT combinations (only if requested and requires DCN+MSN pretrained models)
    if 'dsn_adapt' in models_to_run:
        print("⚠️  WARNING: DSN_ADAPT requires pretrained independent DCN and MSN models.")
        print("    Please ensure you have run dcn_indep.py and msn_indep.py first.")
        
        for cluster_size in clusters:
            for model_config in msn_model_configs:
                combination = {
                    'script': 'dsn_adapt',
                    'clusters': cluster_size,
                    'model': model_config['model'],
                    'embedding': model_config['embedding'],
                    'name': f"DSN Adaptive + {model_config['name']}",
                    'description': f"DeepStateNet Adaptive + {model_config['name']} (c={cluster_size})",
                    'n_subjects': n_subjects,
                    'epochs': epochs,
                    'n_folds': n_folds,
                    'batch_size': batch_size,
                    'lr': lr,
                    'early_stopping_patience': early_stopping_patience
                }
                combinations.append(combination)
    
    return combinations, models_to_run

def build_command(script_name, config):
    """Build command for a specific configuration"""
    # Use full path to script since we know where they are
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{script_name}.py")
    
    # DCN_ADAPT command
    if script_name == 'dcn_adapt':
        command = [
            sys.executable, script_path,  # Use full path instead of just script_name.py
            '--n-subjects', str(config['n_subjects']),
            '--epochs', str(config['epochs']),
            '--n-folds', str(config['n_folds']),
            '--batch-size', str(config['batch_size']),
            '--lr', str(config['lr']),
            '--early-stopping-patience', str(config['early_stopping_patience'])
        ]
        return command
    
    # For MSN_ADAPT and DSN_ADAPT, build full command with parameters
    command = [
        sys.executable, script_path,  # Use full path instead of just script_name.py
        '--n-subjects', str(config['n_subjects']),
        '--epochs', str(config['epochs']),
        '--n-folds', str(config['n_folds']),
        '--batch-size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--early-stopping-patience', str(config['early_stopping_patience'])
    ]
    
    # Add model-specific parameters for MSN_ADAPT and DSN_ADAPT
    if script_name in ['msn_adapt', 'dsn_adapt']:
        command.extend([
            '--n-clusters', str(config['clusters']),
            '--model-name', config['model']
        ])
        
        # Add embedding flag if needed
        if config['embedding']:
            command.append('--use-embedding')
    
    return command
    

def check_prerequisites(models_to_run):
    """Check if all required scripts exist and pretrained models are available"""
    # Get absolute paths like the unified functions do
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from script dir to project root
    
    missing_scripts = []
    
    # Check if adaptive scripts exist (they should be in same directory as this script)
    for model in models_to_run:
        script_path = os.path.join(script_dir, f"{model}.py")
        if not os.path.exists(script_path):
            missing_scripts.append(script_path)
        else:
            print(f"✅ Found script: {script_path}")
    
    if missing_scripts:
        print(f"\n❌ Missing scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        return False
    
    # Check for pretrained independent models (required for adaptive training)
    print(f"\n🔍 Checking for pretrained independent models (required for adaptive training)...")
    output_folder = os.path.join(project_root, 'Output') + os.sep
    
    missing_models = []
    
    # Check for DCN independent models
    if 'dcn_adapt' in models_to_run:
        dcn_results_path = f"{output_folder}ica_rest_all/independent/independent_dcn_4fold_results/independent_dcn_4fold_results.npy"
        if not os.path.exists(dcn_results_path):
            missing_models.append(f"DCN independent: {dcn_results_path}")
            print(f"⚠️  Missing DCN independent results: {dcn_results_path}")
            print("    Please run dcn_indep.py first.")
        else:
            print(f"✅ Found DCN independent results: {dcn_results_path}")

    # Check for MSN independent models
    if any(model in models_to_run for model in ['msn_adapt', 'dsn_adapt']):
        # Check for basic MSN models (we'll check specific combinations later)
        msn_found = False
        for cluster in [5, 12]:
            for embedding in [False, True]:
                embedding_suffix = "_embedded" if embedding else ""
                msn_path = f"{output_folder}ica_rest_all/independent/independent_msn{embedding_suffix}_c{cluster}_4fold_results/independent_msn{embedding_suffix}_c{cluster}_4fold_results.npy"
                if os.path.exists(msn_path):
                    print(f"✅ Found MSN independent results: msn{embedding_suffix}_c{cluster}")
                    msn_found = True
                else:
                    print(f"⚠️  Missing MSN independent results: {msn_path}")
        
        if not msn_found:
            missing_models.append("MSN independent models")
            print("    Please run msn_indep.py first.")
    
    if missing_models:
        print(f"\n❌ Missing required pretrained models:")
        for model in missing_models:
            print(f"  - {model}")
        print(f"\n📋 PREREQUISITES FOR ADAPTIVE TRAINING:")
        print(f"  1. Run dcn_indep.py to generate independent DCN models")
        print(f"  2. Run msn_indep.py to generate independent MSN models")
        print(f"  3. Then run adaptive training to fine-tune these models")
        return False
    
    return True

def main():
    """Main function to run multiple adaptive models"""
    
    print("="*70)
    print("🔄 MULTIPLE ADAPTIVE MODEL TRAINING SCRIPT - DCN, MSN, DSN")
    print("="*70)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Create all model combinations
    combinations, models_to_run = create_model_combinations()
    
    print(f"\n📋 ADAPTIVE MODELS TO RUN: {', '.join(models_to_run)}")
    print(f"🎯 Training Method: Adaptive Fine-tuning")
    print(f"📊 Each subject: 90/10 split + 4-fold CV on 90% for fine-tuning")
    print(f"🔄 Process: Load pretrained independent model → Fine-tune on subject data")
    
    # Check prerequisites
    print(f"\n🔍 Checking prerequisites...")
    if not check_prerequisites(models_to_run):
        print("❌ Prerequisites not met. Please run independent training first.")
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
    
    print(f"\n📋 ADAPTIVE TRAINING PLAN:")
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
    
    # Estimate time based on model type (adaptive is much faster than training from scratch)
    dcn_count = len([c for c in combinations if c['script'] == 'dcn_adapt'])
    msn_count = len([c for c in combinations if c['script'] == 'msn_adapt'])
    dsn_count = len([c for c in combinations if c['script'] == 'dsn_adapt'])
    
    # Adaptive training times (much faster due to early stopping and fewer epochs)
    estimated_time = (dcn_count * 800) + (msn_count * 1200) + (dsn_count * 300)  # minutes
    print(f"Estimated time: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")
    print(f"  - DCN Adaptive: {dcn_count} models × ~13 hours each (50 subjects × ~16 min)")
    print(f"  - MSN Adaptive: {msn_count} models × ~20 hours each (50 subjects × ~24 min)") 
    print(f"  - DSN Adaptive: {dsn_count} models × ~5 hours each (50 subjects × ~6 min)")
    print(f"⚡ Note: Adaptive training is much faster due to early stopping and fewer epochs")
    print(f"🎯 Benefits: Starting from good pretrained weights + adaptive learning rates")
    
    # Show configuration
    if combinations:
        sample_config = combinations[0]
        print(f"\n⚙️ ADAPTIVE TRAINING CONFIGURATION:")
        print(f"  Subjects: {sample_config['n_subjects']} (fine-tune each individually)")
        print(f"  Epochs: {sample_config['epochs']} (reduced from 100 for fine-tuning)")
        print(f"  Folds: {sample_config['n_folds']} (CV on 90% of each subject)")
        print(f"  Test split: 10% held out for final evaluation")
        print(f"  Batch size: {sample_config['batch_size']}")
        print(f"  Base learning rate: {sample_config['lr']} (adaptively adjusted)")
        print(f"  Early stopping patience: {sample_config['early_stopping_patience']}")
        
        if any(c['clusters'] for c in combinations):
            cluster_list = list(set(c['clusters'] for c in combinations if c['clusters']))
            print(f"  Cluster sizes: {cluster_list}")
    
    # Special notes for adaptive training
    print(f"\n📝 ADAPTIVE TRAINING NOTES:")
    print(f"  - Loads pretrained independent models for each subject")
    print(f"  - Adaptive learning rate based on pretrained performance:")
    print(f"    * Good performance (>60% DCN, >40% MSN): Conservative LR (1e-3)")
    print(f"    * Poor performance: Higher LR (3e-3) to escape bad patterns")
    print(f"  - All layers unfrozen for full adaptation")
    print(f"  - Early stopping prevents overfitting on limited subject data")
    print(f"  - Results show improvement over pretrained baseline")
    
    # Ask for confirmation
    response = input(f"\nProceed with adaptive training of {len(combinations)} model combinations? (y/N): ")
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
        remaining_dcn = len([c for c in remaining_combos if c['script'] == 'dcn_adapt'])
        remaining_msn = len([c for c in remaining_combos if c['script'] == 'msn_adapt'])
        remaining_dsn = len([c for c in remaining_combos if c['script'] == 'dsn_adapt'])
        remaining_time = (remaining_dcn * 800) + (remaining_msn * 1200) + (remaining_dsn * 300)
        
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
    print("📋 FINAL ADAPTIVE TRAINING SUMMARY")
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
    
    # Final notes
    print(f"\n🎯 ADAPTIVE TRAINING COMPLETE!")
    print(f"📊 Results compare fine-tuned vs. pretrained independent performance")
    print(f"📈 Look for 'improvement' metrics in saved results files")
    print(f"💾 Results saved in Output/ica_rest_all/adaptive/ folders")

if __name__ == "__main__":
    main()