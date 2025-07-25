import subprocess
import os 
import torch

# change directory to Notebooks folder
if os.path.basename(os.getcwd()) != 'Notebooks':
    if os.path.basename(os.getcwd()) == 'lib':
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
    else:
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
  
# Check available CUDA devices
print("=== GPU Detection ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Use the first (and likely only) CUDA device
    device = torch.device("cuda:0")  # Your RTX GPU
    torch.cuda.set_device(0)
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Test if tensors are actually on GPU
if torch.cuda.is_available():
    test_tensor = torch.randn(10, 10).to(device)
    print(f"Test tensor device: {test_tensor.device}")

# Set environment variables for subprocesses
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first CUDA device (your RTX)
env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

print("\n=== Running Scripts ===")
print("Running scripts with CUDA environment variables...")

# Run scripts sequentially
scripts = ['dcn_adapt.py', 'ms_adapt.py', 'comb_ms_dcn_adapt.py']

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running {script}...") 
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(['python', script], shell=True, env=env, check=True)
        print(f"✅ {script} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} failed with error code {e.returncode}")
        print(f"Error: {e}")
        # Ask whether to continue
        user_input = input(f"Continue with next script? (y/n): ")
        if user_input.lower() != 'y':
            print("Stopping execution.")
            break
    except FileNotFoundError:
        print(f"❌ Script {script} not found!")
        continue

print("\n🎉 Script execution completed!")