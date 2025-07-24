import subprocess
import sys
import os 

# change directory to Notebooks folder
if os.path.basename(os.getcwd()) != 'Notebooks':
    if os.path.basename(os.getcwd()) == 'lib':
        os.chdir(os.path.join(os.getcwd(), '..', 'Notebooks'))
    else:
        os.chdir(os.path.join(os.getcwd(), 'Notebooks'))
else:
    # if already in Notebooks folder, do nothing
    pass

import torch

# More explicit CUDA setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Specify GPU 0 explicitly
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Test if tensors are actually on GPU
test_tensor = torch.randn(10, 10).to(device)
print(f"Test tensor device: {test_tensor.device}")

# Use sys.executable instead of 'python' to use the same Python interpreter
subprocess.run([sys.executable, 'ms_dcn.py'], check=True)
subprocess.run([sys.executable, 'comb_ms_dcn.py'], check=True)