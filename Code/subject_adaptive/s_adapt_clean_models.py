import subprocess
import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

dependent_code_path = os.path.dirname(os.path.abspath(__file__))


# Use sys.executable instead of 'python' to use the same Python interpreter
ms_script_path = os.path.join(dependent_code_path, 'ms_adapt_clean.py')
comb_ms_dcn_script_path = os.path.join(dependent_code_path, 'comb_ms_dcn_adapt_clean.py')
subprocess.run([sys.executable, ms_script_path], check=True)
subprocess.run([sys.executable, comb_ms_dcn_script_path], check=True)