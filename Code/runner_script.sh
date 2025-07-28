#!/bin/bash
# runner_script.sh - Run all three MS experiment types
# Updated for conda environment (myvenv) and correct folder structure

# Make sure we're in the Code directory
if [[ ! -d "lib" || ! -d "subject_dependent" ]]; then
    echo "Error: Please run this script from the Code/ directory"
    echo "Current directory: $(pwd)"
    echo "Expected structure: Code/lib/, Code/subject_dependent/, Code/subject_independent/, Code/subject_adaptive/"
    exit 1
fi

echo "Microstate Experiments Parallel Scheduler"
echo "============================================="
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Check if scripts exist
scripts_to_check=("subject_dependent/ms.py" "subject_independent/ms_indep.py" "subject_adaptive/ms_adapt.py")
for script in "${scripts_to_check[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: Script not found: $script"
        exit 1
    fi
done

# Check if conda environment exists
echo "Checking conda environment 'myvenv'..."
if ! conda env list | grep -q "myvenv"; then
    echo "Error: Conda environment 'myvenv' not found"
    echo "Available environments:"
    conda env list
    echo ""
    echo "Please create the environment first:"
    echo "  conda create -n myvenv python=3.11 -y"
    echo "  conda activate myvenv"
    echo "  pip install torch numpy pandas matplotlib seaborn scikit-learn"
    exit 1
fi

# Test conda environment
echo "Testing conda environment 'myvenv'..."
source ~/miniconda3/bin/activate
conda activate myvenv
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Test imports
if python -c "import torch, numpy, pandas" 2>/dev/null; then
    echo "Essential packages found in myvenv"
else
    echo "Error: Missing required packages in myvenv"
    echo "Please install packages:"
    echo "  conda activate myvenv"
    echo "  pip install torch numpy pandas matplotlib seaborn scikit-learn"
    conda deactivate
    exit 1
fi
conda deactivate

echo "All scripts found and conda environment 'myvenv' is ready"
echo ""

# Submit all three experiments
echo "Submitting experiments to job queue..."
echo ""

echo "1. Submitting SUBJECT DEPENDENT experiment..."
./tmux_scheduler.sh submit subject_dependent/ms.py "subject_dependent_50subj" myvenv
echo ""

echo "2. Submitting SUBJECT INDEPENDENT experiment..."
./tmux_scheduler.sh submit subject_independent/ms_indep.py "subject_independent_50subj" myvenv
echo ""

echo "3. Submitting SUBJECT ADAPTIVE experiment..."
./tmux_scheduler.sh submit subject_adaptive/ms_adapt.py "subject_adaptive_50subj" myvenv
echo ""

echo "All jobs submitted successfully!"
echo ""

# Show current queue
echo "Current job queue:"
./tmux_scheduler.sh queue

echo "Starting parallel worker (3 jobs simultaneously)..."
echo "   This will run all three experiments at the same time!"
echo ""

# Start worker in background
nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &
worker_pid=$!

echo "Worker started in background (PID: $worker_pid)"
echo ""
echo "MONITORING COMMANDS:"
echo "  ./tmux_scheduler.sh status          # Show all job statuses"
echo "  ./tmux_scheduler.sh monitor JOB     # Watch specific job live"
echo "  ./tmux_scheduler.sh logs JOB        # View job output"
echo "  tail -f worker.log                  # Watch worker progress"
echo ""
echo "IMPORTANT LOCATIONS:"
echo "  Logs: ../.logs/                     # All job logs saved here"
echo "  Worker log: worker.log               # Worker status and errors"
echo "  Results will be in: ../Output/ica_rest_all/"
echo "    ├── dependent/"
echo "    ├── independent/"
echo "    └── adaptive/"
echo ""
echo "EXPECTED BEHAVIOR:"
echo "  • 3 tmux sessions will be created:"
echo "    - job_subject_dependent_50subj"
echo "    - job_subject_independent_50subj" 
echo "    - job_subject_adaptive_50subj"
echo "  • Each will process 50 subjects independently"
echo "  • Sessions will auto-close when complete"
echo "  • You can safely disconnect from SSH - jobs will continue running"
echo ""
echo "QUICK STATUS CHECK:"
sleep 3
./tmux_scheduler.sh status
echo ""
echo "All set! Your experiments are now running in parallel."
echo ""
echo "TIP: You can disconnect from SSH now. When you reconnect, use:"
echo "     cd your_project_folder/Code"
echo "     ./tmux_scheduler.sh status"
echo ""
echo "TO STOP ALL JOBS (if needed):"
echo "     ./tmux_scheduler.sh kill subject_dependent_50subj"
echo "     ./tmux_scheduler.sh kill subject_independent_50subj"  
echo "     ./tmux_scheduler.sh kill subject_adaptive_50subj"
echo "     pkill -f tmux_scheduler  # Stop the worker"