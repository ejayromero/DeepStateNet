#!/bin/bash
# run_all_ms_experiments.sh - Run all three MS experiment types
# Custom script for your microstate experiments

# Make sure we're in the Code directory
if [[ ! -d "lib" || ! -d "dependent" ]]; then
    echo "❌ Error: Please run this script from the Code/ directory"
    echo "Current directory: $(pwd)"
    echo "Expected structure: Code/lib/, Code/dependent/, Code/independent/, Code/adaptive/"
    exit 1
fi

echo "🧠 Microstate Experiments Parallel Scheduler"
echo "============================================="
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Check if scripts exist
scripts_to_check=("dependent/ms.py" "independent/ms_indep.py" "adaptive/ms_adapt.py")
for script in "${scripts_to_check[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "❌ Error: Script not found: $script"
        exit 1
    fi
done

# Check if .venv exists
if [[ ! -d ".venv" ]]; then
    echo "⚠️  Warning: .venv directory not found"
    echo "Please create a virtual environment first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install torch numpy pandas matplotlib seaborn"
    exit 1
fi

echo "✅ All scripts found and .venv exists"
echo ""

# Submit all three experiments
echo "📤 Submitting experiments to job queue..."
echo ""

echo "1️⃣  Submitting DEPENDENT experiment..."
./tmux_scheduler.sh submit dependent/ms.py "dependent_50subj" .venv
echo ""

echo "2️⃣  Submitting INDEPENDENT experiment..."
./tmux_scheduler.sh submit independent/ms_indep.py "independent_50subj" .venv
echo ""

echo "3️⃣  Submitting ADAPTIVE experiment..."
./tmux_scheduler.sh submit adaptive/ms_adapt.py "adaptive_50subj" .venv
echo ""

echo "✅ All jobs submitted successfully!"
echo ""

# Show current queue
echo "📋 Current job queue:"
./tmux_scheduler.sh queue

echo "🚀 Starting parallel worker (3 jobs simultaneously)..."
echo "   This will run all three experiments at the same time!"
echo ""

# Start worker in background
nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &
worker_pid=$!

echo "✅ Worker started in background (PID: $worker_pid)"
echo ""
echo "📊 MONITORING COMMANDS:"
echo "  ./tmux_scheduler.sh status          # Show all job statuses"
echo "  ./tmux_scheduler.sh monitor JOB     # Watch specific job live"
echo "  ./tmux_scheduler.sh logs JOB        # View job output"
echo "  tail -f worker.log                  # Watch worker progress"
echo ""
echo "📁 IMPORTANT LOCATIONS:"
echo "  Logs: ../.logs/                     # All job logs saved here"
echo "  Worker log: worker.log               # Worker status and errors"
echo "  Results will be in: ../Output/ica_rest_all/"
echo "    ├── dependent/"
echo "    ├── independent/"
echo "    └── adaptive/"
echo ""
echo "🔄 EXPECTED BEHAVIOR:"
echo "  • 3 tmux sessions will be created: job_dependent_50subj, job_independent_50subj, job_adaptive_50subj"
echo "  • Each will process 50 subjects independently"
echo "  • Sessions will auto-close when complete"
echo "  • You can safely disconnect from SSH - jobs will continue running"
echo ""
echo "⚡ QUICK STATUS CHECK:"
sleep 3
./tmux_scheduler.sh status
echo ""
echo "🎯 All set! Your experiments are now running in parallel."
echo ""
echo "💡 TIP: You can disconnect from SSH now. When you reconnect, use:"
echo "     cd your_project_folder/Code"
echo "     ./tmux_scheduler.sh status"
echo ""
echo "🛑 TO STOP ALL JOBS (if needed):"
echo "     ./tmux_scheduler.sh kill dependent_50subj"
echo "     ./tmux_scheduler.sh kill independent_50subj"  
echo "     ./tmux_scheduler.sh kill adaptive_50subj"
echo "     pkill -f tmux_scheduler  # Stop the worker"