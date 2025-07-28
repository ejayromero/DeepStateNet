#!/bin/bash
# quick_jobs.sh - Individual job submission helper
# For when you want to run specific experiments only

# Make the main scheduler executable
chmod +x tmux_scheduler.sh

case "${1:-}" in
    "dependent")
        echo "🧠 Submitting DEPENDENT experiment..."
        ./tmux_scheduler.sh submit dependent/ms.py "dependent_experiment" .venv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "independent")
        echo "🧠 Submitting INDEPENDENT experiment..."
        ./tmux_scheduler.sh submit independent/ms_indep.py "independent_experiment" .venv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "adaptive")
        echo "🧠 Submitting ADAPTIVE experiment..."
        ./tmux_scheduler.sh submit adaptive/ms_adapt.py "adaptive_experiment" .venv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "all-sequential")
        echo "🧠 Submitting ALL experiments (sequential - one after another)..."
        ./tmux_scheduler.sh submit dependent/ms.py "dependent_seq" .venv
        ./tmux_scheduler.sh submit independent/ms_indep.py "independent_seq" .venv
        ./tmux_scheduler.sh submit adaptive/ms_adapt.py "adaptive_seq" .venv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "all-parallel")
        echo "🧠 Submitting ALL experiments (parallel - all at once)..."
        ./tmux_scheduler.sh submit dependent/ms.py "dependent_par" .venv
        ./tmux_scheduler.sh submit independent/ms_indep.py "independent_par" .venv
        ./tmux_scheduler.sh submit adaptive/ms_adapt.py "adaptive_par" .venv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &"
        ;;
    "status")
        ./tmux_scheduler.sh status
        ;;
    "queue")
        ./tmux_scheduler.sh queue
        ;;
    *)
        echo "🧠 Quick job submission for microstate experiments"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Individual experiments:"
        echo "  $0 dependent      # Submit only dependent experiment"
        echo "  $0 independent    # Submit only independent experiment"
        echo "  $0 adaptive       # Submit only adaptive experiment"
        echo ""
        echo "Multiple experiments:"
        echo "  $0 all-sequential # Submit all (run one after another)"
        echo "  $0 all-parallel   # Submit all (run simultaneously)"
        echo ""
        echo "Monitoring:"
        echo "  $0 status         # Show job status"
        echo "  $0 queue          # Show job queue"
        echo ""
        echo "After submitting, start a worker:"
        echo "  nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &   # Sequential"
        echo "  nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &   # Parallel"
        echo ""
        echo "Monitor jobs:"
        echo "  ./tmux_scheduler.sh status"
        echo "  ./tmux_scheduler.sh monitor JOB_NAME"
        echo "  ./tmux_scheduler.sh logs JOB_NAME"
        ;;
esac