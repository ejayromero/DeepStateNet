#!/bin/bash
# quick_jobs.sh - Individual job submission helper
# Updated for conda environment (myvenv) and correct folder structure

# Make the main scheduler executable
chmod +x tmux_scheduler.sh

case "${1:-}" in
    "dependent")
        echo "Submitting SUBJECT DEPENDENT experiment..."
        ./tmux_scheduler.sh submit subject_dependent/ms.py "subject_dependent_experiment" myvenv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "independent")
        echo "Submitting SUBJECT INDEPENDENT experiment..."
        ./tmux_scheduler.sh submit subject_independent/ms_indep.py "subject_independent_experiment" myvenv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "adaptive")
        echo "Submitting SUBJECT ADAPTIVE experiment..."
        ./tmux_scheduler.sh submit subject_adaptive/ms_adapt.py "subject_adaptive_experiment" myvenv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "all-sequential")
        echo "Submitting ALL experiments (sequential - one after another)..."
        ./tmux_scheduler.sh submit subject_dependent/ms.py "subject_dependent_seq" myvenv
        ./tmux_scheduler.sh submit subject_independent/ms_indep.py "subject_independent_seq" myvenv
        ./tmux_scheduler.sh submit subject_adaptive/ms_adapt.py "subject_adaptive_seq" myvenv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &"
        ;;
    "all-parallel")
        echo "Submitting ALL experiments (parallel - all at once)..."
        ./tmux_scheduler.sh submit subject_dependent/ms.py "subject_dependent_par" myvenv
        ./tmux_scheduler.sh submit subject_independent/ms_indep.py "subject_independent_par" myvenv
        ./tmux_scheduler.sh submit subject_adaptive/ms_adapt.py "subject_adaptive_par" myvenv
        echo "Start worker with: nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &"
        ;;
    "status")
        ./tmux_scheduler.sh status
        ;;
    "queue")
        ./tmux_scheduler.sh queue
        ;;
    *)
        echo "Quick job submission for microstate experiments"
        echo "Updated for conda environment 'myvenv'"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Individual experiments:"
        echo "  $0 dependent      # Submit only subject_dependent experiment"
        echo "  $0 independent    # Submit only subject_independent experiment"
        echo "  $0 adaptive       # Submit only subject_adaptive experiment"
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
        echo ""
        echo "Environment: Uses conda environment 'myvenv'"
        echo "Requirements: Python 3.11 with torch, numpy, pandas, matplotlib, seaborn, scikit-learn"
        ;;
esac