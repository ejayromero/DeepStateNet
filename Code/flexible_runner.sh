#!/bin/bash
# flexible_runner.sh - Universal experiment runner
# Uses configuration from experiment_config.sh

# Load configuration
source ./experiment_config.sh

echo "Flexible Experiment Runner"
echo "=========================="
echo "Configuration loaded: $EXPERIMENT_TYPE experiments"
echo "Execution mode: $EXECUTION_MODE"
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Verify we're in the right directory
if [[ ! -d "lib" || ! -f "tmux_scheduler.sh" ]]; then
    echo "Error: Please run this script from the Code/ directory"
    echo "Current directory: $(pwd)"
    echo "Expected: tmux_scheduler.sh and lib/ should exist here"
    exit 1
fi

# Make scheduler executable
chmod +x tmux_scheduler.sh

# Check if scripts exist (if enabled)
if [[ "$SCRIPT_CHECK_ENABLED" == "true" ]]; then
    echo "Checking if scripts exist..."
    scripts_to_check=("$SCRIPT_1" "$SCRIPT_2" "$SCRIPT_3")
    for script in "${scripts_to_check[@]}"; do
        if [[ -n "$script" && ! -f "$script" ]]; then
            echo "Error: Script not found: $script"
            echo "Please check your experiment_config.sh"
            exit 1
        fi
    done
    echo "✓ All scripts found"
fi

# Check conda environment
if [[ -n "$CONDA_ENV" ]]; then
    echo "Checking conda environment '$CONDA_ENV'..."
    if ! conda env list | grep -q "$CONDA_ENV"; then
        echo "Error: Conda environment '$CONDA_ENV' not found"
        echo "Available environments:"
        conda env list
        exit 1
    fi
    echo "✓ Conda environment '$CONDA_ENV' is available"
fi

echo ""
echo "Submitting jobs..."

# Submit jobs based on execution mode
case "$EXECUTION_MODE" in
    "parallel")
        echo "=== PARALLEL EXECUTION ==="
        echo "All jobs will run simultaneously"
        
        if [[ -n "$SCRIPT_1" ]]; then
            echo "1. Submitting: $JOB_NAME_1 ($SCRIPT_1)"
            ./tmux_scheduler.sh submit "$SCRIPT_1" "$JOB_NAME_1" "$CONDA_ENV" "$GPU_1"
        fi
        
        if [[ -n "$SCRIPT_2" ]]; then
            echo "2. Submitting: $JOB_NAME_2 ($SCRIPT_2)"
            ./tmux_scheduler.sh submit "$SCRIPT_2" "$JOB_NAME_2" "$CONDA_ENV" "$GPU_2"
        fi
        
        if [[ -n "$SCRIPT_3" ]]; then
            echo "3. Submitting: $JOB_NAME_3 ($SCRIPT_3)"
            ./tmux_scheduler.sh submit "$SCRIPT_3" "$JOB_NAME_3" "$CONDA_ENV" "$GPU_3"
        fi
        
        echo ""
        echo "Starting worker with $MAX_PARALLEL_JOBS parallel jobs..."
        nohup ./tmux_scheduler.sh worker $MAX_PARALLEL_JOBS > worker.log 2>&1 &
        worker_pid=$!
        ;;
        
    "sequential")
        echo "=== SEQUENTIAL EXECUTION ==="
        echo "Jobs will run one after another"
        
        if [[ -n "$SCRIPT_1" ]]; then
            echo "1. Submitting: $JOB_NAME_1 ($SCRIPT_1)"
            ./tmux_scheduler.sh submit "$SCRIPT_1" "$JOB_NAME_1" "$CONDA_ENV" "$GPU_1"
        fi
        
        if [[ -n "$SCRIPT_2" ]]; then
            echo "2. Submitting: $JOB_NAME_2 ($SCRIPT_2)"
            ./tmux_scheduler.sh submit "$SCRIPT_2" "$JOB_NAME_2" "$CONDA_ENV" "$GPU_2"
        fi
        
        if [[ -n "$SCRIPT_3" ]]; then
            echo "3. Submitting: $JOB_NAME_3 ($SCRIPT_3)"
            ./tmux_scheduler.sh submit "$SCRIPT_3" "$JOB_NAME_3" "$CONDA_ENV" "$GPU_3"
        fi
        
        echo ""
        echo "Starting sequential worker..."
        nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &
        worker_pid=$!
        ;;
        
    "dependency")
        echo "=== DEPENDENCY EXECUTION ==="
        echo "Scripts 1&2 will run in parallel, then script 3 after both complete"
        
        if [[ -n "$SCRIPT_1" ]]; then
            echo "1. Submitting: $JOB_NAME_1 ($SCRIPT_1)"
            ./tmux_scheduler.sh submit "$SCRIPT_1" "$JOB_NAME_1" "$CONDA_ENV" "$GPU_1"
        fi
        
        if [[ -n "$SCRIPT_2" ]]; then
            echo "2. Submitting: $JOB_NAME_2 ($SCRIPT_2)"
            ./tmux_scheduler.sh submit "$SCRIPT_2" "$JOB_NAME_2" "$CONDA_ENV" "$GPU_2"
        fi
        
        echo ""
        echo "Starting worker for first two jobs..."
        nohup ./tmux_scheduler.sh worker 2 > worker.log 2>&1 &
        worker_pid=$!
        
        if [[ -n "$SCRIPT_3" ]]; then
            echo ""
            echo "Starting dependency monitor for script 3..."
            # Start background script to wait for jobs 1&2 to complete
            nohup bash -c "
                echo 'Waiting for $JOB_NAME_1 and $JOB_NAME_2 to complete...'
                while true; do
                    status1=\$(grep '$JOB_NAME_1' .job_status 2>/dev/null | tail -1 | cut -d'|' -f3 || echo 'NOTFOUND')
                    status2=\$(grep '$JOB_NAME_2' .job_status 2>/dev/null | tail -1 | cut -d'|' -f3 || echo 'NOTFOUND')
                    
                    if [[ \"\$status1\" == \"COMPLETED\" && \"\$status2\" == \"COMPLETED\" ]]; then
                        echo 'Both prerequisite jobs completed! Submitting $JOB_NAME_3...'
                        ./tmux_scheduler.sh submit '$SCRIPT_3' '$JOB_NAME_3' '$CONDA_ENV' '$GPU_3'
                        break
                    elif [[ \"\$status1\" == \"FAILED\" || \"\$status2\" == \"FAILED\" ]]; then
                        echo 'One of the prerequisite jobs failed. Not submitting $JOB_NAME_3.'
                        break
                    fi
                    
                    sleep 30
                done
            " > dependency_monitor.log 2>&1 &
        fi
        ;;
        
    *)
        echo "Error: Unknown execution mode '$EXECUTION_MODE'"
        echo "Valid modes: parallel, sequential, dependency"
        exit 1
        ;;
esac

# Show current queue and status
echo ""
echo "=== CURRENT QUEUE ==="
./tmux_scheduler.sh queue

echo ""
echo "=== CURRENT STATUS ==="
./tmux_scheduler.sh status

echo ""
echo "=== WORKER STARTED ==="
echo "Worker PID: $worker_pid"
echo "Worker log: worker.log"
echo ""

echo "=== MONITORING COMMANDS ==="
echo "  ./tmux_scheduler.sh status           # Show job status"
echo "  ./tmux_scheduler.sh monitor JOB      # Watch job live"
echo "  ./tmux_scheduler.sh logs JOB         # View job output"
echo "  tail -f worker.log                   # Watch worker"
echo ""

echo "=== RESULTS LOCATION ==="
echo "  Results will be saved to: $RESULTS_FOLDER"
echo ""

echo "=== TO STOP ALL JOBS ==="
if [[ -n "$SCRIPT_1" ]]; then echo "  ./tmux_scheduler.sh kill $JOB_NAME_1"; fi
if [[ -n "$SCRIPT_2" ]]; then echo "  ./tmux_scheduler.sh kill $JOB_NAME_2"; fi
if [[ -n "$SCRIPT_3" ]]; then echo "  ./tmux_scheduler.sh kill $JOB_NAME_3"; fi
echo "  pkill -f tmux_scheduler               # Stop worker"
echo ""

echo "✓ All configured experiments submitted and running!"
echo ""
echo "TIP: You can disconnect from SSH - jobs will continue running"
echo "     When you reconnect, use: ./tmux_scheduler.sh status"