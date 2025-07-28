#!/bin/bash

# tmux_scheduler.sh - A SLURM-like job scheduler using tmux
# Usage: ./tmux_scheduler.sh [command] [options]
# Safe for shared servers - uses only user space, no sudo required

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../.logs"
QUEUE_FILE="$SCRIPT_DIR/.job_queue"
STATUS_FILE="$SCRIPT_DIR/.job_status"
CONFIG_FILE="$SCRIPT_DIR/job_config.conf"
DEFAULT_MAX_PARALLEL=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p "$LOG_DIR"
touch "$QUEUE_FILE" "$STATUS_FILE"

# Functions
print_help() {
    cat << EOF
🧠 Tmux Job Scheduler - SLURM-like job management with tmux
Safe for shared servers - uses only your user space!

USAGE:
    ./tmux_scheduler.sh COMMAND [OPTIONS]

COMMANDS:
    submit SCRIPT_NAME JOB_NAME [ENV] [GPU_ID]    Submit a job to queue
    run SCRIPT_NAME JOB_NAME [ENV] [GPU_ID]       Run job immediately (bypass queue)
    status                                        Show all job statuses
    queue                                         Show current queue
    logs JOB_NAME                                Show logs for a job
    kill JOB_NAME                                Kill a running job
    clean                                        Clean completed jobs from status
    monitor JOB_NAME                             Monitor job in real-time
    worker [MAX_PARALLEL]                        Start job worker (default: $DEFAULT_MAX_PARALLEL parallel jobs)
    set-parallel N                               Set default parallel job limit

ENVIRONMENT SUPPORT:
    Conda:       ./tmux_scheduler.sh submit ms.py "job1" my_conda_env
    Python venv: ./tmux_scheduler.sh submit ms.py "job1" .venv
    Custom venv: ./tmux_scheduler.sh submit ms.py "job1" /path/to/my_venv
    No env:      ./tmux_scheduler.sh submit ms.py "job1"

PARALLEL EXECUTION:
    worker           # Default: $DEFAULT_MAX_PARALLEL jobs in parallel
    worker 1         # Sequential execution (1 job at a time)
    worker 4         # Up to 4 jobs in parallel

GPU SUPPORT:
    submit dependent/ms.py "job1" .venv 0     # Use GPU 0
    submit independent/ms_indep.py "job2" .venv 1     # Use GPU 1
    submit adaptive/ms_adapt.py "job3" .venv       # Auto-assign GPU or use CPU

EXAMPLES FOR YOUR SETUP:
    # Submit your three experiments
    ./tmux_scheduler.sh submit dependent/ms.py "dependent_exp" .venv
    ./tmux_scheduler.sh submit independent/ms_indep.py "independent_exp" .venv
    ./tmux_scheduler.sh submit adaptive/ms_adapt.py "adaptive_exp" .venv

    # Start worker with 3 parallel jobs
    nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &

LOGS:
    All logs saved to: ../.logs/JOB_NAME_YYYYMMDD_HHMMSS.log

SESSIONS:
    Each job runs in: tmux session "job_JOB_NAME"
    Sessions auto-close when job completes

SAFETY:
    ✅ Uses only your user space
    ✅ No sudo required
    ✅ No system-wide changes
    ✅ Safe for shared servers
EOF
}

log_message() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error_message() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success_message() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning_message() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Generate unique job ID
generate_job_id() {
    local job_name="$1"
    echo "${job_name}_$(date '+%Y%m%d_%H%M%S')"
}

# Add job to queue
submit_job() {
    local script_name="$1"
    local job_name="$2"
    local env_name="$3"
    local gpu_id="$4"
    
    if [[ -z "$script_name" || -z "$job_name" ]]; then
        error_message "Usage: submit SCRIPT_NAME JOB_NAME [ENV] [GPU_ID]"
        exit 1
    fi
    
    if [[ ! -f "$script_name" ]]; then
        error_message "Script '$script_name' not found!"
        exit 1
    fi
    
    local job_id=$(generate_job_id "$job_name")
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="$LOG_DIR/${job_name}_${timestamp}.log"
    
    # Add to queue
    echo "$job_id|$script_name|$job_name|$env_name|$gpu_id|$log_file|QUEUED" >> "$QUEUE_FILE"
    
    # Add to status
    echo "$job_id|$job_name|QUEUED|$(date)|$log_file|$gpu_id" >> "$STATUS_FILE"
    
    success_message "Job '$job_name' (ID: $job_id) submitted to queue"
    if [[ -n "$gpu_id" ]]; then
        log_message "Assigned GPU: $gpu_id"
    fi
    log_message "Log will be saved to: $log_file"
}

# Run job immediately
run_job() {
    local script_name="$1"
    local job_name="$2"
    local env_name="$3"
    local gpu_id="$4"
    
    if [[ -z "$script_name" || -z "$job_name" ]]; then
        error_message "Usage: run SCRIPT_NAME JOB_NAME [ENV] [GPU_ID]"
        exit 1
    fi
    
    if [[ ! -f "$script_name" ]]; then
        error_message "Script '$script_name' not found!"
        exit 1
    fi
    
    local job_id=$(generate_job_id "$job_name")
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="$LOG_DIR/${job_name}_${timestamp}.log"
    
    execute_job "$job_id" "$script_name" "$job_name" "$env_name" "$gpu_id" "$log_file"
}

# Execute a job in tmux
execute_job() {
    local job_id="$1"
    local script_name="$2"
    local job_name="$3"
    local env_name="$4"
    local gpu_id="$5"
    local log_file="$6"
    
    local session_name="job_$job_name"
    
    log_message "Starting job '$job_name' (ID: $job_id)"
    log_message "Session: $session_name"
    if [[ -n "$gpu_id" ]]; then
        log_message "GPU: $gpu_id"
    fi
    log_message "Log file: $log_file"
    
    # Update status to RUNNING
    update_job_status "$job_id" "RUNNING"
    
    # Create tmux session and run job
    tmux new-session -d -s "$session_name" -c "$SCRIPT_DIR"
    
    # Set GPU if specified
    if [[ -n "$gpu_id" ]]; then
        tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$gpu_id" Enter
        sleep 1
    fi
    
    # Set up environment if specified
    if [[ -n "$env_name" ]]; then
        if [[ "$env_name" == .venv* ]] || [[ "$env_name" == */.venv* ]]; then
            # Handle .venv (Python virtual environment)
            if [[ "$env_name" == ".venv" ]]; then
                # Relative path from script directory
                venv_path="$SCRIPT_DIR/.venv"
            else
                # Absolute or custom path
                venv_path="$env_name"
            fi
            
            tmux send-keys -t "$session_name" "source $venv_path/bin/activate" Enter
            log_message "Activated Python venv: $venv_path"
        else
            # Handle conda environment
            tmux send-keys -t "$session_name" "conda activate $env_name" Enter
            log_message "Activated conda env: $env_name"
        fi
        sleep 2
    fi
    
    # Create the command to run
    local run_command="python $script_name 2>&1 | tee $log_file"
    
    # Send the command
    tmux send-keys -t "$session_name" "$run_command" Enter
    
    # Set up a monitoring script that runs after the main command
    tmux send-keys -t "$session_name" "echo 'Job completed with exit code: \$?'" Enter
    tmux send-keys -t "$session_name" "echo 'Job finished at: \$(date)'" Enter
    
    # Schedule session cleanup and status update
    (
        # Wait for the session to finish
        while tmux has-session -t "$session_name" 2>/dev/null; do
            sleep 10
        done
        
        # Check if job completed successfully
        if tail -5 "$log_file" | grep -q "Job completed with exit code: 0"; then
            update_job_status "$job_id" "COMPLETED"
            success_message "Job '$job_name' completed successfully"
        else
            update_job_status "$job_id" "FAILED" 
            error_message "Job '$job_name' failed"
        fi
        
        # Clean up session
        tmux kill-session -t "$session_name" 2>/dev/null || true
        
    ) &
    
    success_message "Job '$job_name' started in tmux session '$session_name'"
    log_message "Monitor with: ./tmux_scheduler.sh monitor $job_name"
    log_message "View logs with: ./tmux_scheduler.sh logs $job_name"
}

# Update job status
update_job_status() {
    local job_id="$1"
    local new_status="$2"
    
    # Create temp file with updated status
    awk -F'|' -v jid="$job_id" -v status="$new_status" '
        $1 == jid { $3 = status; $4 = strftime("%Y-%m-%d %H:%M:%S") }
        { print $1"|"$2"|"$3"|"$4"|"$5"|"$6 }
    ' "$STATUS_FILE" > "${STATUS_FILE}.tmp"
    
    mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
}

# Show job status
show_status() {
    echo -e "\n${BLUE}=== JOB STATUS ===${NC}"
    printf "%-20s %-15s %-10s %-5s %-20s\n" "JOB_NAME" "STATUS" "JOB_ID" "GPU" "LAST_UPDATE"
    echo "--------------------------------------------------------------------------------"
    
    if [[ -s "$STATUS_FILE" ]]; then
        while IFS='|' read -r job_id job_name status last_update log_file gpu_id; do
            case "$status" in
                "RUNNING") color="$YELLOW" ;;
                "COMPLETED") color="$GREEN" ;;
                "FAILED") color="$RED" ;;
                "QUEUED") color="$BLUE" ;;
                *) color="$NC" ;;
            esac
            gpu_display="${gpu_id:-"-"}"
            printf "%-20s ${color}%-15s${NC} %-10s %-5s %-20s\n" "$job_name" "$status" "${job_id##*_}" "$gpu_display" "$last_update"
        done < "$STATUS_FILE"
    else
        echo "No jobs found."
    fi
    echo ""
}

# Show queue
show_queue() {
    echo -e "\n${BLUE}=== JOB QUEUE ===${NC}"
    printf "%-5s %-20s %-25s %-5s %-20s\n" "POS" "JOB_NAME" "SCRIPT" "GPU" "STATUS"
    echo "--------------------------------------------------------------------------------"
    
    if [[ -s "$QUEUE_FILE" ]]; then
        local pos=1
        while IFS='|' read -r job_id script_name job_name env_name gpu_id log_file status; do
            gpu_display="${gpu_id:-"-"}"
            printf "%-5s %-20s %-25s %-5s %-20s\n" "$pos" "$job_name" "$script_name" "$gpu_display" "$status"
            ((pos++))
        done < "$QUEUE_FILE"
    else
        echo "Queue is empty."
    fi
    echo ""
}

# Show logs
show_logs() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        error_message "Usage: logs JOB_NAME"
        exit 1
    fi
    
    # Find the most recent log file for this job
    local log_file=$(find "$LOG_DIR" -name "${job_name}_*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        log_message "Showing logs for job '$job_name': $log_file"
        echo "--------------------------------------------------------------------------------"
        cat "$log_file"
    else
        error_message "No log file found for job '$job_name'"
        echo "Available log files:"
        ls -la "$LOG_DIR" | grep "$job_name" || echo "None found."
    fi
}

# Monitor job in real-time
monitor_job() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        error_message "Usage: monitor JOB_NAME"
        exit 1
    fi
    
    local session_name="job_$job_name"
    
    # Check if session exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        log_message "Attaching to job '$job_name' session..."
        log_message "Detach with: Ctrl+B, then D"
        sleep 2
        tmux attach-session -t "$session_name"
    else
        warning_message "No active session found for job '$job_name'"
        echo "Job may have completed or not started yet."
        echo "Check status with: ./tmux_scheduler.sh status"
        echo "View logs with: ./tmux_scheduler.sh logs $job_name"
    fi
}

# Kill a job
kill_job() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        error_message "Usage: kill JOB_NAME"
        exit 1
    fi
    
    local session_name="job_$job_name"
    
    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
        
        # Update status to KILLED
        local job_id=$(grep "|$job_name|" "$STATUS_FILE" | head -1 | cut -d'|' -f1)
        if [[ -n "$job_id" ]]; then
            update_job_status "$job_id" "KILLED"
        fi
        
        success_message "Job '$job_name' killed"
    else
        warning_message "No active session found for job '$job_name'"
    fi
}

# Clean completed jobs from status
clean_jobs() {
    log_message "Cleaning completed and failed jobs from status..."
    
    grep -v -E "COMPLETED|FAILED|KILLED" "$STATUS_FILE" > "${STATUS_FILE}.tmp" || touch "${STATUS_FILE}.tmp"
    mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
    
    success_message "Cleaned completed jobs from status"
}

# Job worker (processes queue)
start_worker() {
    local max_parallel="${1:-$DEFAULT_MAX_PARALLEL}"
    
    log_message "Starting job worker with max $max_parallel parallel jobs..."
    log_message "Press Ctrl+C to stop"
    
    while true; do
        # Count currently running jobs
        local running_jobs=$(grep "|RUNNING|" "$STATUS_FILE" 2>/dev/null | wc -l)
        
        # Check if we can start more jobs
        if [[ "$running_jobs" -lt "$max_parallel" ]]; then
            # Get next queued job
            local queued_job=$(grep "|QUEUED$" "$QUEUE_FILE" 2>/dev/null | head -1)
            
            if [[ -n "$queued_job" ]]; then
                # Parse the queued job
                IFS='|' read -r job_id script_name job_name env_name gpu_id log_file status <<< "$queued_job"
                
                log_message "Processing queued job: $job_name (Running: $running_jobs/$max_parallel)"
                
                # Remove from queue
                grep -v "^$job_id|" "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
                mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
                
                # Execute the job
                execute_job "$job_id" "$script_name" "$job_name" "$env_name" "$gpu_id" "$log_file"
                
                sleep 2  # Brief pause between job starts
            else
                sleep 10  # No jobs to process, wait longer
            fi
        else
            sleep 30  # At max capacity, check every 30 seconds
        fi
    done
}

# Set default parallel job limit
set_parallel_limit() {
    local new_limit="$1"
    
    if [[ -z "$new_limit" || ! "$new_limit" =~ ^[0-9]+$ ]]; then
        error_message "Usage: set-parallel NUMBER"
        exit 1
    fi
    
    # Update the config (simple approach - just modify this script's default)
    sed -i "s/DEFAULT_MAX_PARALLEL=.*/DEFAULT_MAX_PARALLEL=$new_limit/" "$0"
    
    success_message "Default parallel job limit set to $new_limit"
    log_message "Restart worker to apply changes"
}

# Main script logic
case "${1:-}" in
    "submit")
        submit_job "$2" "$3" "$4" "$5"
        ;;
    "run")
        run_job "$2" "$3" "$4" "$5"
        ;;
    "status")
        show_status
        ;;
    "queue")
        show_queue
        ;;
    "logs")
        show_logs "$2"
        ;;
    "monitor")
        monitor_job "$2"
        ;;
    "kill")
        kill_job "$2"
        ;;
    "clean")
        clean_jobs
        ;;
    "worker")
        start_worker "$2"
        ;;
    "set-parallel")
        set_parallel_limit "$2"
        ;;
    "help"|"-h"|"--help"|"")
        print_help
        ;;
    *)
        error_message "Unknown command: $1"
        echo "Use './tmux_scheduler.sh help' for usage information"
        exit 1
        ;;
esac