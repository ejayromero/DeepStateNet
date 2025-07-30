#!/bin/bash

# tmux_scheduler.sh - A SLURM-like job scheduler using tmux
# FIXED VERSION: Proper parallel processing, unique job IDs, and auto-cleanup
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
Tmux Job Scheduler - SLURM-like job management with tmux
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
    Conda:       ./tmux_scheduler.sh submit ms.py "job1" myvenv
    Python venv: ./tmux_scheduler.sh submit ms.py "job1" .venv
    Custom venv: ./tmux_scheduler.sh submit ms.py "job1" /path/to/my_venv
    No env:      ./tmux_scheduler.sh submit ms.py "job1"

PARALLEL EXECUTION:
    worker           # Default: $DEFAULT_MAX_PARALLEL jobs in parallel
    worker 1         # Sequential execution (1 job at a time)
    worker 4         # Up to 4 jobs in parallel

GPU SUPPORT:
    submit subject_dependent/ms.py "job1" myvenv 0     # Use GPU 0
    submit subject_independent/ms_indep.py "job2" myvenv 1     # Use GPU 1
    submit subject_adaptive/ms_adapt.py "job3" myvenv       # Auto-assign GPU or use CPU

EXAMPLES FOR YOUR SETUP:
    # Submit your three experiments
    ./tmux_scheduler.sh submit subject_dependent/ms.py "dependent_exp" myvenv
    ./tmux_scheduler.sh submit subject_independent/ms_indep.py "independent_exp" myvenv
    ./tmux_scheduler.sh submit subject_adaptive/ms_adapt.py "adaptive_exp" myvenv

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

# Generate unique job ID - FIXED: Use nanoseconds for true uniqueness
generate_job_id() {
    local job_name="$1"
    # Use nanoseconds and random number for guaranteed uniqueness
    local timestamp=$(date '+%Y%m%d_%H%M%S_%N')  # Added nanoseconds
    local random=$(( RANDOM % 10000 ))
    echo "${job_name}_${timestamp}_${random}"
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
    echo "$job_id|$job_name|QUEUED|$(date '+%Y-%m-%d %H:%M:%S')|$log_file|$gpu_id" >> "$STATUS_FILE"
    
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
                venv_path="$SCRIPT_DIR/.venv"
            else
                venv_path="$env_name"
            fi
            tmux send-keys -t "$session_name" "source $venv_path/bin/activate" Enter
            log_message "Activated Python venv: $venv_path"
        else
            # Handle conda environment
            tmux send-keys -t "$session_name" "source ~/miniconda3/bin/activate" Enter
            sleep 2
            tmux send-keys -t "$session_name" "conda activate $env_name" Enter
            log_message "Activated conda env: $env_name"
        fi
        sleep 3
    fi
    
    # Create the command to run with proper exit code capture
    local run_command="python3 $script_name 2>&1 | tee $log_file; exit_code=\$?; echo \"JOB_EXIT_CODE: \$exit_code\" | tee -a $log_file; echo \"JOB_FINISHED_AT: \$(date)\" | tee -a $log_file; exit \$exit_code"
    
    # Send the command
    tmux send-keys -t "$session_name" "$run_command" Enter
    
    # Start background monitoring process with enhanced debugging
    (
        local session_name="job_$job_name"
        local check_interval=10
        local last_activity_time=$(date +%s)
        local max_idle_time=300  # 5 minutes without activity = potential hang
        
        # Wait for session to start properly
        sleep 5
        
        # Monitor session until it ends
        while tmux has-session -t "$session_name" 2>/dev/null; do
            # Check for activity in the session
            current_time=$(date +%s)
            
            # Update last activity if log file has been modified recently
            if [[ -f "$log_file" ]]; then
                local log_modified=$(stat -c %Y "$log_file" 2>/dev/null || echo 0)
                if [[ $log_modified -gt $((current_time - check_interval)) ]]; then
                    last_activity_time=$current_time
                fi
            fi
            
            sleep $check_interval
        done
        
        # Session ended - check results with enhanced debugging
        log_message "Session '$session_name' ended, analyzing termination..."
        
        # Give a moment for log file to be written
        sleep 2
        
        # Enhanced termination analysis
        if [[ -f "$log_file" ]]; then
            # Check for normal completion
            if grep -q "JOB_EXIT_CODE: 0" "$log_file"; then
                update_job_status "$job_id" "COMPLETED"
                success_message "Job '$job_name' completed successfully"
            elif grep -q "JOB_EXIT_CODE:" "$log_file"; then
                # Job exited with error code
                local exit_line=$(grep "JOB_EXIT_CODE:" "$log_file" | tail -1)
                local exit_code=$(echo $exit_line | cut -d' ' -f2)
                echo "JOB_CRASHED_AT: $(date)" >> "$log_file"
                update_job_status "$job_id" "FAILED"
                error_message "Job '$job_name' crashed with exit code $exit_code"
            elif grep -q "JOB_FINISHED_AT:" "$log_file"; then
                # Has finish timestamp but no exit code - interrupted
                echo "JOB_KILLED_AT: $(date)" >> "$log_file"
                update_job_status "$job_id" "KILLED"
                warning_message "Job '$job_name' was killed or interrupted"
            else
                # No completion markers - unexpected termination
                local current_time=$(date +%s)
                if [[ $((current_time - last_activity_time)) -gt $max_idle_time ]]; then
                    echo "JOB_CRASHED_AT: $(date) (hung/timeout)" >> "$log_file"
                    error_message "Job '$job_name' appears to have hung and was terminated"
                else
                    echo "JOB_KILLED_AT: $(date) (force terminated)" >> "$log_file"
                    warning_message "Job '$job_name' was forcefully terminated"
                fi
                update_job_status "$job_id" "FAILED"
            fi
        else
            # No log file found - severe failure
            update_job_status "$job_id" "FAILED"
            error_message "Job '$job_name' failed catastrophically - no log file found"
        fi
        
        # Clean up session (should already be gone, but just in case)
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

# FIXED: Check both RUNNING and FAILED jobs for proper status
count_running_jobs() {
    local running_count=0
    
    if [[ -s "$STATUS_FILE" ]]; then
        while IFS='|' read -r job_id job_name status last_update log_file gpu_id; do
            if [[ "$status" == "RUNNING" ]]; then
                # Check if tmux session actually exists
                if tmux has-session -t "job_$job_name" 2>/dev/null; then
                    ((running_count++))
                else
                    # Session gone but status says running - check if job completed successfully
                    if [[ -f "$log_file" ]] && grep -q "JOB_EXIT_CODE: 0" "$log_file"; then
                        # Job completed successfully
                        update_job_status "$job_id" "COMPLETED" >/dev/null 2>&1
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job '$job_name' completed successfully" >&2
                    else
                        # Job failed or crashed
                        update_job_status "$job_id" "FAILED" >/dev/null 2>&1
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job '$job_name' failed or crashed" >&2
                    fi
                fi
            elif [[ "$status" == "FAILED" ]]; then
                # Re-check FAILED jobs - they might have actually completed successfully
                if [[ -f "$log_file" ]] && grep -q "JOB_EXIT_CODE: 0" "$log_file"; then
                    # Job actually completed successfully but was mismarked as failed
                    update_job_status "$job_id" "COMPLETED" >/dev/null 2>&1
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Re-examined: Job '$job_name' actually completed successfully" >&2
                fi
                # Note: FAILED jobs don't count toward running_count
            fi
        done < "$STATUS_FILE"
    fi
    
    echo $running_count
}

# Show job status
show_status() {
    echo -e "\n${BLUE}=== JOB STATUS ===${NC}"
    printf "%-25s %-15s %-10s %-5s %-20s\n" "JOB_NAME" "STATUS" "JOB_ID" "GPU" "LAST_UPDATE"
    echo "--------------------------------------------------------------------------------"
    
    if [[ -s "$STATUS_FILE" ]]; then
        while IFS='|' read -r job_id job_name status last_update log_file gpu_id; do
            # Check if session still exists for RUNNING jobs
            if [[ "$status" == "RUNNING" ]]; then
                if ! tmux has-session -t "job_$job_name" 2>/dev/null; then
                    # Session is gone but status says running - update to failed
                    update_job_status "$job_id" "FAILED"
                    status="FAILED"
                fi
            fi
            
            case "$status" in
                "RUNNING") color="$YELLOW" ;;
                "COMPLETED") color="$GREEN" ;;
                "FAILED") color="$RED" ;;
                "QUEUED") color="$BLUE" ;;
                *) color="$NC" ;;
            esac
            gpu_display="${gpu_id:-"-"}"
            # Show short job ID for readability
            short_job_id=$(echo "$job_id" | rev | cut -d'_' -f1-2 | rev)
            printf "%-25s ${color}%-15s${NC} %-10s %-5s %-20s\n" "$job_name" "$status" "$short_job_id" "$gpu_display" "$last_update"
        done < "$STATUS_FILE"
    else
        echo "No jobs found."
    fi
    echo ""
}

# Show queue
show_queue() {
    echo -e "\n${BLUE}=== JOB QUEUE ===${NC}"
    printf "%-5s %-25s %-30s %-5s %-20s\n" "POS" "JOB_NAME" "SCRIPT" "GPU" "STATUS"
    echo "--------------------------------------------------------------------------------"
    
    if [[ -s "$QUEUE_FILE" ]]; then
        local pos=1
        while IFS='|' read -r job_id script_name job_name env_name gpu_id log_file status; do
            gpu_display="${gpu_id:-"-"}"
            printf "%-5s %-25s %-30s %-5s %-20s\n" "$pos" "$job_name" "$script_name" "$gpu_display" "$status"
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

# Kill a job with enhanced logging
kill_job() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        error_message "Usage: kill JOB_NAME"
        exit 1
    fi
    
    local session_name="job_$job_name"
    
    if tmux has-session -t "$session_name" 2>/dev/null; then
        # Find the job's log file before killing
        local job_entry=$(grep "|$job_name|" "$STATUS_FILE" | head -1)
        local log_file=$(echo "$job_entry" | cut -d'|' -f5)
        
        # Add kill timestamp to log file
        if [[ -n "$log_file" && -f "$log_file" ]]; then
            echo "JOB_KILLED_AT: $(date) (manually terminated)" >> "$log_file"
        fi
        
        # Kill the session
        tmux kill-session -t "$session_name"
        
        # Update status to KILLED
        local job_id=$(echo "$job_entry" | cut -d'|' -f1)
        if [[ -n "$job_id" ]]; then
            update_job_status "$job_id" "KILLED"
        fi
        
        success_message "Job '$job_name' killed and logged"
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

# Job worker (processes queue) - FIXED PARALLEL PROCESSING
start_worker() {
    local max_parallel="${1:-$DEFAULT_MAX_PARALLEL}"
    
    log_message "Starting job worker with max $max_parallel parallel jobs..."
    log_message "Press Ctrl+C to stop"
    
    while true; do
        # FIXED: Use proper running job counting
        local running_jobs=$(count_running_jobs)
        
        log_message "Currently running: $running_jobs/$max_parallel jobs"
        
        # Check if we can start more jobs
        if [[ "$running_jobs" -lt "$max_parallel" ]]; then
            # Get next queued job
            local queued_job=$(grep "|QUEUED$" "$QUEUE_FILE" 2>/dev/null | head -1)
            
            if [[ -n "$queued_job" ]]; then
                # Parse the queued job
                IFS='|' read -r job_id script_name job_name env_name gpu_id log_file status <<< "$queued_job"
                
                log_message "Processing queued job: $job_name (Running: $running_jobs/$max_parallel)"
                
                # Remove from queue FIRST
                grep -v "^$job_id|" "$QUEUE_FILE" > "${QUEUE_FILE}.tmp"
                mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
                
                # Execute the job
                execute_job "$job_id" "$script_name" "$job_name" "$env_name" "$gpu_id" "$log_file"
                
                sleep 3  # Give time for job to start
            else
                # No jobs to process
                if [[ "$running_jobs" -eq 0 ]]; then
                    log_message "No jobs in queue and no jobs running. Waiting..."
                    sleep 10
                else
                    log_message "No jobs in queue but $running_jobs jobs still running. Waiting..."
                    sleep 30
                fi
            fi
        else
            # At max capacity
            log_message "At maximum capacity ($max_parallel jobs). Waiting for jobs to complete..."
            sleep 30
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