# Tmux Job Scheduler

A SLURM-like job management system using tmux for parallel execution of Python scripts on shared servers. No sudo access required.

## Overview

This scheduler allows you to submit multiple jobs to a queue and execute them in parallel using tmux sessions. Each job runs in its own isolated tmux session with automatic logging and cleanup.

## Features

- **Parallel Execution**: Run multiple jobs simultaneously
- **Job Queue Management**: Submit jobs to queue for sequential or parallel processing
- **Environment Support**: Supports both conda environments and Python virtual environments
- **GPU Assignment**: Assign specific GPUs to jobs
- **Automatic Logging**: All output saved with timestamps
- **Session Management**: Jobs run in persistent tmux sessions that survive SSH disconnections
- **Safe for Shared Servers**: Uses only user space, no system modifications required

## Directory Structure

```
your_project_folder/
├── Code/
│   ├── dependent/
│   │   └── ms.py
│   ├── independent/
│   │   └── ms_indep.py
│   ├── adaptive/
│   │   └── ms_adapt.py
│   ├── tmux_scheduler.sh       # Main scheduler
│   ├── run_all_ms_experiments.sh  # Custom runner
│   ├── quick_jobs.sh           # Individual job helper
│   ├── .venv/                  # Python virtual environment
│   └── lib/
│       └── my_models.py
├── Data/
└── Output/
```

## Installation

1. Navigate to your Code directory:
   ```bash
   cd your_project_folder/Code
   ```

2. Create the scheduler scripts:
   ```bash
   nano tmux_scheduler.sh          # Copy main scheduler content
   nano run_all_ms_experiments.sh # Copy custom runner content
   nano quick_jobs.sh              # Copy helper script content
   ```

3. Make scripts executable:
   ```bash
   chmod +x tmux_scheduler.sh run_all_ms_experiments.sh quick_jobs.sh
   ```

4. Create logs directory:
   ```bash
   mkdir -p ../.logs
   ```

## Usage

### Quick Start - Run All Experiments

Run all three microstate experiments in parallel:

```bash
./run_all_ms_experiments.sh
```

This will:
- Submit dependent/ms.py, independent/ms_indep.py, and adaptive/ms_adapt.py to the queue
- Start a worker to run 3 jobs simultaneously
- Create log files in `../.logs/`
- Allow safe SSH disconnection

### Manual Job Submission

#### Submit Individual Jobs

```bash
# Submit a single job
./tmux_scheduler.sh submit dependent/ms.py "dependent_exp" .venv

# Submit with GPU assignment
./tmux_scheduler.sh submit dependent/ms.py "dependent_exp" .venv 0

# Submit multiple jobs
./tmux_scheduler.sh submit dependent/ms.py "dep_exp" .venv
./tmux_scheduler.sh submit independent/ms_indep.py "indep_exp" .venv
./tmux_scheduler.sh submit adaptive/ms_adapt.py "adapt_exp" .venv
```

#### Start Worker

```bash
# Sequential execution (1 job at a time)
nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &

# Parallel execution (3 jobs simultaneously)
nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &
```

### Monitoring Jobs

#### Check Job Status
```bash
./tmux_scheduler.sh status
```

#### View Job Queue
```bash
./tmux_scheduler.sh queue
```

#### Monitor Job in Real-time
```bash
./tmux_scheduler.sh monitor job_name
# Detach with: Ctrl+B, then D
```

#### View Job Logs
```bash
./tmux_scheduler.sh logs job_name
```

#### Check Worker Status
```bash
tail -f worker.log
```

### Using Quick Jobs Helper

```bash
# Submit individual experiments
./quick_jobs.sh dependent     # Submit only dependent experiment
./quick_jobs.sh independent   # Submit only independent experiment
./quick_jobs.sh adaptive      # Submit only adaptive experiment

# Submit all experiments
./quick_jobs.sh all-sequential # Run one after another
./quick_jobs.sh all-parallel   # Run simultaneously

# Check status
./quick_jobs.sh status
./quick_jobs.sh queue
```

## Commands Reference

### Main Scheduler Commands

| Command | Description |
|---------|-------------|
| `submit SCRIPT JOB_NAME [ENV] [GPU]` | Submit job to queue |
| `run SCRIPT JOB_NAME [ENV] [GPU]` | Run job immediately |
| `status` | Show all job statuses |
| `queue` | Show current queue |
| `logs JOB_NAME` | Show logs for a job |
| `monitor JOB_NAME` | Monitor job in real-time |
| `kill JOB_NAME` | Kill a running job |
| `clean` | Clean completed jobs from status |
| `worker [MAX_PARALLEL]` | Start job worker |
| `set-parallel N` | Set default parallel limit |

### Job States

- **QUEUED**: Job submitted and waiting to run
- **RUNNING**: Job currently executing
- **COMPLETED**: Job finished successfully
- **FAILED**: Job terminated with errors
- **KILLED**: Job manually terminated

## Environment Support

### Python Virtual Environment (.venv)
```bash
./tmux_scheduler.sh submit script.py "job_name" .venv
```

### Conda Environment
```bash
./tmux_scheduler.sh submit script.py "job_name" my_conda_env
```

### Custom Virtual Environment Path
```bash
./tmux_scheduler.sh submit script.py "job_name" /path/to/venv
```

### No Environment
```bash
./tmux_scheduler.sh submit script.py "job_name"
```

## GPU Support

Assign specific GPUs to jobs:

```bash
# Use GPU 0
./tmux_scheduler.sh submit script.py "job_name" .venv 0

# Use GPU 1
./tmux_scheduler.sh submit script.py "job_name" .venv 1

# Auto-assign or use CPU
./tmux_scheduler.sh submit script.py "job_name" .venv
```

## File Locations

### Log Files
- Location: `../.logs/`
- Format: `JOB_NAME_YYYYMMDD_HHMMSS.log`
- Contains: Complete job output with timestamps

### Tracking Files
- `.job_queue`: Current job queue
- `.job_status`: Job status tracking
- `worker.log`: Worker process log

### Tmux Sessions
- Format: `job_JOB_NAME`
- Auto-created when jobs start
- Auto-removed when jobs complete

## Troubleshooting

### Check Running Jobs
```bash
./tmux_scheduler.sh status
tmux ls
```

### Restart Everything
```bash
# Kill worker
pkill -f tmux_scheduler

# Clear queues
> .job_queue
> .job_status

# Resubmit jobs
./run_all_ms_experiments.sh
```

### View All Sessions
```bash
tmux ls
```

### Kill Specific Session
```bash
tmux kill-session -t job_name
```

### Check System Resources
```bash
htop                    # CPU and memory usage
nvidia-smi             # GPU usage (if available)
df -h                  # Disk usage
```

## Safety Notes

This scheduler is designed for shared servers and:

- Uses only user space (no sudo required)
- Creates files only in your directories
- Uses standard tmux (usually pre-installed)
- Does not modify system configurations
- Does not affect other users
- Resource usage equivalent to running scripts manually

## Examples

### Microstate Experiments (Recommended)

```bash
# Run all three experiments in parallel
./run_all_ms_experiments.sh

# Monitor progress
./tmux_scheduler.sh status
./tmux_scheduler.sh monitor dependent_50subj
```

### Custom Workflow

```bash
# Submit jobs with different priorities
./tmux_scheduler.sh submit dependent/ms.py "high_priority" .venv 0
./tmux_scheduler.sh submit analysis/postprocess.py "low_priority" .venv

# Start worker with limited parallelism
nohup ./tmux_scheduler.sh worker 2 > worker.log 2>&1 &

# Monitor and manage
./tmux_scheduler.sh status
./tmux_scheduler.sh logs high_priority
```

## Advanced Usage

### Set Default Parallel Limit
```bash
./tmux_scheduler.sh set-parallel 4
```

### Clean Completed Jobs
```bash
./tmux_scheduler.sh clean
```

### Run Jobs Immediately (Bypass Queue)
```bash
./tmux_scheduler.sh run script.py "urgent_job" .venv
```

### Background Worker with Logging
```bash
nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &
tail -f worker.log
```

## Support

For issues or questions:
1. Check job status: `./tmux_scheduler.sh status`
2. View logs: `./tmux_scheduler.sh logs JOB_NAME`
3. Check worker log: `tail -f worker.log`
4. Verify tmux sessions: `tmux ls`