# Tmux Job Scheduler - Complete User Guide

A flexible, SLURM-like job scheduler using tmux sessions. Perfect for running machine learning experiments on shared servers without requiring admin privileges.

## Table of Contents

- [Overview](#overview)
  - [What is This System?](#what-is-this-system)
  - [Key Components](#key-components)
  - [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Your First Job](#your-first-job)
- [Experiment Presets](#experiment-presets)
  - [Basic Experiments](#basic-experiments)
  - [Combined Workflows](#combined-workflows)
  - [Custom Experiments](#custom-experiments)
- [Execution Modes](#execution-modes)
- [Normal Operation Guide](#normal-operation-guide)
  - [Step 1: Choose Your Experiment](#step-1-choose-your-experiment)
  - [Step 2: Run the Experiments](#step-2-run-the-experiments)
  - [Step 3: Monitor Progress](#step-3-monitor-progress)
- [Monitoring & Management](#monitoring--management)
  - [Check Job Status](#check-job-status)
  - [View Logs](#view-logs)
  - [Monitor Jobs Live](#monitor-jobs-live)
  - [Queue Management](#queue-management)
- [Troubleshooting & Problem Handling](#troubleshooting--problem-handling)
  - [Common Issues](#common-issues)
  - [Emergency Procedures](#emergency-procedures)
  - [Worker Management](#worker-management)
- [Advanced Usage](#advanced-usage)
  - [GPU Assignment](#gpu-assignment)
  - [Environment Management](#environment-management)
  - [Custom Configurations](#custom-configurations)
- [File Structure](#file-structure)
- [Safety Features](#safety-features)

---

## Overview

### What is This System?

This is a **job scheduling system** that uses tmux sessions to run your Python experiments safely on shared servers. Think of it as a lightweight alternative to SLURM that:

- **Requires no admin privileges** - runs entirely in your user space
- **Survives SSH disconnections** - jobs continue running when you disconnect
- **Manages job queues** - automatically runs jobs when resources are available
- **Provides easy monitoring** - check status, view logs, monitor live progress
- **Handles dependencies** - run jobs in sequence or after others complete

### Key Components

| Component | Purpose | What It Does |
|-----------|---------|-------------|
| **tmux_scheduler.sh** | Core scheduler engine | Manages job queue, creates tmux sessions, handles job lifecycle |
| **Worker** | Job processor | Background process that takes jobs from queue and runs them |
| **Jobs** | Your experiments | Individual Python scripts running in isolated tmux sessions |
| **Queue** | Job waiting list | Holds jobs waiting to be executed by the worker |
| **quick_switch.sh** | Configuration manager | Instantly switch between experiment presets |
| **flexible_runner.sh** | Universal launcher | Reads config and submits jobs to scheduler |
| **experiment_config.sh** | Settings file | Defines which scripts to run and how |

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  quick_switch   │───▶│ experiment_config │───▶│  flexible_runner    │
│  Set presets    │    │  Store settings   │    │  Submit jobs        │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│     Worker      │◀───│  tmux_scheduler  │◀───│     Job Queue       │
│ Process queue   │    │  Core engine     │    │  Waiting jobs       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Tmux Sessions                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐      │
│  │ job_experiment1 │ │ job_experiment2 │ │ job_experiment3 │ ...  │
│  │   Running       │ │   Running       │ │   Running       │      │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Setup

1. **Make files executable:**
```bash
chmod +x tmux_scheduler.sh experiment_config.sh flexible_runner.sh quick_switch.sh
```

2. **Verify your environment:**
```bash
# Check conda environment exists
conda env list | grep myvenv

# Test tmux is available
tmux -V
```

### Your First Job

```bash
# 1. Choose microstate experiments in parallel mode
./quick_switch.sh ms parallel

# 2. Run the experiments
./flexible_runner.sh

# 3. Check status
./tmux_scheduler.sh status
```

That's it! Your experiments are now running in the background.

---

## Experiment Presets

### Basic Experiments

| Preset | Scripts | Description |
|--------|---------|-------------|
| **`ms`** | `ms.py`, `ms_indep.py`, `ms_adapt.py` | Microstate analysis experiments |
| **`dcn`** | `dcn.py`, `dcn_indep.py`, `dcn_adapt.py` | Deep Convolutional Network experiments |

**Examples:**
```bash
# Run all microstate experiments in parallel
./quick_switch.sh ms parallel
./flexible_runner.sh

# Run DCN experiments sequentially (one after another)
./quick_switch.sh dcn sequential
./flexible_runner.sh
```

### Combined Workflows

| Preset | Workflow | Description |
|--------|----------|-------------|
| **`dep`** | `dcn.py` + `ms.py` → `comb_ms_dcn.py` | Dependent analysis workflow |
| **`indep`** | `dcn_indep.py` + `ms_indep.py` → `comb_ms_dcn_indep.py` | Independent analysis workflow |
| **`adapt`** | `dcn_adapt.py` + `ms_adapt.py` → `comb_ms_dcn_adapt.py` | Adaptive analysis workflow |

**Examples:**
```bash
# Run dependent workflow: dcn.py + ms.py in parallel, then combined analysis
./quick_switch.sh dep dependency
./flexible_runner.sh

# Run independent workflow with sequential execution
./quick_switch.sh indep sequential
./flexible_runner.sh

# Run adaptive workflow
./quick_switch.sh adapt dependency
./flexible_runner.sh
```

### Custom Experiments

**Interactive setup:**
```bash
./quick_switch.sh custom

# You'll be prompted for:
# - Script 1 path: your_folder/script1.py
# - Script 2 path: your_folder/script2.py  
# - Script 3 path: your_folder/script3.py
# - Job names for each script

./flexible_runner.sh
```

**Manual configuration:**
Edit `experiment_config.sh` directly and uncomment the custom section:
```bash
# --- CUSTOM EXPERIMENTS ---
EXPERIMENT_TYPE="custom"
SCRIPT_1="your_folder/analysis.py"
SCRIPT_2="your_folder/visualization.py"
SCRIPT_3="your_folder/report.py"
JOB_NAME_1="analysis_job"
JOB_NAME_2="viz_job"
JOB_NAME_3="report_job"
```

---

## Execution Modes

| Mode | Behavior | Best For |
|------|----------|----------|
| **`parallel`** | All scripts run simultaneously | When you have enough resources and scripts are independent |
| **`sequential`** | Scripts run one after another | Limited resources or when order matters |
| **`dependency`** | Scripts 1&2 run in parallel, then script 3 | When script 3 needs results from scripts 1&2 |

**Examples:**
```bash
# Parallel execution (default for basic experiments)
./quick_switch.sh ms parallel

# Sequential execution
./quick_switch.sh dcn sequential

# Dependency execution (default for combined workflows)
./quick_switch.sh dep dependency
```

---

## Normal Operation Guide

### Step 1: Choose Your Experiment

```bash
# Choose from presets
./quick_switch.sh ms parallel      # Microstate experiments
./quick_switch.sh dcn sequential   # DCN experiments  
./quick_switch.sh dep dependency   # Combined dependent workflow
./quick_switch.sh indep dependency # Combined independent workflow
./quick_switch.sh adapt dependency # Combined adaptive workflow
./quick_switch.sh custom           # Custom setup
```

This updates your `experiment_config.sh` with the chosen preset.

### Step 2: Run the Experiments

```bash
./flexible_runner.sh
```

This will:
- Validate your scripts exist
- Check your conda environment
- Submit jobs to the queue
- Start a worker to process jobs
- Show you monitoring commands

**Expected output:**
```
Flexible Experiment Runner
==========================
Configuration loaded: microstate experiments
Execution mode: parallel
Working directory: /your/path/Code
...
✓ All scripts found
✓ Conda environment 'myvenv' is available

Submitting jobs...
=== PARALLEL EXECUTION ===
1. Submitting: ms_dependent (subject_dependent/ms.py)
2. Submitting: ms_independent (subject_independent/ms_indep.py)  
3. Submitting: ms_adaptive (subject_adaptive/ms_adapt.py)

Starting worker with 3 parallel jobs...
Worker started in background (PID: 12345)
```

### Step 3: Monitor Progress

```bash
# Quick status check
./tmux_scheduler.sh status

# Watch logs in real-time
./tmux_scheduler.sh logs ms_dependent

# Monitor a job live (enter its tmux session)
./tmux_scheduler.sh monitor ms_dependent
```

---

## Monitoring & Management

### Check Job Status

```bash
# View all job statuses
./tmux_scheduler.sh status
```

**Sample output:**
```
=== JOB STATUS ===
JOB_NAME                 STATUS          JOB_ID     GPU  LAST_UPDATE
ms_dependent            RUNNING         12345      -    2024-07-30 10:30:00
ms_independent          RUNNING         12346      1    2024-07-30 10:30:05
ms_adaptive             QUEUED          12347      -    2024-07-30 10:30:00
```

**Status meanings:**
- **RUNNING**: Job is actively executing
- **QUEUED**: Job is waiting for worker to start it
- **COMPLETED**: Job finished successfully
- **FAILED**: Job crashed or returned error code
- **KILLED**: Job was manually terminated

### View Logs

```bash
# View complete logs for a job
./tmux_scheduler.sh logs ms_dependent

# Watch logs in real-time (tail -f style)
tail -f ../.logs/ms_dependent_*.log

# View most recent log entries
tail -20 ../.logs/ms_dependent_*.log
```

### Monitor Jobs Live

```bash
# Enter the job's tmux session to watch it run
./tmux_scheduler.sh monitor ms_dependent
```

**To exit monitor mode:**
1. Press `Ctrl + B`
2. Release both keys  
3. Press `D` (detach)

**Alternative exit methods:**
- `Ctrl + C` (may work in some cases)
- Type `exit` and press Enter (will kill the job - use carefully!)

### Queue Management

```bash
# View current job queue
./tmux_scheduler.sh queue

# Clean completed jobs from status display
./tmux_scheduler.sh clean
```

---

## Troubleshooting & Problem Handling

### Common Issues

#### **Issue: "Script not found"**
```bash
# Check if your scripts exist
ls -la subject_dependent/ms.py
ls -la subject_independent/ms_indep.py

# Verify you're in the correct directory
pwd  # Should be in your Code/ directory
```

#### **Issue: "Conda environment not found"**
```bash
# List available environments
conda env list

# Create the environment if missing
conda create -n myvenv python=3.11 -y
conda activate myvenv
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

#### **Issue: Jobs stuck in QUEUED status**
```bash
# Check if worker is running
ps aux | grep tmux_scheduler

# If no worker found, start one
nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &

# Check worker logs
tail -f worker.log
```

#### **Issue: Job shows RUNNING but tmux session doesn't exist**
```bash
# List all tmux sessions
tmux list-sessions

# Clean up stale job statuses
./tmux_scheduler.sh clean

# Check system resources
htop  # or top
df -h  # Check disk space
```

### Emergency Procedures

#### **Stop All Jobs Immediately**

```bash
# Method 1: Kill specific jobs
./tmux_scheduler.sh kill ms_dependent
./tmux_scheduler.sh kill ms_independent
./tmux_scheduler.sh kill ms_adaptive

# Method 2: Kill all tmux sessions (NUCLEAR OPTION)
tmux kill-server

# Method 3: Stop worker and clean up
pkill -f tmux_scheduler
./tmux_scheduler.sh clean
```

#### **System Recovery**

```bash
# 1. Stop everything
pkill -f tmux_scheduler
tmux kill-server

# 2. Clean up status files
rm -f .job_queue .job_status

# 3. Start fresh
./quick_switch.sh ms parallel
./flexible_runner.sh
```

#### **Disk Space Issues**

```bash
# Check log directory size
du -sh ../.logs/

# Clean old logs (older than 7 days)
find ../.logs/ -name "*.log" -mtime +7 -delete

# Clean completed job statuses
./tmux_scheduler.sh clean
```

### Worker Management

#### **Check Worker Status**
```bash
# Check if worker is running
ps aux | grep tmux_scheduler | grep worker

# Check worker logs  
tail -f worker.log

# Check worker performance
cat worker.log | grep "Currently running"
```

#### **Restart Worker with Different Settings**
```bash
# Stop current worker
pkill -f tmux_scheduler

# Start with 1 job at a time (sequential)
nohup ./tmux_scheduler.sh worker 1 > worker.log 2>&1 &

# Start with 5 parallel jobs (high performance)
nohup ./tmux_scheduler.sh worker 5 > worker.log 2>&1 &

# Check new worker status
./tmux_scheduler.sh status
```

#### **Worker Not Processing Queue**
```bash
# Check queue contents
./tmux_scheduler.sh queue

# Check if jobs are properly formatted
cat .job_queue

# Restart worker
pkill -f tmux_scheduler
nohup ./tmux_scheduler.sh worker 3 > worker.log 2>&1 &
```

---

## Advanced Usage

### GPU Assignment

```bash
# Assign specific GPUs to jobs
./tmux_scheduler.sh submit subject_dependent/ms.py "gpu_job1" myvenv 0    # Use GPU 0
./tmux_scheduler.sh submit subject_independent/ms_indep.py "gpu_job2" myvenv 1    # Use GPU 1

# Check GPU usage
nvidia-smi

# Auto-assign GPUs (leave GPU parameter empty)
./tmux_scheduler.sh submit subject_adaptive/ms_adapt.py "auto_gpu_job" myvenv
```

### Environment Management

```bash
# Use different conda environments
./tmux_scheduler.sh submit my_script.py "job1" pytorch_env
./tmux_scheduler.sh submit my_script.py "job2" tensorflow_env

# Use Python virtual environments
./tmux_scheduler.sh submit my_script.py "job3" .venv
./tmux_scheduler.sh submit my_script.py "job4" /path/to/my_venv

# Run without any environment
./tmux_scheduler.sh submit my_script.py "job5"
```

### Custom Configurations

Edit `experiment_config.sh` for complex setups:

```bash
# Example: Mixed environments and GPU assignments
SCRIPT_1="analysis/data_prep.py"
SCRIPT_2="training/model_train.py"  
SCRIPT_3="evaluation/test_model.py"
JOB_NAME_1="data_prep"
JOB_NAME_2="training"
JOB_NAME_3="evaluation"
CONDA_ENV="myvenv"
GPU_1=""      # No GPU for data prep
GPU_2="0"     # GPU 0 for training
GPU_3="1"     # GPU 1 for evaluation
```

---

## File Structure

```
Code/
├── tmux_scheduler.sh           # Core job scheduler
├── experiment_config.sh        # Experiment configuration
├── flexible_runner.sh          # Universal job runner
├── quick_switch.sh            # Preset switcher
├── .job_queue                 # Job queue (auto-generated)
├── .job_status               # Job status tracking (auto-generated)
├── worker.log                # Worker process log
├── subject_dependent/        # Your experiment scripts
│   ├── ms.py
│   └── dcn.py
├── subject_independent/
│   ├── ms_indep.py
│   └── dcn_indep.py
├── subject_adaptive/
│   ├── ms_adapt.py
│   └── dcn_adapt.py
├── combined/                 # Combined analysis scripts
│   ├── comb_ms_dcn.py
│   ├── comb_ms_dcn_indep.py
│   └── comb_ms_dcn_adapt.py
└── lib/                     # Your libraries

../.logs/                    # Log directory
├── ms_dependent_20240730_103000.log
├── ms_independent_20240730_103005.log
└── ms_adaptive_20240730_103010.log
```

---

## Safety Features

- **No sudo required**: Runs entirely in user space
- **SSH disconnect safe**: Jobs continue running after disconnect
- **Automatic cleanup**: Completed jobs are tracked and cleaned
- **Error handling**: Failed jobs are marked and logged
- **Resource protection**: Worker limits prevent system overload
- **Log preservation**: All output is saved with timestamps
- **Safe termination**: Jobs can be killed without affecting others
- **Backup configs**: Configuration changes are backed up

---

## Quick Reference Commands

```bash
# Setup
chmod +x *.sh

# Switch experiments
./quick_switch.sh ms parallel
./quick_switch.sh dep dependency

# Run experiments  
./flexible_runner.sh

# Monitor
./tmux_scheduler.sh status
./tmux_scheduler.sh monitor JOB_NAME
./tmux_scheduler.sh logs JOB_NAME

# Control
./tmux_scheduler.sh kill JOB_NAME
pkill -f tmux_scheduler  # Stop worker

# Exit monitor mode
Ctrl+B, then D
```

---

**Happy experimenting!** 🚀

For questions or issues, check the troubleshooting section or examine the log files in `../.logs/` directory.