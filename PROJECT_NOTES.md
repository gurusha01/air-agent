# Project Notes: MLGym + Prime-RL Integration

This document describes the issues encountered while integrating MLGym with prime-rl and how to solve them.

## Overview

The goal is to train RL agents on MLGym tasks (ML research problems) using prime-rl's distributed training infrastructure. This requires:

1. A verifiers-compatible environment wrapper (`air/mlgym_env.py`)
2. Proper GPU allocation (inference, trainer, and Docker container need separate GPUs)
3. Container lifecycle management (MLGym runs tasks in Docker)

---

## Issues and Solutions

### Issue 1: Container "pip folder not found" Error

**Symptom:**
```
RuntimeError: Failed to install flake8 (lint library): The folder you are executing pip from can no longer be found.
```

Containers fail after the first episode when `reset()` is called.

**Root Cause:**
MLGym's `reset()` method does `rm -rf /home/agent/workspace` while the shell's current working directory is inside that workspace. This invalidates the shell's cwd, causing all subsequent commands (including `pip install flake8`) to fail.

**Sequence:**
1. First `reset()` works - workspace doesn't exist yet
2. At end of `reset()`, shell cwd is set to `/home/agent/workspace` (line 362 in env.py)
3. Subsequent `reset()` calls `_setup_workspace()` which does `rm -rf workspace`
4. Shell's cwd becomes invalid → pip fails

**Solution:**
Add `cd /home/agent` before calling `reset()` in `mlgym_env.py`:

```python
# In _get_or_create_env(), before reset:
env.communicate("cd /home/agent")  # Move to safe directory
env.reset()
```

**File:** `air/mlgym_env.py`, line ~301

---

### Issue 2: Verifiers Completion Format

**Symptom:**
Worker subprocesses crash with exit code 1 before any logging occurs.

**Root Cause:**
The `completion` field in verifiers trajectories is a **list of messages**, not a string.

**Wrong:**
```python
last = state["trajectory"][-1].get("completion", "")
if "done" in last.lower():  # ERROR: completion is a list!
```

**Correct:**
```python
completion = state["trajectory"][-1].get("completion", [])
if completion:
    last_msg = [m for m in completion if m.get("role") == "assistant"]
    if last_msg:
        content = last_msg[-1].get("content", "")
```

---

### Issue 3: GPU Memory Conflicts

**Symptom:**
```
ValueError: Free memory on device cuda:0 (13.33/22.07 GiB) on startup is less than desired GPU memory utilization
```

**Root Cause:**
Old vLLM processes from previous runs still hold GPU memory.

**Solution:**
Kill all GPU processes before starting a new run:

```bash
# Check GPU processes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Kill all
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill -9 {}
```

---

### Issue 4: Stale Processes Causing Trainer Hang

**Symptom:**
Trainer shows "Starting training loop" but never completes any steps, even though orchestrator is generating rollouts.

**Root Cause:**
Processes from previous runs (orchestrator, trainer) are still running and interfering with the new run.

**Solution:**
Clean up all processes before starting:

```bash
pkill -9 -f "orchestrator"
pkill -9 -f "prime_rl.trainer"
pkill -9 -f "torchrun"
pkill -9 -f "vllm"
rm -rf outputs/run_default outputs/logs outputs/torchrun
```

---

### Issue 5: Model Output Format

**Symptom:**
Model outputs explanatory text instead of just commands, causing MLGym to fail parsing.

**Root Cause:**
The model (Qwen3) outputs thinking and explanations, not just the raw command.

**Solution:**
Added `_extract_command()` method to parse commands from model output:

```python
def _extract_command(self, raw_output: str) -> str:
    """Extract executable command from model output."""
    # Remove thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove code blocks
    text = re.sub(r'```[a-z]*\n?', '', text)
    # Look for known command patterns (open, edit, python, validate, etc.)
    ...
```

---

### Issue 6: Working Directory for Subprocesses

**Symptom:**
Worker subprocesses can't find MLGym task configs.

**Root Cause:**
prime-rl spawns worker subprocesses that inherit the parent's cwd. If parent is not in MLGym directory, relative paths fail.

**Solution:**
Run training from the MLGym directory:

```bash
cd /home/ubuntu/MLScientist/MLGym
source /home/ubuntu/MLScientist/air-agent/.venv/bin/activate
uv run --project /home/ubuntu/MLScientist/air-agent rl @ /home/ubuntu/MLScientist/air-agent/configs/mlgym/rl_debug.toml
```

Or use absolute paths in the code:

```python
MLGYM_PATH = Path("/home/ubuntu/MLScientist/MLGym").resolve()
os.chdir(MLGYM_PATH)
```

---

## Debugging Tips

### Enable Worker Logs

Add to config:
```toml
[orchestrator.log]
env_worker_logs = true
level = "DEBUG"
```

Logs appear at: `outputs/run_default/logs/env_workers/{env_name}.log`

### Check Orchestrator Logs

```bash
tail -f outputs/logs/orchestrator.stdout
```

### Check Trainer Logs

```bash
tail -f outputs/logs/trainer.stdout
```

### Check Container Status

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

---

## GPU Allocation (8x A10G Example)

```
GPU 0: vLLM inference server
GPU 1: (often occupied by other processes - skip)
GPUs 2-6: FSDP trainer (5 GPUs)
GPU 7: MLGym Docker container
```

Config:
```toml
inference_gpu_ids = [0]
trainer_gpu_ids = [2, 3, 4, 5, 6]

[[orchestrator.env]]
args = { env_gpu = "7", ... }
```

---

## Replication Steps

1. Clone repositories:
   ```bash
   git clone <air-agent-repo>
   git clone <MLGym-repo>
   git clone <prime-rl-repo>
   ```

2. Install dependencies:
   ```bash
   cd air-agent
   uv sync
   ```

3. Pull MLGym Docker image:
   ```bash
   docker pull aigym/mlgym-agent:latest
   ```

4. Clean GPU processes:
   ```bash
   nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill -9 {}
   ```

5. Run training:
   ```bash
   cd /home/ubuntu/MLScientist/MLGym
   source /home/ubuntu/MLScientist/air-agent/.venv/bin/activate
   uv run --project /home/ubuntu/MLScientist/air-agent rl @ /home/ubuntu/MLScientist/air-agent/configs/mlgym/rl_debug.toml
   ```

6. Monitor:
   ```bash
   # In separate terminals:
   tail -f outputs/logs/trainer.stdout
   tail -f outputs/logs/orchestrator.stdout
   ```
