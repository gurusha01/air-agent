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

### Issue 7: MLGym Commands Not Found (`open`, `edit`, etc.)

**Symptom:**
```
/bin/bash: line 15147: open: command not found
```

**Root Cause:**
MLGym's custom commands (`open`, `edit`, `validate`, `create`, etc.) are shell functions defined in `tools/defaults.sh`. These need to be loaded into the container, but our wrapper wasn't doing this.

**Solution:**
Added `_load_commands()` method to load shell functions:

```python
def _load_commands(self, env: MLGymEnv) -> None:
    command_files_paths = [
        "tools/defaults.sh",      # open, goto, scroll, edit
        "tools/search.sh",        # search, find_file
        "tools/edit_linting.sh",  # edit with linting
    ]
    # Load and source each file in container
    env.add_commands(command_files)
```

**File:** `air/mlgym_env.py`, `_load_commands()` method

---

### Issue 8: jq Errors in `open` Command

**Symptom:**
```
jq: error: syntax error, unexpected '/', expecting $end at /2
```

**Root Cause:**
The `open` command uses `jq` for formatting and relies on `$WINDOW` environment variable. If `WINDOW` is not set, `$WINDOW/2` becomes `/2`, causing jq syntax errors.

**Solution:**
Set required environment variables before loading commands:

```python
env_vars = {
    "WINDOW": "100",
    "OVERLAP": "2",
    "CURRENT_LINE": "0",
    "CURRENT_FILE": "",
}
for var, value in env_vars.items():
    env.communicate(f"export {var}={value}")
```

**File:** `air/mlgym_env.py`, `_load_commands()` method

---

### Issue 9: Files in `data/` Subdirectory

**Symptom:**
```
File train.csv not found
```

**Root Cause:**
Data files are in `data/` subdirectory, not workspace root. Agent was trying `open train.csv` instead of `open data/train.csv`.

**Solution:**
Updated system prompt to inform model about workspace structure:

```
WORKSPACE STRUCTURE:
- data/train.csv - Training data
- data/test.csv - Test data
- evaluate.py - Evaluation script (read-only)
```

**File:** `air/mlgym_env.py`, system prompt in `_generate_init_prompt()`

---

### Issue 10: Trajectories Ending Early (1-3 turns instead of max_turns)

**Symptom:**
```
exit_status: "submission_not_found (max_steps)"
```
Trajectories only have 1-3 turns saved, but exit_status says max_steps was reached.

**Background:**
There are **two different step counters**:
1. **Our `max_turns=20`** - How many turns we want the agent to take
2. **MLGym's internal `max_steps`** - A safety limit inside the Docker container

We had set MLGym's `max_steps = max_turns * 2 = 40`.

**Root Cause:**
With **branching trajectory strategy** and `rollouts_per_example=8`:
- At each turn, 8 different rollouts are generated from the same prompt
- All 8 rollouts share the **same MLGym container**
- Each rollout calls `env.step()`, incrementing MLGym's internal counter

So the math:
```
Turn 1: 8 rollouts × 1 step each = 8 steps used
Turn 2: 8 rollouts × 1 step each = 8 steps used
Turn 3: 8 rollouts × 1 step each = 8 steps used
Turn 4: 8 rollouts × 1 step each = 8 steps used
Turn 5: 8 rollouts × 1 step each = 8 steps used
----------------------------------------
Total after 5 turns: 40 steps → MLGym says "done!"
```

MLGym hit its 40-step limit after only ~5 turns, returning `done=True` and stopping early.

**Visual Summary:**
```
Before fix:
┌─────────────────────────────────────────┐
│ MLGym max_steps = 40                    │
│ 8 rollouts × 5 turns = 40 steps         │
│ → Episode ends at turn 5!               │
└─────────────────────────────────────────┘

After fix:
┌─────────────────────────────────────────┐
│ MLGym max_steps = 400                   │
│ 8 rollouts × 20 turns = 160 steps       │
│ → Plenty of headroom, episode runs full │
└─────────────────────────────────────────┘
```

**Solution:**
Increase MLGym's `max_steps` to account for parallel rollouts:

```python
env_args = EnvironmentArguments(
    max_steps=self.max_turns * 20,  # Generous buffer for parallel rollouts
    ...
)
```

**File:** `air/mlgym_env.py`, `_create_new_env()` method

---

### Issue 11: Context Length Limit (Rollouts Stop at Turn 3)

**Symptom:**
```
VLLMValidationError: This model's maximum context length is 8192 tokens.
However, your request has 9358 input tokens.
```
Rollouts only reach turn 3 (num_messages: 3, 5, 7) despite max_turns=20.

**Root Cause:**
The `open data/train.csv` command returns ~7000 characters of CSV data. With:
- System prompt (~500 tokens)
- User prompt (~100 tokens)
- Assistant response (~50-100 tokens per turn)
- Observation (~2000+ tokens for CSV file content)

After 2-3 turns, the accumulated context exceeds 8192 tokens.

**Sequence:**
```
Turn 1: ~2000 tokens (prompt + initial response)
Turn 2: ~4500 tokens (+ open data/train.csv observation ~2500 tokens)
Turn 3: ~7000 tokens (+ another large observation)
Turn 4: ~9500 tokens → EXCEEDS 8192 LIMIT → 400 Bad Request
```

**Solution:**
Increase context length. Qwen3-4B supports up to 262K tokens (rope_scaling). Set:

```toml
# In config file
seq_len = 32768  # For trainer padding/batching
max_model_len = 32768  # For vLLM inference server
```

**Config:** `configs/mlgym/rl_full.toml`, lines 28 and 83

**Memory Considerations:**
- 8K context: ~3-5 GB KV cache (fits easily)
- 32K context: ~15-20 GB KV cache (fits on 24GB A10G at ~17GB peak)
- Model supports 256K but that would exceed GPU memory

---

### Issue 12: Trajectories Not Saving When max_turns Reached

**Symptom:**
Trajectories only have 10, 11, 16 turns saved despite max_turns=20 being set.

**Root Cause:**
Trajectories were only being saved when MLGym returned `done=True` from `step()`. However, most episodes end via `check_done()` reaching max_turns, not via MLGym's internal completion. The `check_done()` path wasn't triggering a save.

**Sequence:**
```
1. Agent takes turns 1-20
2. check_done() sees turn 20, returns True
3. Episode ends but _save_trajectory() was never called!
4. Trajectory is lost (or saved with fewer turns from earlier partial saves)
```

**Solution:**
Add trajectory save in `check_done()` when returning True:

```python
def check_done(self, state: dict) -> bool:
    traj_len = len(state.get("trajectory", []))
    done = traj_len >= self.max_turns
    if done:
        logger.info(f"[check_done] max_turns reached, traj_len={traj_len}")
        self._save_trajectory(state, {"exit_reason": "max_turns"})
    return done
```

Also added `_trajectory_saved` flag to prevent double-saving:

```python
def _save_trajectory(self, state: dict, metrics: dict) -> None:
    if state.get("_trajectory_saved"):
        logger.debug(f"[_save_trajectory] Already saved, skipping")
        return
    state["_trajectory_saved"] = True
    # ... rest of save logic
```

**File:** `air/mlgym_env.py`, `check_done()` and `_save_trajectory()` methods

---

### Issue 13: validate/submit Commands Not Found After Reset

**Symptom:**
```
/bin/bash: line 853: validate: command not found
```

Even after fixing Issue 7 (loading commands), validate still fails.

**Root Cause:**
Two problems:
1. `validate.sh` and `submit.sh` weren't in the `command_files_paths` list
2. `env.reset()` clears all shell functions, but we only loaded commands once at container creation

**Sequence:**
```
1. Create container, load commands (including validate)
2. Episode completes
3. env.reset() is called → all shell functions are cleared!
4. New episode starts, agent tries validate → command not found
```

**Solution:**
1. Add validate and submit scripts to the command files:

```python
command_files_paths = [
    "tools/defaults.sh",      # open, goto, scroll, edit, create
    "tools/search.sh",        # search, find_file
    "tools/edit_linting.sh",  # edit with linting
    "tools/validate.sh",      # validate - evaluate solution
    "tools/submit.sh",        # submit - submit final solution
]
```

2. Reload commands after every reset:

```python
# In _get_or_create_env(), after reset:
env.communicate("cd /home/agent")
env.reset()
self._load_commands(env)  # Reload commands after reset clears them
```

**File:** `air/mlgym_env.py`, `_load_commands()` and `_get_or_create_env()` methods

---

### Issue 14: Wrong Policy Step Labels in Trajectory Filenames

**Symptom:**
After training step 0 completes, trajectory counts show:
```
π_0: 27 files
π_1: 5 files
π_2: 3 files
```

Expected: All files from step 0's rollouts should be labeled π_0.

**Root Cause:**
Policy step was being recorded at **save time**, not at **rollout start time**:

```python
# WRONG: Gets policy at save time
policy_step = self._get_current_policy_step()
filename = f"{task_name}_pi{policy_step}_{timestamp}.json"
```

With interleaved training:
1. Rollouts start with π_0
2. Some rollouts complete quickly, save as π_0
3. Training step 0 completes, weights update → now π_1
4. Remaining rollouts (still from π_0) complete and save as π_1 (wrong!)

**Solution:**
Record policy step at rollout START by storing in state dict:

```python
def env_response(self, ...):
    # Record policy step when rollout STARTS (first call for this episode)
    if "_policy_step_at_start" not in state:
        state["_policy_step_at_start"] = self._get_current_policy_step()
        logger.info(f"[DEBUG] Rollout starting with policy π_{state['_policy_step_at_start']}")
    # ...

def _save_trajectory(self, state: dict, metrics: dict) -> None:
    # Use policy step from start, not current
    policy_step = state.get("_policy_step_at_start", self._get_current_policy_step())
    filename = f"{task_name}_pi{policy_step}_{timestamp}.json"
```

**File:** `air/mlgym_env.py`, `env_response()` and `_save_trajectory()` methods

---

### Issue 15: Multi-Command Output Causes Container Hang (Deadlock)

**Symptom:**
Training stalls with `step()` never returning. Last log shows:
```
[DEBUG] Got env, calling step()...
```
No follow-up log. Orchestrator hangs indefinitely.

**Root Cause:**
The model outputs multiple commands in a single response:
```
edit 1:1
#!/usr/bin/env python3
end_of_edit
python train_and_predict.py
validate
```

When this multi-command string is sent to `env.step()`:
1. MLGym's `communicate()` sends it to the container's bash
2. Bash starts executing commands sequentially
3. The python script might take a long time or hang
4. `communicate()` waits indefinitely for all commands to finish
5. Deadlock: orchestrator waiting for step(), step() waiting for bash

**Solution:**
Detect multi-command outputs and reject them with a -0.5 reward:

```python
def _extract_command(self, raw_output: str) -> tuple[str | None, bool]:
    """Returns (command, is_multi_command)"""
    # Count commands in output
    # If multiple commands detected, return (None, True)
    ...

async def env_response(self, messages, state, **kwargs):
    action, is_multi_command = self._extract_command(raw_action)

    if is_multi_command:
        state["last_tool_success"] = False  # Triggers -0.5 reward
        return [{"role": "user", "content": "Error: Multiple commands detected. Please output only ONE command at a time."}]
```

This teaches the model to output single commands through the reward signal.

**Additional Safety:**
Added `_step_with_timeout()` wrapper (60s timeout) to prevent infinite hangs from other causes (e.g., python script running forever).

**File:** `air/mlgym_env.py`, `_extract_command()` and `env_response()` methods

---

### Issue 16: Edit Command Fails with Bash Syntax Errors

**Symptom:**
Model's `edit` commands fail with bash syntax errors:
```
bash: line 7: syntax error near unexpected token '('
bash: line 7: `train_df = pd.read_csv('data/train.csv')'
```

Python code like `pd.read_csv()` is being interpreted as bash commands.

**Root Cause:**
MLGym's `_check_syntax()` function in `mlgym/tools/commands.py` runs `bash -n` on ALL input:

```python
def _check_syntax(code: str) -> tuple[str, int]:
    # ... runs subprocess(['bash', '-n'], input=code)
```

When the model uses `edit` to write Python code:
```
edit 1:10
import pandas as pd
train_df = pd.read_csv('data/train.csv')
end_of_edit
```

MLGym parses this and passes the Python code to `_check_syntax`, which runs `bash -n` on it. Bash interprets the Python syntax as invalid shell commands.

**Solution:**
Instead of using MLGym's `edit` command, use heredoc file writing which bash handles correctly:

```python
system_prompt = """
To write a Python file, use this EXACT format:
cat << 'ENDOFFILE' > train_and_predict.py
import pandas as pd
# your code here
ENDOFFILE
"""
```

The heredoc approach works because:
1. `cat << 'ENDOFFILE'` tells bash to read until the delimiter
2. Content between heredoc markers is treated as literal text, not parsed
3. The file is written directly without syntax checking

**Updated `_extract_command()` to handle heredoc:**
```python
# Heredoc pattern: cat << 'EOF' > file.py
heredoc_pattern = r'^cat\s+<<\s*[\'"]?(\w+)[\'"]?\s*>\s*\S+'
if re.match(heredoc_pattern, action, re.MULTILINE):
    # Extract entire heredoc block including content until delimiter
    ...
```

**File:** `air/mlgym_env.py`, system prompt and `_extract_command()` method

---

### Issue 17: Episode-End Reward Never Fires

**Symptom:**
W&B `metric_improvement` always 0. Episode-end reward (+1/-0.2/-1) never applied despite `episode_done` check in reward function.

**Root Cause:**
`state["episode_done"]` is set in `env_response()` from MLGym's `done` flag (line 748):
```python
state["episode_done"] = done  # done from MLGym's step() return
```
But episodes end via `check_done()` returning True (max_turns reached), which did NOT set `episode_done`. MLGym only returns `done=True` when the agent calls `submit`, which basically never happened.

**Impact:**
This bug affected ALL experiments 1.1-1.4. The only active reward signal was per-step tool call penalty.

**Solution:**
Set `state["episode_done"] = True` in `check_done()` before returning True, on all 3 exit paths (submit, exit_keyword, max_turns). Also added `submit` instruction to system prompt so the model actually calls it.

**File:** `air/mlgym_env.py`, `check_done()` method

---

### Issue 18: Zero-Variance Reward Kills GRPO Learning

**Symptom:**
Exp 1.5 completed 200 steps but model showed no improvement. Reward was -0.2688 at step 0 and -0.2000 at step 199.

**Root Cause:**
Threshold-based reward: 92% of trajectories got the exact same reward (-0.2, "mediocre improvement"). GRPO computes `advantage = reward - mean(batch)`. When all 16 trajectories in a batch get -0.2, advantages are all 0 → zero gradient → no learning. 26% of batches had literally zero gradient.

**Solution:**
Switched to continuous reward in [-1, 1]:
```python
improvement = final_score - baseline_score
imp_reward = improvement * 5.0 - 0.5  # 0% → -0.5, 10% → 0.0, 20% → +0.5
format_penalty = -(error_count / total_steps) * 0.5
reward = clip(imp_reward + format_penalty, -1, 1)
```
This ensures every batch has non-zero variance → every step produces gradient.

**Lesson:** For GRPO, reward variance within batches matters more than reward scale. Continuous rewards always produce gradient; threshold rewards often don't.

**File:** `air/mlgym_env.py`, `compute_delta_reward()`

---

### Issue 19: Policy Collapse After ~100 Steps (Exp 1.6)

**Symptom:**
Model peaked at steps 80-100 (mean accuracy 0.890, max 0.986), then degraded to mean 0.852 by step 199. Sequence length dropped from 5500 → 1300 tokens, indicating the model learned to generate degenerate short responses.

**Root Cause (suspected):**
- Learning rate 1e-5 may be too high for 200 steps of GRPO
- No KL penalty to anchor policy to the reference model
- Model drifts too far from the base policy and collapses

**Status:** Not yet fixed. Possible solutions:
1. Lower learning rate (e.g., 5e-6 or 1e-6)
2. Add KL penalty term to reward
3. Use fewer training steps or early stopping
4. Increase batch size for more stable gradient estimates

**File:** `configs/mlgym/rl_full.toml`

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
