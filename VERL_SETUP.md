# veRL Setup Guide

## Quick Setup (Same Pattern as MLGym)

You're already using MLGym as a local editable dependency. Do the same for veRL:

```bash
# 1. Clone veRL to a sibling directory (next to MLGym and air-agent)
cd /home/ubuntu/MLScientist
git clone https://github.com/volcengine/verl.git

# Your structure will be:
# /home/ubuntu/MLScientist/
#   ├── MLGym/          (already here)
#   ├── air-agent/      (already here)
#   └── verl/           (new)

# 2. Add veRL to air-agent dependencies
cd air-agent
uv add verl --editable --path ../verl

# That's it! uv will add it just like MLGym
```

## What This Does

The `uv add` command will:
1. Add `verl` to your `dependencies` list in `pyproject.toml`
2. Add the local path to `[tool.uv.sources]` section
3. Install it in editable mode (changes to veRL code take effect immediately)

Your `pyproject.toml` will look like:

```toml
[project]
dependencies = [
    "accelerate>=1.12.0",
    "bitsandbytes>=0.49.1",
    "mlgym",
    "peft>=0.18.1",
    "trl>=0.26.2",
    "verl",  # ← Added
]

[tool.uv.sources]
mlgym = { path = "../MLGym", editable = true }
verl = { path = "../verl", editable = true }  # ← Added
```

## Alternative: Install from PyPI

If you don't want to clone the repo:

```bash
# Just install the released version
uv add verl

# Or specific version
uv add "verl>=0.2.0"
```

But editable local install is better for:
- Debugging
- Contributing fixes back
- Customizing for your use case

## Verify Installation

```bash
cd /home/ubuntu/MLScientist/air-agent
uv run python -c "import verl; print(verl.__version__)"
```

## Dependencies veRL Needs

veRL requires:
- Python 3.10+
- PyTorch with CUDA
- vLLM (for inference backend)
- Ray (for distributed training)

These will be auto-installed by uv when you add veRL.

### If vLLM Installation Fails

vLLM needs specific CUDA version. If you get errors:

```bash
# Check your CUDA version
nvcc --version

# Install vLLM with correct CUDA version
uv pip install vllm-cuda121  # for CUDA 12.1
# or
uv pip install vllm-cuda118  # for CUDA 11.8
```

## Using veRL in Your Code

After setup, just import normally:

```python
# In air/verl_orchestrator.py
from verl import GRPOTrainer
from verl.trainer.ppo.grpo_config import GRPOConfig

# Works just like MLGym!
from mlgym.environment.env import EnvironmentArguments
```

## Running Your Code

```bash
# Activate the uv environment
cd /home/ubuntu/MLScientist/air-agent
uv run python -m air.verl_orchestrator
```

## Troubleshooting

### "Module not found: verl"
```bash
# Re-sync dependencies
uv sync
```

### "veRL requires Ray"
```bash
uv add ray
```

### "vLLM CUDA mismatch"
```bash
# Check CUDA
python -c "import torch; print(torch.version.cuda)"

# Reinstall vLLM for your CUDA version
uv pip uninstall vllm
uv pip install vllm-cuda121  # Match your CUDA version
```

### Want to update veRL?
```bash
cd /home/ubuntu/MLScientist/verl
git pull origin main

# Changes auto-reflected (editable install!)
```

## Full Commands (Copy-Paste)

```bash
# Complete setup from scratch
cd /home/ubuntu/MLScientist

# Clone veRL
git clone https://github.com/volcengine/verl.git

# Add to air-agent
cd air-agent
uv add verl --editable --path ../verl

# Sync all dependencies
uv sync

# Verify
uv run python -c "import verl; import mlgym; print('✓ All imports work!')"

# Run training
uv run python -m air.verl_orchestrator
```

## Why This Approach?

✅ **Consistent with MLGym**: Same pattern you already use  
✅ **Editable**: Modify veRL source if needed  
✅ **Isolated**: Each project can use different versions  
✅ **Fast**: uv is much faster than pip  
✅ **Clean**: No global pip installs messing up your system  

