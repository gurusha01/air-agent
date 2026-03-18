#!/bin/bash
export HF_HOME=/scratch/jarnav/hf_cache
export HUGGINGFACE_HUB_VERBOSITY=info
echo "Downloading Qwen3-8B..."
/home/jarnav/MLScientist/air-agent/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-8B',
    cache_dir='/scratch/jarnav/hf_cache',
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*']
)
print('Download complete')
"
